#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Szu-Wei Fu

"""Training flow of VQVAE """

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import copy
import torch
import torchaudio
import numpy as np
import shutil
import pandas as pd
import torch.nn.functional as F

from scipy.stats import entropy
from scipy.stats import pearsonr, spearmanr

from trainer.eval_dataset import load_IUB, load_Tencent, load_DNS1, load_DNS3, load_VCTK_validSet, load_VCTK_testSet
from trainer.trainerAE import TrainerAE

from pesq import pesq
sys.path.append('./DNSMOS')
from dnsmos_local import ComputeScore
compute_score = ComputeScore('./DNSMOS/ONNX_models/sig_bak_ovr.onnx', './DNSMOS/ONNX_models/model_v8.onnx')

class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)


def resynthesize(enhanced_mag, noisy_inputs, hop_size):
    """Function for resynthesizing waveforms from enhanced mags.
    Arguments
    ---------
    enhanced_mag : torch.Tensor
        Predicted spectral magnitude, should be three dimensional.
    noisy_inputs : torch.Tensor
        The noisy waveforms before any processing, to extract phase.
    Returns
    -------
    enhanced_wav : torch.Tensor
        The resynthesized waveforms of the enhanced magnitudes with noisy phase.
    """

    # Extract noisy phase from inputs
        
    noisy_feats = torch.stft(noisy_inputs, n_fft=512, hop_length=hop_size, win_length=512, 
                             window=torch.hamming_window(512).to('cuda'),
                             center=True,
                             pad_mode="constant",
                             onesided=True,
                             return_complex=False).transpose(2, 1)
            
    noisy_phase = torch.atan2(noisy_feats[:, :, :, 1], noisy_feats[:, :, :, 0])[:,0:enhanced_mag.shape[1],:]
    
    # Combine with enhanced magnitude
    predictions = torch.mul(
        torch.unsqueeze(enhanced_mag, -1),
        torch.cat(
            (
                torch.unsqueeze(torch.cos(noisy_phase), -1),
                torch.unsqueeze(torch.sin(noisy_phase), -1),
            ),
            -1,
        ),
    ).permute(0, 2, 1, 3)
    
    # isft ask complex input
    complex_predictions = torch.complex(predictions[..., 0], predictions[..., 1])
    pred_wavs = torch.istft(input=complex_predictions, n_fft=512, hop_length=hop_size, win_length=512, 
                            window=torch.hamming_window(512).to('cuda'),
                            center=True,
                            onesided=True,
                            length=noisy_inputs.shape[1])
    
    return pred_wavs

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

eps=1e-5
class Trainer(TrainerAE):
    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        super(Trainer, self).__init__(
           steps=steps,
           epochs=epochs,
           data_loader=data_loader,
           model=model,
           optimizer=optimizer,
           scheduler=scheduler,
           config=config,
           device=device,
        )
        
        self.VQVAE_start = config.get('start_steps', {}).get('VQVAE', 0)     
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.exp_dir = './exp/'+self.config["name"]+'/'       
        
        
        ## load evaluation data
        self.vctk_test = load_VCTK_testSet('./VCTK_noisy_testSet_with_scores.pickle') # it is the validation set for QE, and one of the test set for SE
        if self.config['task'] == 'Speech_Enhancement':
            self.hop_size = 128
            self.vctk_Clean_path = self.config['vctk_Clean_path']
            self.dns1 = load_DNS1(self.config['DNS1_test'])
            self.dns3 = load_DNS3(self.config['DNS3_test'])
            self.vctk_valid = load_VCTK_validSet('./VCTK_noisy_validationSet.pickle') # As in MetricGAN-U, using noisy data of speakers (p226 and p287) as validation set           
            self.highest_pesq, self.highest_dnsmos_ovr = 0, 0
        elif self.config['task'] == 'Quality_Estimation':
            self.hop_size = 256      
            self.tencent = load_Tencent(pickle_path ='./Tencent_ind2.pickle', number_test_set = 250)       
            self.iub = load_IUB(pickle_path = './IUB_ind2.pickle', number_test_set = 200)
            self.highest_dnsmos_ovr_CC = 0
        
        # Copy code to the current exp directory for tracing modification
        shutil.copyfile('./trainer/autoencoder.py', self.exp_dir+'autoencoder.py')
        shutil.copyfile('./trainer/trainerAE.py', self.exp_dir+'trainerAE.py')
        shutil.copyfile('./models/VQVAE_models.py', self.exp_dir+'VQVAE_models.py')
        shutil.copyfile('./config/'+self.config["name"]+'.yaml', self.exp_dir+self.config["name"]+'.yaml')
               
    def stft_magnitude(self, x, hop_size, fft_size=512, win_length=512):
        if x.is_cuda:
            x_stft = torch.stft(
                x, fft_size, hop_size, win_length, window=torch.hann_window(win_length).to('cuda'), return_complex=False
            )
        else:
            x_stft = torch.stft(
                x, fft_size, hop_size, win_length, window=torch.hann_window(win_length), return_complex=False
            )   
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        
        return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
    
    def scatter_plot(self, Real_scores, predicted_scores, image_name):
        # Plotting the scatter plot
        plt.scatter(Real_scores, predicted_scores, s=14)
        plt.xlabel('Real_scores')
        plt.ylabel('Predicted_scores')
        
        LCC = pearsonr(Real_scores, predicted_scores)[0]
        SRCC = spearmanr(Real_scores, predicted_scores)[0]
        MSE = np.mean((np.asarray(Real_scores)-np.asarray(predicted_scores))**2)
        
        plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC, SRCC, MSE))
        plt.show()
        plt.savefig(self.exp_dir+image_name, dpi=150)
        plt.clf()
                  
    def cos_loss(self, SP_noisy, SP_y_noisy):  
        SP_noisy_norm = torch.norm(SP_noisy, p=2, dim=-1, keepdim=True)+eps
        SP_y_noisy_norm = torch.norm(SP_y_noisy, p=2, dim=-1, keepdim=True)+eps  
        Cos_frame = torch.sum(SP_noisy/SP_noisy_norm * SP_y_noisy/SP_y_noisy_norm, dim=-1) # torch.Size([B, T, 1])
        
        return -torch.mean(Cos_frame)
        
                                                                                                                  
    def _train_step(self, batch):
        """Single step of training."""
        mode = 'train'
        x_c = batch[0]  # torch.Size([64, 1, 48000])
        scalar = torch.rand((x_c.shape[0],1,1))*1.95+0.05 # to have diverse volume
        sample_max = torch.max(abs(x_c), dim=-1, keepdim=True)[0]
        scalar2 = torch.clamp(scalar, max=1/sample_max)
        
        x = (x_c*scalar2).to(self.device)
        
        # check VQVAE step
        if self.steps < self.VQVAE_start:
            self.VQVAE_train = False
        else:
            self.VQVAE_train = True
            
        if self.steps == self.config['AT_training_start_steps']:
            VQVAE_optimizer_class = getattr(
                torch.optim, 
                self.config['VQVAE_optimizer_type'])
            
            self.optimizer = {
                'VQVAE': VQVAE_optimizer_class(self.model['VQVAE'].parameters(), **self.config['VQVAE_AT_optimizer_params'])}
            
            # start with the model which has the highest pesq on validation set
            print('Load highest pesq model for AT training...')
            self.model["VQVAE"].load_state_dict(torch.load(self.exp_dir + 'checkpoint-pesq='+ str(self.highest_pesq)[0:5]+'.pkl')['model']['VQVAE'])
            self.teacher_model = copy.deepcopy(self.model["VQVAE"])
            
        if self.config['task'] == 'Speech_Enhancement' and self.steps >= self.config['AT_training_start_steps']:
            # Fix the teacher model
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad = False 
            self.teacher_model.quantizer.eval()
            self.teacher_model.eval()
            
            # Fix the quantizer of the student model
            for parameter in self.model["VQVAE"].quantizer.parameters():
                parameter.requires_grad = False
            self.model["VQVAE"].quantizer.eval()
          

        #######################
        #      VQVAE      #
        #######################
        if self.VQVAE_train:     
            gen_loss = 0.0 # initialize VQVAE loss
            x = torch.squeeze(x)
            if self.config['task'] == 'Speech_Enhancement' and self.steps >= self.config['AT_training_start_steps']:
                x.requires_grad = True   # Set requires_grad attribute of input tensor. Important for AT Attack. 
            X = self.stft_magnitude(x, hop_size=self.hop_size) # shape = torch.Size([B, T, F])
            if self.config['input_transform'] == 'log1p':
                X = torch.log1p(X)
            
            # Sec 2.5: step 2 SELF-DISTILLATION WITH ADVERSARIAL TRAINING in the paper
            if self.config['task'] == 'Speech_Enhancement' and self.steps >= self.config['AT_training_start_steps']: 
                with torch.no_grad():
                    z_teacher = self.teacher_model.CNN_1D_encoder(X)
                    teacher_zq, indices_teacher, _, _ = self.teacher_model.quantizer(z_teacher, stochastic=False, update=False)
                
                ###### Step 2-1 adversarial attack start ######
                self.model["VQVAE"].eval()
                z_att = self.model["VQVAE"].CNN_1D_encoder(X)
                zq_att, attack_cross_entropy_loss = self.model["VQVAE"].quantizer(z_att, stochastic=False, update=False, indices=indices_teacher.detach())
                
                # Zero all existing gradients
                self.model["VQVAE"].CNN_1D_encoder.zero_grad()

                # Calculate gradients of model in backward pass
                attack_cross_entropy_loss.backward()
                
                # Get input gradient                     
                adversarial_noise = x.grad.data
                
                power_ratio = torch.norm(adversarial_noise, p=2, dim=-1, keepdim=True)/torch.norm(x, p=2, dim=-1, keepdim=True)
                random_ratio = (self.config['adv_max_epsilon']-self.config['adv_min_epsilon'])*torch.rand((X.shape[0],1), device='cuda')+self.config['adv_min_epsilon']
                
                perturbed_wav = x + (random_ratio/power_ratio) * adversarial_noise # gradient accent
                perturbed_X = self.stft_magnitude(perturbed_wav, hop_size=self.hop_size)
                if self.config['input_transform'] == 'log1p':
                    perturbed_X = torch.log1p(perturbed_X)
                ###### adversarial attack end ######
                
                ###### Step 2-2 Adversarial training start ######
                self.model["VQVAE"].train()
                perturbed_z = self.model["VQVAE"].CNN_1D_encoder(perturbed_X.detach())
                perturbed_zq, noisy_cross_entropy_loss = self.model["VQVAE"].quantizer(perturbed_z, stochastic=False, update=False, indices=indices_teacher.detach())
                perturbed_Y_ = self.model["VQVAE"].CNN_1D_decoder(perturbed_zq.detach())
                
                z = self.model["VQVAE"].CNN_1D_encoder(X.detach())
                zq, clean_cross_entropy_loss = self.model["VQVAE"].quantizer(z, stochastic=False, update=False, indices=indices_teacher.detach()) 
                Y_ = self.model["VQVAE"].CNN_1D_decoder(zq.detach())
                                        
                # save the attacked audio for listening
                if self.steps % 1000 == 0:
                    torchaudio.save(self.exp_dir + 'original.wav', x[0:1,:].cpu(), 16000)
                    torchaudio.save(self.exp_dir + 'attacked.wav', perturbed_wav[0:1,:].cpu(), 16000)
                    print([attack_cross_entropy_loss.item(), noisy_cross_entropy_loss.item()])
                          
                # cross_entropy_loss e.q. 7 in the paper           
                ce_loss = noisy_cross_entropy_loss + clean_cross_entropy_loss
                ce_loss *= self.config["lambda_ce_loss"]
                gen_loss += ce_loss
                self.total_train_loss["train/noisy_cross_entropy_loss"] += noisy_cross_entropy_loss
                self.total_train_loss["train/clean_cross_entropy_loss"] += clean_cross_entropy_loss
                
                # reconstruction loss
                noisy_SP_loss = self.config["lambda_stft_loss"] * self.spectral_convergence_loss(perturbed_Y_, X.detach())
                clean_SP_loss = self.config["lambda_stft_loss"] * self.spectral_convergence_loss(Y_, X.detach())
                gen_loss += (noisy_SP_loss + clean_SP_loss)
                self.total_train_loss["train/noisy_SP_loss"] += noisy_SP_loss
                self.total_train_loss["train/clean_SP_loss"] += clean_SP_loss
            
            # Normal VQVAE training
            else: 
                z = self.model["VQVAE"].CNN_1D_encoder(X)                 
                zq, indices, vqloss, distance = self.model["VQVAE"].quantizer(z, stochastic=False, update=True)        
                Y_ = self.model["VQVAE"].CNN_1D_decoder(zq) 
                
                vqloss *= self.config["lambda_vq_loss"]
                gen_loss += vqloss
                self.total_train_loss["train/vqloss"] += vqloss.item()

                main_loss = self.config["lambda_stft_loss"] * (self.cos_loss(X, Y_) if self.config['cos_loss'] else self.spectral_convergence_loss(Y_, X))
                gen_loss += main_loss 
                self.total_train_loss["train/main_loss"] += main_loss.item()
                  
            # update VQVAE
            self._record_loss('VQVAE_loss', gen_loss, mode=mode)
            self._update_VQVAE(gen_loss)
        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()


    @torch.no_grad()
    def _eval_step(self, batch):
        """Single step of evaluation."""
        mode = 'eval'
        x_c = batch[0] #[:,:,0:batch[1].min().item()] # torch.Size([B, 1, 48000])
        x = x_c.to(self.device)        
            
        # initialize VQVAE loss
        gen_loss = 0.0
        x = torch.squeeze(x)
        X = self.stft_magnitude(x, hop_size=self.hop_size)
        if self.config['input_transform'] == 'log1p':
            X = torch.log1p(X)
        
        z = self.model["VQVAE"].CNN_1D_encoder(X)
        zq, indices, vqloss, distance = self.model["VQVAE"].quantizer(z, stochastic=False, update=False)
        Y_ = self.model["VQVAE"].CNN_1D_decoder(zq)   
        
        # vq_loss
        self.total_eval_loss["eval/vqloss"] += F.mse_loss(zq, z.transpose(2, 1))
        
        # metric loss
        SP_loss = self.config["lambda_stft_loss"] * (self.cos_loss(X, Y_) if self.config['cos_loss'] else self.spectral_convergence_loss(Y_, X))
        gen_loss += SP_loss
        self.total_eval_loss["eval/SP_loss"] += SP_loss.item()
        
        if self.config['task'] == 'Speech_Enhancement':
            if self.config['input_transform'] == 'log1p':
                Y_ = torch.expm1(Y_)
            y_ = resynthesize(Y_, x, self.hop_size)
            waveform_loss = F.l1_loss(y_, x)
            self.total_eval_loss["eval/waveform_loss"] += waveform_loss.item()
          
        self._record_loss('VQVAE_loss', gen_loss, mode=mode)
    
        
    @torch.no_grad()
    def run_VQVAE(self, wav_input):
        wav_input = wav_input.to(self.device)
        SP_input = self.stft_magnitude(wav_input, hop_size=self.hop_size)            
        if self.config['input_transform'] == 'log1p':
            SP_input = torch.log1p(SP_input)
        
        z = self.model["VQVAE"].CNN_1D_encoder(SP_input)
        zq, indices, vqloss, distance = self.model["VQVAE"].quantizer(z, stochastic=False, update=False)
        SP_output = self.model["VQVAE"].CNN_1D_decoder(zq)                
                                     
        if self.config['input_transform'] == 'log1p':
            wav_output = resynthesize(torch.expm1(SP_output), wav_input, self.hop_size)
        else:
            wav_output = resynthesize(SP_output, wav_input, self.hop_size)
                                                       
        return SP_input.cpu(), SP_output.cpu(), z.transpose(2, 1).cpu(), zq.cpu(), wav_output.cpu(), indices
        
    @torch.no_grad()
    def VQScore_Evaluation(self, data_dict, mos_list, dataset, dataset_sub_name):
        VQScore_l2_x, VQScore_cos_x = [], []
        VQScore_l2_z, VQScore_cos_z = [], []
        
        whole_name = dataset + '_' + dataset_sub_name # ex: IUB_cosine
        #if not os.path.exists(self.exp_dir + whole_name):
        #    os.mkdir(self.exp_dir + whole_name)
        
        for file in data_dict:
            input_wav = data_dict[file]
            SP_input, SP_output, zT, zqT, wav_output, indices = self.run_VQVAE(input_wav)
                       
            ###### Input_output error           
            Square_diff, Square_input = torch.square(SP_input-SP_output), torch.square(SP_input)    
            
            VQScore_l2_x.append(torch.mean(Square_diff / (torch.mean(Square_input, dim=-1, keepdim=True)+eps) ).numpy())       
            VQScore_cos_x.append(-self.cos_loss(SP_input, SP_output).numpy())
            
            ##### Quantization error         
            Square_z_diff, Square_z_input = torch.square(zT-zqT), torch.square(zT)
            
            VQScore_l2_z.append(torch.mean(Square_z_diff / (torch.mean(Square_z_input, dim=-1, keepdim=True)+eps) ).numpy())
            VQScore_cos_z.append(-self.cos_loss(zT, zqT).numpy())
            
            # torchaudio.save(self.exp_dir + whole_name + '/' + self.config["name"] + '_' + whole_name + '_'+ file.split('/')[-1], wav_output, 16000)
        
        ###### Record_CC: Input_output error
        self._record_loss(dataset_sub_name + '_VQScore_l2_x_pearsonr',  pearsonr(VQScore_l2_x, mos_list)[0],  mode=dataset)   
        self._record_loss(dataset_sub_name + '_VQScore_cos_x_pearsonr', pearsonr(VQScore_cos_x, mos_list)[0], mode=dataset)                
        self.scatter_plot(mos_list, VQScore_l2_x,  whole_name + 'VQScore_l2_x.png')
        self.scatter_plot(mos_list, VQScore_cos_x, whole_name + 'VQScore_cos_x.png')
        
        ###### Record_CC: Quantization error
        self._record_loss(dataset_sub_name + '_VQScore_l2_z_pearsonr',  pearsonr(VQScore_l2_z, mos_list)[0],  mode=dataset+'_z')
        self._record_loss(dataset_sub_name + '_VQScore_cos_z_pearsonr', pearsonr(VQScore_cos_z, mos_list)[0], mode=dataset+'_z')     
        self.scatter_plot(mos_list, VQScore_l2_z,  whole_name + 'VQScore_l2_z.png')   
        self.scatter_plot(mos_list, VQScore_cos_z, whole_name + 'VQScore_cos_z.png')
               
        df = pd.DataFrame({'file': list(data_dict.keys()),'Real mos': mos_list, 
                           'VQScore_l2_x': VQScore_l2_x, 'VQScore_cos_x': VQScore_cos_x,
                           'VQScore_l2_z': VQScore_l2_z, 'VQScore_cos_z': VQScore_cos_z})
        df.to_csv(self.exp_dir + '/' + whole_name + '.csv')
        
    @torch.no_grad()
    def _eval_IUB(self):
        print('_eval_IUB.........')
        self.VQScore_Evaluation(self.iub.IUB_cosine_data_dict, self.iub.IUB_cosine_mos, 'IUB', 'cosine')
        self.VQScore_Evaluation(self.iub.IUB_voices_data_dict, self.iub.IUB_voices_mos, 'IUB', 'voices')
    
    @torch.no_grad() 
    def _eval_Tencent(self):
        print('_eval_Tencent.........')
        self.VQScore_Evaluation(self.tencent.Tencent_wR_data_dict,  self.tencent.Tencent_wR_mos, 'Tencent', 'wR')
        self.VQScore_Evaluation(self.tencent.Tencent_woR_data_dict, self.tencent.Tencent_woR_mos, 'Tencent', 'woR')
    
    @torch.no_grad()
    def _eval_DNS1_test(self):  
        print('_eval_DNS1_test.........')
        if not os.path.exists(self.exp_dir + '_DNS1'):
            os.mkdir(self.exp_dir + '_DNS1')
            
        Real_enhanced_dnsmos_p835 = []
        for file in self.dns1.DNS1_Real_dict:
            noisy = self.dns1.DNS1_Real_dict[file]
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)
            Real_enhanced_dnsmos_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))
            torchaudio.save(self.exp_dir + '_DNS1' + '/' + self.config["name"] + '_'+ file.split('/')[-1], y_noisy, 16000)                   
        self._record_loss('Real_enhanced_dnsmos_sig', np.mean([i['SIG'] for i in Real_enhanced_dnsmos_p835]), mode='dns1')
        self._record_loss('Real_enhanced_dnsmos_bak', np.mean([i['BAK'] for i in Real_enhanced_dnsmos_p835]), mode='dns1')
        self._record_loss('Real_enhanced_dnsmos_ovr', np.mean([i['OVRL'] for i in Real_enhanced_dnsmos_p835]), mode='dns1')
        
        Noreverb_enhanced_dnsmos_p835 = []
        for file in self.dns1.DNS1_Noreverb_dict:
            noisy = self.dns1.DNS1_Noreverb_dict[file]
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)
            Noreverb_enhanced_dnsmos_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))
            torchaudio.save(self.exp_dir + '_DNS1' + '/' + self.config["name"] + '_'+ file.split('/')[-1], y_noisy, 16000)                  
        self._record_loss('Noreverb_enhanced_dnsmos_sig', np.mean([i['SIG'] for i in Noreverb_enhanced_dnsmos_p835]), mode='dns1')
        self._record_loss('Noreverb_enhanced_dnsmos_bak', np.mean([i['BAK'] for i in Noreverb_enhanced_dnsmos_p835]), mode='dns1')
        self._record_loss('Noreverb_enhanced_dnsmos_ovr', np.mean([i['OVRL'] for i in Noreverb_enhanced_dnsmos_p835] ), mode='dns1')
        
        Reverb_enhanced_dnsmos_p835 = []
        for file in self.dns1.DNS1_Reverb_dict:
            noisy = self.dns1.DNS1_Reverb_dict[file]
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)
            Reverb_enhanced_dnsmos_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))
            torchaudio.save(self.exp_dir + '_DNS1' + '/' + self.config["name"] + '_'+ file.split('/')[-1], y_noisy, 16000)                  
        self._record_loss('Reverb_enhanced_dnsmos_sig', np.mean([i['SIG'] for i in Reverb_enhanced_dnsmos_p835]), mode='dns1')
        self._record_loss('Reverb_enhanced_dnsmos_bak', np.mean([i['BAK'] for i in Reverb_enhanced_dnsmos_p835]), mode='dns1')
        self._record_loss('Reverb_enhanced_dnsmos_ovr', np.mean([i['OVRL'] for i in Reverb_enhanced_dnsmos_p835]  ), mode='dns1')
    
    @torch.no_grad()
    def _eval_DNS3_test(self):
        print('_eval_DNS3_test.........')
        if not os.path.exists(self.exp_dir + '_DNS3'):
            os.mkdir(self.exp_dir + '_DNS3')
        
        nonenglish_synthetic_p835 = []
        for file in self.dns3.DNS3_nonenglish_synthetic_dict:
            noisy = self.dns3.DNS3_nonenglish_synthetic_dict[file]
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)
            nonenglish_synthetic_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))   
            torchaudio.save(self.exp_dir + '_DNS3' + '/' + self.config["name"] + '_'+ file.split('/')[-1], y_noisy, 16000)           
        self._record_loss('nonenglish_synthetic_sig', np.mean([i['SIG'] for i in nonenglish_synthetic_p835]), mode='dns3')
        self._record_loss('nonenglish_synthetic_bak', np.mean([i['BAK'] for i in nonenglish_synthetic_p835]), mode='dns3')
        self._record_loss('nonenglish_synthetic_ovr', np.mean([i['OVRL'] for i in nonenglish_synthetic_p835]), mode='dns3')
        
        stationary_p835 = []
        for file in self.dns3.DNS3_stationary_dict:
            noisy = self.dns3.DNS3_stationary_dict[file]
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)
            stationary_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))  
            torchaudio.save(self.exp_dir + '_DNS3' + '/' + self.config["name"] + '_'+ file.split('/')[-1], y_noisy, 16000)       
        self._record_loss('stationary_sig', np.mean([i['SIG'] for i in stationary_p835]), mode='dns3')
        self._record_loss('stationary_bak', np.mean([i['BAK'] for i in stationary_p835]), mode='dns3')
        self._record_loss('stationary_ovr', np.mean([i['OVRL'] for i in stationary_p835]), mode='dns3')
        
        ms_realrec_nonenglish_p835 = []
        for file in self.dns3.DNS3_ms_realrec_nonenglish_dict:
            noisy = self.dns3.DNS3_ms_realrec_nonenglish_dict[file]
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)
            ms_realrec_nonenglish_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))
            torchaudio.save(self.exp_dir + '_DNS3' + '/' + self.config["name"] + '_'+ file.split('/')[-1], y_noisy, 16000)                        
        self._record_loss('ms_realrec_nonenglish_sig', np.mean([i['SIG'] for i in ms_realrec_nonenglish_p835]), mode='dns3')
        self._record_loss('ms_realrec_nonenglish_bak', np.mean([i['BAK'] for i in ms_realrec_nonenglish_p835]), mode='dns3')
        self._record_loss('ms_realrec_nonenglish_ovr', np.mean([i['OVRL'] for i in ms_realrec_nonenglish_p835]), mode='dns3')
        
        ms_realrec_p835 = []
        for file in self.dns3.DNS3_ms_realrec_dict:
            noisy = self.dns3.DNS3_ms_realrec_dict[file]
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)
            ms_realrec_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))
            torchaudio.save(self.exp_dir + '_DNS3' + '/' + self.config["name"] + '_'+ file.split('/')[-1], y_noisy, 16000)            
        self._record_loss('ms_realrec_sig', np.mean([i['SIG'] for i in ms_realrec_p835]), mode='dns3')
        self._record_loss('ms_realrec_bak', np.mean([i['BAK'] for i in ms_realrec_p835]), mode='dns3')
        self._record_loss('ms_realrec_ovr', np.mean([i['OVRL'] for i in ms_realrec_p835]), mode='dns3')
    
    
    @torch.no_grad()
    def _eval_vctk_ValidSet(self):
        print('_eval_vctk_ValidSet.........')
        enhanced_dnsmos_p835, enhanced_pesq = [], []   
        for file in self.vctk_valid.VCTK_data_dict: 
            ########## Noisy ##########
            noisy = self.vctk_valid.VCTK_data_dict[file]         
            SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)  
                         
            ########## Clean ##########
            clean, fs = torchaudio.load(file.replace('noisy','clean'))
            SP_clean, SP_y_clean, zT, zqT, y_clean, clean_indices = self.run_VQVAE(clean) 
            
            ## objective Metrics                   
            enhanced_dnsmos_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))
            enhanced_pesq.append(pesq(fs=16000, ref=clean[0].numpy(), deg=y_noisy[0].numpy(), mode="wb"))
                   
        ######## Objective Metrics (DNSMOS and PESQ)
        # noisy enhanced results
        enhanced_dnsmos_sig = [i['SIG'] for i in enhanced_dnsmos_p835]
        enhanced_dnsmos_bak = [i['BAK'] for i in enhanced_dnsmos_p835]
        enhanced_dnsmos_ovr = [i['OVRL'] for i in enhanced_dnsmos_p835]                   
        self._record_loss('enhanced_dnsmos_sig', np.mean(enhanced_dnsmos_sig), mode='vctk_valid')
        self._record_loss('enhanced_dnsmos_bak', np.mean(enhanced_dnsmos_bak), mode='vctk_valid')
        self._record_loss('enhanced_dnsmos_ovr', np.mean(enhanced_dnsmos_ovr), mode='vctk_valid')
        self._record_loss('enhanced_pesq', np.mean(enhanced_pesq), mode='vctk_valid')       
        
        # Save SE model checkpoints
        if self.steps <= self.config['AT_training_start_steps']:
            if np.mean(enhanced_pesq) > self.highest_pesq:
                if os.path.isfile(os.path.join(self.config["outdir"], 'checkpoint-pesq='+ str(self.highest_pesq)[0:5]+'.pkl')):
                    os.remove(os.path.join(self.config["outdir"], 'checkpoint-pesq='+ str(self.highest_pesq)[0:5]+'.pkl'))
                self.highest_pesq = np.mean(enhanced_pesq)
                self.save_checkpoint(os.path.join(self.config["outdir"], 'checkpoint-pesq='+ str(self.highest_pesq)[0:5]+'.pkl'))
            
            if np.mean(enhanced_dnsmos_ovr) > self.highest_dnsmos_ovr:
                if os.path.isfile(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr='+ str(self.highest_dnsmos_ovr)[0:5]+'.pkl')):
                    os.remove(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr='+ str(self.highest_dnsmos_ovr)[0:5]+'.pkl'))
                self.highest_dnsmos_ovr = np.mean(enhanced_dnsmos_ovr)
                self.save_checkpoint(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr='+ str(self.highest_dnsmos_ovr)[0:5]+'.pkl'))
        else:
            if np.mean(enhanced_pesq) > self.highest_pesq:
                if os.path.isfile(os.path.join(self.config["outdir"], 'checkpoint-pesq='+ str(self.highest_pesq)[0:5]+'_AT.pkl')):
                    os.remove(os.path.join(self.config["outdir"], 'checkpoint-pesq='+ str(self.highest_pesq)[0:5]+'_AT.pkl'))
                self.highest_pesq = np.mean(enhanced_pesq)
                self.save_checkpoint(os.path.join(self.config["outdir"], 'checkpoint-pesq='+ str(self.highest_pesq)[0:5]+'_AT.pkl'))
            
            if np.mean(enhanced_dnsmos_ovr) > self.highest_dnsmos_ovr:
                if os.path.isfile(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr='+ str(self.highest_dnsmos_ovr)[0:5]+'_AT.pkl')):
                    os.remove(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr='+ str(self.highest_dnsmos_ovr)[0:5]+'_AT.pkl'))
                self.highest_dnsmos_ovr = np.mean(enhanced_dnsmos_ovr)
                self.save_checkpoint(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr='+ str(self.highest_dnsmos_ovr)[0:5]+'_AT.pkl'))
                
    @torch.no_grad()
    def _eval_vctk_TestSet(self):
        print('_eval_vctk_TestSet.........')
        # Quality estimation
        if self.config['task'] == 'Quality_Estimation':
          VQScore_l2_x, VQScore_cos_x = [], []
          VQScore_l2_z, VQScore_cos_z = [], []
          
          for file in self.vctk_test.VCTK_data_dict: 
              noisy = self.vctk_test.VCTK_data_dict[file]                       
              SP_input, SP_output, zT, zqT, wav_output, indices = self.run_VQVAE(noisy)
                         
              ###### Input_output error            
              Square_diff, Square_input = torch.square(SP_input-SP_output), torch.square(SP_input)    
              
              VQScore_l2_x.append(torch.mean(Square_diff / (torch.mean(Square_input, dim=-1, keepdim=True)+eps) ).numpy())       
              VQScore_cos_x.append(-self.cos_loss(SP_input, SP_output).numpy())
              
              ##### Quantization error         
              Square_z_diff, Square_z_input = torch.square(zT-zqT), torch.square(zT)
              
              VQScore_l2_z.append(torch.mean(Square_z_diff / (torch.mean(Square_z_input, dim=-1, keepdim=True)+eps) ).numpy())
              VQScore_cos_z.append(-self.cos_loss(zT, zqT).numpy())
              
          ###### Record_CC: Input_output error
          self._record_loss('sig_VQScore_l2_x_pearsonr',  pearsonr(VQScore_l2_x, self.vctk_test.sig)[0], mode='vctk')  
          self._record_loss('bak_VQScore_l2_x_pearsonr',  pearsonr(VQScore_l2_x, self.vctk_test.bak)[0], mode='vctk')  
          self._record_loss('ovr_VQScore_l2_x_pearsonr',  pearsonr(VQScore_l2_x, self.vctk_test.ovr)[0], mode='vctk') 
          self._record_loss('pesq_VQScore_l2_x_pearsonr', pearsonr(VQScore_l2_x, self.vctk_test.PESQ_list)[0], mode='vctk')
          self._record_loss('stoi_VQScore_l2_x_pearsonr', pearsonr(VQScore_l2_x, self.vctk_test.STOI_list)[0], mode='vctk')
          self._record_loss('snr_VQScore_l2_x_pearsonr',  pearsonr(VQScore_l2_x, self.vctk_test.SNR_list)[0], mode='vctk')

          self._record_loss('sig_VQScore_cos_x_pearsonr',  pearsonr(VQScore_cos_x, self.vctk_test.sig)[0], mode='vctk')  
          self._record_loss('bak_VQScore_cos_x_pearsonr',  pearsonr(VQScore_cos_x, self.vctk_test.bak)[0], mode='vctk')  
          self._record_loss('ovr_VQScore_cos_x_pearsonr',  pearsonr(VQScore_cos_x, self.vctk_test.ovr)[0], mode='vctk') 
          self._record_loss('pesq_VQScore_cos_x_pearsonr', pearsonr(VQScore_cos_x, self.vctk_test.PESQ_list)[0], mode='vctk')
          self._record_loss('stoi_VQScore_cos_x_pearsonr', pearsonr(VQScore_cos_x, self.vctk_test.STOI_list)[0], mode='vctk')          
          self._record_loss('snr_VQScore_cos_x_pearsonr',  pearsonr(VQScore_cos_x, self.vctk_test.SNR_list)[0], mode='vctk')
                
          ###### Record_CC: Quantization error        
          self._record_loss('sig_VQScore_l2_z_pearsonr',  pearsonr(VQScore_l2_z, self.vctk_test.sig)[0], mode='vctk_z')  
          self._record_loss('bak_VQScore_l2_z_pearsonr',  pearsonr(VQScore_l2_z, self.vctk_test.bak)[0], mode='vctk_z')  
          self._record_loss('ovr_VQScore_l2_z_pearsonr',  pearsonr(VQScore_l2_z, self.vctk_test.ovr)[0], mode='vctk_z') 
          self._record_loss('pesq_VQScore_l2_z_pearsonr', pearsonr(VQScore_l2_z, self.vctk_test.PESQ_list)[0], mode='vctk_z')
          self._record_loss('stoi_VQScore_l2_z_pearsonr', pearsonr(VQScore_l2_z, self.vctk_test.STOI_list)[0], mode='vctk_z')
          self._record_loss('snr_VQScore_l2_z_pearsonr',  pearsonr(VQScore_l2_z, self.vctk_test.SNR_list)[0], mode='vctk_z')
              
          self._record_loss('sig_VQScore_cos_z_pearsonr',  pearsonr(VQScore_cos_z, self.vctk_test.sig)[0], mode='vctk_z')  
          self._record_loss('bak_VQScore_cos_z_pearsonr',  pearsonr(VQScore_cos_z, self.vctk_test.bak)[0], mode='vctk_z')  
          self._record_loss('ovr_VQScore_cos_z_pearsonr',  pearsonr(VQScore_cos_z, self.vctk_test.ovr)[0], mode='vctk_z') 
          self._record_loss('pesq_VQScore_cos_z_pearsonr', pearsonr(VQScore_cos_z, self.vctk_test.PESQ_list)[0], mode='vctk_z')
          self._record_loss('stoi_VQScore_cos_z_pearsonr', pearsonr(VQScore_cos_z, self.vctk_test.STOI_list)[0], mode='vctk_z')
          self._record_loss('snr_VQScore_cos_z_pearsonr',  pearsonr(VQScore_cos_z, self.vctk_test.SNR_list)[0], mode='vctk_z')
          
          self.scatter_plot(self.vctk_test.sig, VQScore_cos_z, 'VCTK_VQScore_cos_z_sig.png')
          self.scatter_plot(self.vctk_test.bak, VQScore_cos_z, 'VCTK_VQScore_cos_z_bak.png')
          self.scatter_plot(self.vctk_test.ovr, VQScore_cos_z, 'VCTK_VQScore_cos_z_ovr.png')
          self.scatter_plot(self.vctk_test.PESQ_list, VQScore_cos_z, 'VCTK_VQScore_cos_z_pesq.png')
          self.scatter_plot(self.vctk_test.STOI_list, VQScore_cos_z, 'VCTK_VQScore_cos_z_stoi.png')
          self.scatter_plot(self.vctk_test.SNR_list, VQScore_cos_z, 'VCTK_VQScore_cos_z_SNR.png')
          
          if pearsonr(VQScore_cos_z, self.vctk_test.ovr)[0] > self.highest_dnsmos_ovr_CC:
              if os.path.isfile(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr_CC='+ str(self.highest_dnsmos_ovr_CC)[0:5]+'.pkl')):
                  os.remove(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr_CC='+ str(self.highest_dnsmos_ovr_CC)[0:5]+'.pkl'))
              self.highest_dnsmos_ovr_CC = pearsonr(VQScore_cos_z, self.vctk_test.ovr)[0]
              self.save_checkpoint(os.path.join(self.config["outdir"], 'checkpoint-dnsmos_ovr_CC='+ str(self.highest_dnsmos_ovr_CC)[0:5]+'.pkl'))
          
        elif self.config['task'] == 'Speech_Enhancement':    
            ### speech enhancement
            index_accuracy = []
            
            Noisy_spectral_convergence_loss, Clean_spectral_convergence_loss = [], []
            Noisy_to_clean_spectral_convergence_loss = []     
            
            clean_recovery_dnsmos_p835, clean_recovery_pesq = [], []      
            enhanced_dnsmos_p835, enhanced_pesq = [], []
            
            Noisy_index_hist = np.zeros(self.config['VQVAE_params']['codebook_size'])
            Clean_index_hist = np.zeros(self.config['VQVAE_params']['codebook_size'])
            
            if not os.path.exists(self.exp_dir + '_vctk_noisy'):
                os.mkdir(self.exp_dir + '_vctk_noisy')
            if not os.path.exists(self.exp_dir + '_vctk_clean'):
                os.mkdir(self.exp_dir + '_vctk_clean')
            
            for file in self.vctk_test.VCTK_data_dict: 
                ########## Noisy ##########
                noisy = self.vctk_test.VCTK_data_dict[file]         
                SP_noisy, SP_y_noisy, zT, zqT, y_noisy, noisy_indices = self.run_VQVAE(noisy)  
                             
                ########## Clean ##########
                clean, fs = torchaudio.load(self.vctk_Clean_path + file.split('/')[-1])
                SP_clean, SP_y_clean, zT, zqT, y_clean, clean_indices = self.run_VQVAE(clean) 
                
                for ind in noisy_indices.cpu().numpy():
                    Noisy_index_hist[ind] += 1
                
                for ind in clean_indices.cpu().numpy():
                    Clean_index_hist[ind] += 1
    
                index_accuracy.append((torch.sum(noisy_indices[0,:]==clean_indices[0,:])/clean_indices.shape[1]).cpu().numpy())
                
                Noisy_spectral_convergence_loss.append(self.spectral_convergence_loss(SP_y_noisy, SP_noisy).numpy())    
                Clean_spectral_convergence_loss.append(self.spectral_convergence_loss(SP_y_clean, SP_clean).numpy())    
                Noisy_to_clean_spectral_convergence_loss.append(self.spectral_convergence_loss(SP_y_noisy, SP_clean).numpy())
                                
                torchaudio.save(self.exp_dir + '_vctk_noisy' + '/' + self.config["name"] + '_vctk_noisy' + '_'+ file.split('/')[-1], y_noisy, 16000)
                torchaudio.save(self.exp_dir + '_vctk_clean' + '/' + self.config["name"] + '_vctk_clean' + '_'+ file.split('/')[-1], y_clean, 16000)
                
                ## objective Metrics 
                clean_recovery_dnsmos_p835.append(compute_score(y_clean[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))        
                clean_recovery_pesq.append(pesq(fs=16000, ref=clean[0].numpy(), deg=y_clean[0].numpy(), mode="wb"))
                
                enhanced_dnsmos_p835.append(compute_score(y_noisy[0].numpy(), 16000, is_personalized_MOS=False, is_normalized=True, is_p808=False))
                enhanced_pesq.append(pesq(fs=16000, ref=clean[0].numpy(), deg=y_noisy[0].numpy(), mode="wb"))
                       
            ######## Objective Metrics (DNSMOS and PESQ)
            # clean recovery results
            clean_recovery_dnsmos_sig = [i['SIG'] for i in clean_recovery_dnsmos_p835]
            clean_recovery_dnsmos_bak = [i['BAK'] for i in clean_recovery_dnsmos_p835]
            clean_recovery_dnsmos_ovr = [i['OVRL'] for i in clean_recovery_dnsmos_p835]            
            self._record_loss('clean_recovery_dnsmos_sig', np.mean(clean_recovery_dnsmos_sig), mode='vctk')
            self._record_loss('clean_recovery_dnsmos_bak', np.mean(clean_recovery_dnsmos_bak), mode='vctk')
            self._record_loss('clean_recovery_dnsmos_ovr', np.mean(clean_recovery_dnsmos_ovr), mode='vctk')
            self._record_loss('clean_recovery_pesq', np.mean(clean_recovery_pesq), mode='vctk')
            
            # noisy enhanced results
            enhanced_dnsmos_sig = [i['SIG'] for i in enhanced_dnsmos_p835]
            enhanced_dnsmos_bak = [i['BAK'] for i in enhanced_dnsmos_p835]
            enhanced_dnsmos_ovr = [i['OVRL'] for i in enhanced_dnsmos_p835]                   
            self._record_loss('enhanced_dnsmos_sig', np.mean(enhanced_dnsmos_sig), mode='vctk')
            self._record_loss('enhanced_dnsmos_bak', np.mean(enhanced_dnsmos_bak), mode='vctk')
            self._record_loss('enhanced_dnsmos_ovr', np.mean(enhanced_dnsmos_ovr), mode='vctk')
            self._record_loss('enhanced_pesq', np.mean(enhanced_pesq), mode='vctk')
               
            self._record_loss('Clean_spectral_convergence_loss', np.mean(Clean_spectral_convergence_loss), mode='vctk')      
            self._record_loss('Noisy_spectral_convergence_loss', np.mean(Noisy_spectral_convergence_loss), mode='vctk')
            self._record_loss('Noisy_to_clean_spectral_convergence_loss', np.mean(Noisy_to_clean_spectral_convergence_loss), mode='vctk')
            self._record_loss('index_accuracy', np.mean(index_accuracy), mode='vctk')
            
                
            ######### Dictionary usage (optional)
            cluster_size = self.model["VQVAE"].quantizer.quantizer._codebook.cluster_size[0].cpu().numpy()
            plt.plot(range(self.config['VQVAE_params']['codebook_size']), cluster_size)
            plt.xlabel('Index')
            plt.ylabel('self_cluster_size')        
            plt.savefig(self.exp_dir +'self_cluster_size_hist.png', dpi=150)
            plt.clf()
            
            plt.plot(range(self.config['VQVAE_params']['codebook_size']), Noisy_index_hist)
            plt.xlabel('Index')
            plt.ylabel('Noisy_index_hist')        
            plt.savefig(self.exp_dir +'Noisy_VCTK_index_hist.png', dpi=150)
            plt.clf()
            
            plt.plot(range(self.config['VQVAE_params']['codebook_size']), Clean_index_hist)
            plt.xlabel('Index')
            plt.ylabel('Clean_index_hist')        
            plt.savefig(self.exp_dir +'Clean_VCTK_index_hist.png', dpi=150)
            plt.clf()
            
            Pr_N = Noisy_index_hist/Noisy_index_hist.sum()
            self._record_loss('Noisy_index_entropy', entropy(Pr_N, base=2), mode='vctk')
            
            Pr_C = Clean_index_hist/Clean_index_hist.sum()
            self._record_loss('Clean_index_entropy', entropy(Pr_C, base=2), mode='vctk')
        
        