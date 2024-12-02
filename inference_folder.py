# -*- coding: utf-8 -*-
"""
@author: szuweif
"""
import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
import pandas as pd
import time
from models.VQVAE_models import VQVAE_SE, VQVAE_QE

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

def stft_magnitude(x, hop_size, fft_size=512, win_length=512):
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

def cos_loss(SP_noisy, SP_y_noisy):  
    eps=1e-5
    SP_noisy_norm = torch.norm(SP_noisy, p=2, dim=-1, keepdim=True)+eps
    SP_y_noisy_norm = torch.norm(SP_y_noisy, p=2, dim=-1, keepdim=True)+eps  
    Cos_frame = torch.sum(SP_noisy/SP_noisy_norm * SP_y_noisy/SP_y_noisy_norm, dim=-1) # torch.Size([B, T, 1])
    
    return -torch.mean(Cos_frame)
    
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-m','--path_of_model_weights', type=str, required=True)
parser.add_argument('-i', '--path_of_input_audio_folder', type=str, required=True)
parser.add_argument('-o', '--path_of_output_audio_folder', type=str, default="./enhanced/")
args = parser.parse_args()

# initialize config
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if not torch.cuda.is_available():
    device = torch.device('cpu')
    print("device: cpu")
else:
    device = torch.device('cuda')
    print("device: gpu")
    torch.backends.cudnn.benchmark = True

with torch.no_grad():        
    if config['task'] == "Speech_Enhancement":
        VQVAE = VQVAE_SE(**config['VQVAE_params']).to(device).eval()
        hop_size = 128
        
        VQVAE.load_state_dict(torch.load(args.path_of_model_weights)['model']['VQVAE'])
        if not os.path.exists(args.path_of_output_audio_folder):
            os.mkdir(args.path_of_output_audio_folder)
        
        threshold = 1
        cluster_size = VQVAE.quantizer.quantizer._codebook.cluster_size[0].cpu().numpy()
        preserved_num = np.sum(cluster_size>threshold)
        temp = torch.zeros([1, preserved_num, 200])
        j=0
        for i in range(cluster_size.shape[0]):
            if cluster_size[i] > threshold:
                temp [:,j,:] = VQVAE.quantizer.quantizer._codebook.embed[:,i,:]
                j = j+1
        VQVAE.quantizer.quantizer._codebook.embed = temp.to(device)
        
        file_list = get_filepaths(args.path_of_input_audio_folder)
        estimated_pesq = []
        for file in file_list:
            clean, fs = torchaudio.load('/vctk_data/clean_testset_wav_16k/' + file.split('/')[-1])
            
            wav_input, fs = torchaudio.load(file)            
            wav_input = wav_input.to(device)
            SP_input = stft_magnitude(wav_input, hop_size=hop_size)            
            if config['input_transform'] == 'log1p':
                SP_input = torch.log1p(SP_input)
        
            z = VQVAE.CNN_1D_encoder(SP_input)
            zq, indices, vqloss, distance = VQVAE.quantizer(z, stochastic=False, update=False)
            SP_output = VQVAE.CNN_1D_decoder(zq)                
                                         
            if config['input_transform'] == 'log1p':
                wav_output = resynthesize(torch.expm1(SP_output), wav_input, hop_size).cpu()
            else:
                wav_output = resynthesize(SP_output, wav_input, hop_size).cpu()    
            
            torchaudio.save(args.path_of_output_audio_folder + file.split('/')[-1], wav_output, 16000)  
            #print('==============================================================================')
            #print('enhanced wav is saved at:' + args.enhanced_wav_path)
            #print('==============================================================================')
            
            estimated_pesq.append(pesq(fs=16000, ref=clean[0].numpy(), deg=wav_output[0].numpy(), mode="wb"))
        print(np.mean(estimated_pesq))
        
    elif config['task'] == "Quality_Estimation":
        hop_size = 256
        VQVAE = VQVAE_QE(**config['VQVAE_params']).to(device).eval()
        VQVAE.load_state_dict(torch.load(args.path_of_model_weights)['model']['VQVAE'])
        
        file_list = get_filepaths(args.path_of_input_audio_folder)
        num = 0
        original_VQ = []
        start_time = time.time()
        for file in file_list:
            speech, fs   = torchaudio.load(file)
            if fs != 16000:
                speech = torchaudio.functional.resample(speech, fs, 16000).to(device)
            
            
            SP_original = stft_magnitude(speech, hop_size=hop_size)
            if config['input_transform'] == 'log1p':
                SP_original = torch.log1p(SP_original)
            
            z = VQVAE.CNN_1D_encoder(SP_original.cuda())
            zq, indices, vqloss, distance = VQVAE.quantizer(z, stochastic=False, update=False)
            #SP_output = VQVAE.CNN_1D_decoder(zq)                                                                 
            VQScore_cos_z_original = -cos_loss(z.transpose(2, 1).cpu(), zq.cpu()).numpy()
            
            original_VQ.append(VQScore_cos_z_original)
            
        sort_index = np.argsort(original_VQ) # in ascending order
        
        sorted_VQ = [original_VQ[i] for i in sort_index]
        sorted_file_list = [file_list[i] for i in sort_index]
        score_dict = {'filename':sorted_file_list, 'VQScore':sorted_VQ}

        df = pd.DataFrame.from_dict(score_dict)
        df.to_csv('VQscore.csv', index=False)
        end_time = time.time()

        print('Total number of files evaluated:', len(original_VQ))
        print('Average VQScore:',  np.mean(original_VQ))
        print('VQScore list (in ascending order) has been saved in the ./VQscore.csv')
        print ('The evaluation takes around %.2fmin' % ((end_time - start_time) / 60.))
        #import pdb;pdb.set_trace()
        #print('end')
        
        
            