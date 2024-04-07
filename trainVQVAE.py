#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import soundfile as sf
from torch.utils.data import DataLoader

from dataloader.dataset import SingleDataset
from models.VQVAE_models import VQVAE_SE, VQVAE_QE
from trainer.autoencoder import Trainer as TrainerAutoEncoder
from bin.train import Train


class TrainMain(Train):
    def __init__(self, args,):
        super(TrainMain, self).__init__(args=args,)
        self.train_mode = self.config.get('train_mode', 'autoencoder')
        self.data_path = self.config['data']['path']
    
    
    def initialize_data_loader(self):
        logging.info("Loading datasets...")

        if self.train_mode in ['autoencoder']:
            train_set = self._audio('clean_train')
            valid_set = self._audio('clean_valid')
            # collater = CollaterAudio(batch_length=self.config['batch_length'])
            collater = None
            self.Trainer = TrainerAutoEncoder
        else:
            raise NotImplementedError(f"Train mode: {self.train_mode} is not supported!")

        logging.info(f"The number of training files = {len(train_set)}.")
        logging.info(f"The number of validation files = {len(valid_set)}.")
        dataset = {'train': train_set, 'dev': valid_set}
        self._data_loader(dataset, collater)
    
    def define_model(self):    
        if self.config['task'] == "Speech_Enhancement":
            VQVAE = VQVAE_SE(
                **self.config['VQVAE_params']).to(self.device)
        elif self.config['task'] == "Quality_Estimation":
            VQVAE = VQVAE_QE(
                **self.config['VQVAE_params']).to(self.device)
    
        self.model = {"VQVAE": VQVAE}
        self._define_optimizer_scheduler()
    
    def define_trainer(self):
        self._show_setting()
        trainer_parameters = {}
        trainer_parameters['steps'] = 0
        trainer_parameters['epochs'] = 0
        trainer_parameters['data_loader'] = self.data_loader
        trainer_parameters['model'] = self.model 
        trainer_parameters['criterion'] = self.criterion 
        trainer_parameters['optimizer'] = self.optimizer
        trainer_parameters['scheduler'] = self.scheduler
        trainer_parameters['config'] = self.config
        trainer_parameters['device'] = self.device
        self.trainer = self.Trainer(**trainer_parameters)
    

    def _data_loader(self, dataset, collater):
        self.data_loader = {
            'train': DataLoader(
                dataset=dataset['train'],
                shuffle=True,
                collate_fn=collater,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
            ),
            'dev': DataLoader(
                dataset=dataset['dev'],
                shuffle=False,
                collate_fn=collater,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
            ),
        }
    
    
    def _audio(self, subset, subset_num=-1, return_utt_id=False):
        audio_dir = os.path.join(
            self.data_path, self.config['data']['subset'][subset])
        params = {
            'data_path': '/',
            'files': audio_dir,
            'query': "*.wav",
            'load_fn': sf.read,
            'return_utt_id': return_utt_id,
            'subset_num': subset_num,
            'batch_length': self.config['batch_length'],
        }
        return SingleDataset(**params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="exp")
    parser.add_argument("--resume", default="", type=str, nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--disable_cudnn', choices=('True','False'), default='False', help='Disable CUDNN')
    args = parser.parse_args()
        
    # initial train_main
    train_main = TrainMain(args=args)   

    # get dataset
    train_main.initialize_data_loader()
    
    # define models, optimizers, and schedulers
    train_main.define_model()
    
    # define criterions
    # train_main.define_criterion()

    # define trainer
    train_main.define_trainer()

    # model initialization
    train_main.initialize_model()

    # run training loop
    train_main.run()

if __name__ == "__main__":
    main()
