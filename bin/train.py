#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import random
import logging
import torch
import numpy as np

class Train(object):
    def __init__(
        self,
        args,
    ):
        # set logger
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        # Fix seed and make backends deterministic
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            logging.info(f"device: cpu")
        else:
            self.device = torch.device('cuda')
            logging.info(f"device: gpu")
            torch.cuda.manual_seed_all(args.seed)
            if args.disable_cudnn == "False":
                torch.backends.cudnn.benchmark = True
        
        # initialize config
        with open(args.config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config.update(vars(args))

        # initialize model folder
        expdir = os.path.join(args.exp_root, args.tag)
        os.makedirs(expdir, exist_ok=True)
        self.config["outdir"] = expdir

        # save config
        with open(os.path.join(expdir, "config.yml"), "w") as f:
            yaml.dump(self.config, f, Dumper=yaml.Dumper)
        for key, value in self.config.items():
            logging.info(f"[TrainGAN] {key} = {value}")
        
        # initialize attribute
        self.resume = args.resume
        self.data_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None

        
    def initialize_data_loader(self):
        pass
    

    def define_model(self):
        pass
    

    def define_trainer(self):
        pass


    def initialize_model(self):
        initial = self.config.get("initial", "")
        if len(self.resume) != 0:
            self.trainer.load_checkpoint(self.resume)
            logging.info(f"Successfully resumed from {self.resume}.")
        elif len(initial) != 0:
            self.trainer.load_checkpoint(initial, load_only_params=True)
            logging.info(f"Successfully initialize parameters from {initial}.")
        else:
            logging.info("Train from scrach")
    

    def run(self):
        try:
            self.trainer.run()
        finally:
            self.trainer.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.trainer.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.trainer.steps}steps.")
    
    
    def _define_optimizer_scheduler(self):
        VQVAE_optimizer_class = getattr(
            torch.optim, 
            self.config['VQVAE_optimizer_type'])
        
        self.optimizer = {
            'VQVAE': VQVAE_optimizer_class(
                self.model['VQVAE'].parameters(),
                **self.config['VQVAE_optimizer_params'])}

        VQVAE_scheduler_class = getattr(
            torch.optim.lr_scheduler,
            self.config.get('VQVAE_scheduler_type', "StepLR"))
        
        self.scheduler = {
            'VQVAE': VQVAE_scheduler_class(
                optimizer=self.optimizer['VQVAE'],
                **self.config['VQVAE_scheduler_params'])}
    
    def _show_setting(self):
        logging.info(self.model['VQVAE'])
        logging.info(self.optimizer['VQVAE'])
        logging.info(self.scheduler['VQVAE'])
        
