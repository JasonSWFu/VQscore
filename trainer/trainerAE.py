#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import torch
import time

from collections import defaultdict
from tensorboardX import SummaryWriter
from tqdm import tqdm

class TrainerAE(object):
    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models.
            optimizer (dict): Dict of optimizers. 
            scheduler (dict): Dict of schedulers. 
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.finish_train = False


    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")


    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {"VQVAE": self.optimizer["VQVAE"].state_dict()},
            "scheduler": {"VQVAE": self.scheduler["VQVAE"].state_dict()},
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {
            "VQVAE": self.model["VQVAE"].state_dict(),
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def _train_step(self, batch):
        """Single step of training."""
        pass
        

    def _train_epoch(self):
        """One epoch of training."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        if train_steps_per_epoch > 200:
            logging.info(
                f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                f"({self.train_steps_per_epoch} steps per epoch)."
            )


    def _eval_step(self, batch):
        """Single step of evaluation."""
        pass


    def _eval_epoch(self):
        """One epoch of evaluation."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)
        
        #self.remove_code(self.steps)
        start_time = time.time() 
        
           
        if self.config['task'] == 'Speech_Enhancement':
            self._eval_vctk_ValidSet() # validation set for SE
            self._eval_vctk_TestSet()
            self._eval_DNS1_test()
            self._eval_DNS3_test()
        elif self.config['task'] == 'Quality_Estimation':
            self._eval_vctk_TestSet() # validation set for QE
            self._eval_Tencent()
            self._eval_IUB()
        else:
            raise NotImplementedError("Task is not supported!")
        
        end_time = time.time()
        print ('Evaluation takes %.2fm' % ((end_time - start_time) / 60.))
        
        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            if key.startswith('eval'):
                self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()


    def _update_VQVAE(self, gen_loss):
        """Update VQVAE."""
        self.optimizer["VQVAE"].zero_grad()
        gen_loss.backward()
        if self.config["VQVAE_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["VQVAE"].parameters(),
                self.config["VQVAE_grad_norm"],
            )
        self.optimizer["VQVAE"].step()
        #self.scheduler["VQVAE"].step()  
    

    def _record_loss(self, name, loss, mode='train'):
        """Record loss."""
        if mode == 'train':
            self.total_train_loss[f"train/{name}"] += loss.item()
        elif mode == 'eval':
            self.total_eval_loss[f"eval/{name}"] += loss.item()
        elif mode == 'vctk':
            self.total_eval_loss[f"vctk/{name}"] += loss
        elif mode == 'vctk_valid':
            self.total_eval_loss[f"vctk_valid/{name}"] += loss
        elif mode == 'IUB':
            self.total_eval_loss[f"IUB/{name}"] += loss 
        elif mode == 'Tencent':
            self.total_eval_loss[f"Tencent/{name}"] += loss           
        elif mode == 'IUB_z':
            self.total_eval_loss[f"IUB_z/{name}"] += loss 
        elif mode == 'Tencent_z':
            self.total_eval_loss[f"Tencent_z/{name}"] += loss
        elif mode == 'vctk_z':
            self.total_eval_loss[f"vctk_z/{name}"] += loss
        elif mode == 'dns1':
            self.total_eval_loss[f"dns1/{name}"] += loss
        elif mode == 'dns3':
            self.total_eval_loss[f"dns3/{name}"] += loss
        else:
            raise NotImplementedError(f"Mode ({mode}) is not supported!")


    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)


    def _check_save_interval(self):
        if self.steps and (self.steps % self.config["save_interval_steps"] == 0):
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")


    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()


    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)


    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

