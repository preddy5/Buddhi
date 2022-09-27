
import os
import sys
sys.path.append('..'); sys.path.append('.')
from shutil import copytree, ignore_patterns, rmtree
import random
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import src
from src.argparser import args
from src.utils import update_progress, to_one_hot, accuracy
from src.io import get_dataloader
from src.models import BITnet6, BITnet6_v101

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
import json

steps = args.steps
n_hinge = args.steps
img_size = 64
models_dict = {'BITnet6': BITnet6,
               'BITnet6_v101': BITnet6_v101}

class PLmodel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        output, states = model(X, state, epoch=epoch, steps=steps, select_first=True, normalize=False)
        return output, states

    def configure_optimizers(self):
        return 

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        bs = X.shape[0]
        one_hot_label = to_one_hot(y, self.model.num_classes)
        X, one_hot_label_c, y = X.cuda(), one_hot_label.cuda(), y.cuda()
        
        states = 0
        output, states = self.model(X, states, steps=steps, select_first=True, normalize=False)
        output = output.clamp(max=n_hinge)
        output = output/n_hinge
    
        loss = self.criterion(output, one_hot_label_c)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        train_acc1 = acc1.item()/bs
        train_acc5 = acc5.item()/bs
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred1', train_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('pred5', train_acc5, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        accuracy_log = {'pred1':train_acc1, 'pred5':train_acc5}
        return {'loss': loss, 'log': accuracy_log}

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        bs = X.shape[0]
        output, _ = self.model(X, steps=steps, select_first=True)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        test_acc1 = acc1.item()/bs
        test_acc5 = acc5.item()/bs
        self.log('val_pred1', test_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pred5', test_acc5, on_step=False, on_epoch=True, prog_bar=True, logger=True)       
        return test_acc1

        
if __name__ == '__main__':
    model_path = None
    checkpoint_callback = None
    model_save_path = args.checkpoint_dir
    print('Checkpoint directory: ', model_save_path)
    
    if os.path.exists(model_save_path):        
        weights = [os.path.join(model_save_path, x) for x in os.listdir(model_save_path) if '.ckpt' in x]
        weights.sort(key=lambda x: os.path.getmtime(x))
        if len(weights)>0:
            model_path = weights[-1]
            print('loading: ', weights[-1])
        else:
            print('Checkpoint doesnt exist')
            quit()
    else:
        print('Directory doesnt exist')
        quit()
    checkpoint_callback = ModelCheckpoint(
                                save_top_k=1,
                                monitor="val_pred1",
                                mode="max",
                                verbose=True,
                                save_last=True,
                                dirpath=model_save_path,
                                filename="checkpoint-{epoch:02d}-{val_pred1:.4f}",
                            )
    callbacks = [checkpoint_callback, ]
    
    train_loader, val_loader, num_classes, img_size = get_dataloader(args)
    model = models_dict[args.model]
    model = PLmodel(model(num_classes, img_size, args.fs))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(gpus=args.n_gpu, precision=32, strategy="ddp", resume_from_checkpoint=model_path, logger=False, callbacks=callbacks, max_epochs=args.epochs)
    trainer.validate(model, dataloaders=val_loader)
