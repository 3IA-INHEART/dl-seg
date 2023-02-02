import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import click
import torch
from dl_seg_med.training.ptl.segmentation_3d_model import lightningSeg3dModel
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import numpy as np
from dl_seg_med.utils.config_loader import load_config, load_cfg_trainer_params



@click.command()
@click.argument('model_config_file', type=click.Path(exists=True))
@click.argument('train_config_file', type=click.Path(exists=True))
@click.argument('ckpt_dir', type=click.Path(exists=False))
@click.argument('log_dir', type=click.Path(exists=False))
def main(model_config_file, train_config_file, ckpt_dir, log_dir):
    """_summary_
    Args:
        config_file (_type_): _description_
    """
    
    model_cfg = load_config(model_config_file)
    training_cfg = load_config(train_config_file)
    
    light_model = lightningSeg3dModel(**model_cfg)
    
    training_cfg["MODEL_CHECKPOINT_CALLBACK"]["PARAMS"]["dirpath"] = ckpt_dir
    training_cfg["TENSORBOARD_LOGGER"]["LOG_PATH"] = log_dir
    
    training_params = load_cfg_trainer_params(training_cfg)
            
    light_model = light_model.double().cuda()
    
    trainer = pl.Trainer(**training_params)

    if (training_params.get("auto_scale_batch_size") or training_params.get("auto_lr_find")):
        trainer.tune(light_model, train_dataloaders=light_model.train_dataloader() , val_dataloaders=light_model.val_dataloader())
    else:
        trainer.fit(light_model, train_dataloader=light_model.train_dataloader(), val_dataloaders=light_model.val_dataloader())

if __name__ == "__main__":
    main()