import os
import json
import yaml
from types import SimpleNamespace
from pathlib import Path


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

Loader.add_constructor('!include', Loader.include)

def load_config(config_file_path: str):
    
    extension = Path(config_file_path).suffix.lower()
    
    if extension == ".json":
        with open(config_file_path) as f:
            return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    if extension == ".yaml":
        with open(config_file_path) as f:
            dct = yaml.load(f, Loader)
            return json.loads(json.dumps(dct), object_hook=lambda d: dict(**d))
    
    #_logger.error(f"Config file with format {extension} is not supported")
    raise Exception("Config file format is unknown!")


import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def load_cfg_trainer_params(cfg):
    """
    """

    callbacks = []
    checkpoint_callback = None
    # Checkpoint callback loading
    if cfg["MODEL_CHECKPOINT_CALLBACK"]["USE"]:
        if cfg["MODEL_CHECKPOINT_CALLBACK"]["PARAMS"] is not None:
            os.makedirs(cfg["MODEL_CHECKPOINT_CALLBACK"]["PARAMS"]["dirpath"], exist_ok=True)
            save_params = dict(cfg["MODEL_CHECKPOINT_CALLBACK"]["PARAMS"])
            checkpoint_callback = (
                ModelCheckpoint(**save_params)
            )
        else:
            raise Exception("ModelCheckpoint parameters missing!")   
    
    # EarlyStopping callback loading
    if cfg["EARLY_STOPPING"]["USE"]:
        if cfg["EARLY_STOPPING"]["PARAMS"] is not None:
            stopping_params = vars(cfg["EARLY_STOPPING"]["PARAMS"])
            callbacks.append(
                EarlyStopping(**stopping_params)
            )
        else:
            raise Exception("EarlyStopping parameters missing!")   

    # Logger loading
    logger=None
    if cfg["TENSORBOARD_LOGGER"]["USE"]:
        os.makedirs(cfg["TENSORBOARD_LOGGER"]["LOG_PATH"], exist_ok=True)
        logger = TensorBoardLogger(cfg["TENSORBOARD_LOGGER"]["LOG_PATH"], cfg["TENSORBOARD_LOGGER"]["NAME"])

    lightning_params = dict(cfg["LIGHTNING_PARAMS"])

    lightning_params.update(
        {
            "checkpoint_callback": checkpoint_callback,
            "callbacks":callbacks,
            "logger":logger
        }
    )

    return lightning_params