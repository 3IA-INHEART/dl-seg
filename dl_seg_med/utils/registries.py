import torch
from dl_seg_med.training.losses import DiceLoss, WeightedCrossEntropyLoss

optimizers_registery = {    
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "LBFGS": torch.optim.LBFGS,    
}

schedulers_registry = {    
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR, 
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 
    "CyclicLR": torch.optim.lr_scheduler.CyclicLR, 
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR, 
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR, 
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau, 
    "StepLR": torch.optim.lr_scheduler.StepLR,     
}


loss_registery = {
    "DiceLoss": DiceLoss,
    "WeightedCrossEntropyLoss":WeightedCrossEntropyLoss
}