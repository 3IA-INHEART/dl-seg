
import pytorch_lightning as pl
import numpy as np
import segmentation_models_pytorch_3d as smp
import torch
from dl_seg_med.utils.registries import loss_registery, optimizers_registery, schedulers_registry
from dl_seg_med.training.metrics import Accuracy, AveragePrecision, DiceCoefficient, MeanIoU, PSNR, MSE
from dl_seg_med.data.ptl.scans_dataset import ScanLearningDataset, ScansDataset, UniqueShapeScanDataset


class lightningSeg3dModel(pl.LightningModule):
    
    def __init__(self, metrics=[Accuracy(0.5), AveragePrecision(), DiceCoefficient(), MeanIoU()], **kwargs):
                
        super(lightningSeg3dModel, self).__init__()
        self.save_hyperparameters()       
        # Create model
        self.model = smp.create_model(arch=self.hparams["arch"], encoder_name=self.hparams["encoder_name"], encoder_weights=self.hparams["encoder_weights"],
                                      in_channels=self.hparams["in_channels"], classes=self.hparams["classes"], decoder_channels=self.hparams["decoder_channels"],
                                      encoder_depth=len(self.hparams["decoder_channels"]),
                                      decoder_use_batchnorm=self.hparams["decoder_use_batchnorm"], decoder_attention_type=self.hparams["decoder_attention_type"])
        
        self.lr = self.hparams["lr"]
        self.batch_size = self.hparams["batch_size"]
        self.class_weights=self.hparams["class_weights"]
        self.metrics = metrics if metrics else []
        self.losses_names = []; self.losses_weight = []; self.losses_callable = []
        for c_loss_obj in self.hparams["LOSSES"]:
            c_loss_name = c_loss_obj["NAME"]; self.losses_names.append(c_loss_name)  
            c_loss_weight = c_loss_obj["WEIGHT"]; self.losses_weight.append(c_loss_weight)  
            c_loss_kwargs = dict(c_loss_obj["ARGS"]) if c_loss_obj["ARGS"] else {}
            c_raw_loss = loss_registery.get(c_loss_name)
            self.losses_callable.append(
                c_raw_loss( **c_loss_kwargs )
            )
        
        self.is_defined_dataset = False
        
    
    def _set_datasets(self, path, val_split_ratio=0.25, manual_seed=None):
        dataset = ScanLearningDataset(UniqueShapeScanDataset(ScansDataset(path)), preprocessing=self.get_preprocessing_fn())
        train_len = int(val_split_ratio * len(dataset)); valid_len = len(dataset)-train_len;
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [train_len, valid_len], generator=torch.Generator().manual_seed(manual_seed))
        # Sample elements for visualization
        self.sample_val = self.train_dataset[0]
        self.sample_train = self.valid_dataset[0]        
        self.is_defined_dataset = True
    
    def train_dataloader(self):
        if not self.is_defined_dataset:
            self._set_datasets(**self.hparams["DATASET"])
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
    
    def val_dataloader(self):
        if not self.is_defined_dataset:
            self._set_datasets(**self.hparams["DATASET"])
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
    
    def forward(self, images_batch):
        # Forward pass
        image_out = self.model(images_batch)
        return image_out
        
    def calc_loss(self, logits, labels):
        seperate_losses = []
        for c_l in self.losses_callable:
            seperate_losses.append(c_l(logits, labels))
        total_loss = sum([ c_w * c_l for c_w, c_l in zip(self.losses_weight, seperate_losses) ]) / sum(self.losses_weight)
        return total_loss, seperate_losses
    
    def calc_metrics_scores(self, logits, labels):        
        
        scores = {
            str(metric_callable): metric_callable(logits.cpu(), labels.cpu()) for metric_callable in self.metrics
        }
        return scores
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        tot_loss, seperate_losses = self.calc_loss(logits, y)
        
        for loss_name, loss_value in zip(self.losses_names, seperate_losses):
            self.log(f'train_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        self.log('train_loss', tot_loss, on_step=True, on_epoch=True, prog_bar=True)
        return tot_loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        tot_loss, seperate_losses = self.calc_loss(logits, y)
        
        for loss_name, loss_value in zip(self.losses_names, seperate_losses):
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        self.log('val_loss', tot_loss, on_step=True, on_epoch=True)
        return self.calc_metrics_scores(logits, y)

    def predict_step(self, batch, batch_idx=None, dataloader_idx= None):
        return self.forward(batch)
    
    def image_grid(self, x, y, preds, prefix):  
        """
        preds = self.softmax(preds)
        classes = torch.argmax(preds,1)
        bs = preds.shape[0]
                
        for i in range(bs):
            fixed_rgb = (x[i]-x[i].min())/(x[i].max()-x[i].min())
            self.logger.experiment.add_image(prefix+"rgb_{}".format(i),fixed_rgb,self.current_epoch)
            fixed_gt = y[0][i]
            self.logger.experiment.add_image(prefix+"gt_{}".format(i),fixed_gt,self.current_epoch)
            fixed_preds = (preds[i]-preds[i].min())/(preds[i].max()-preds[i].min())
            self.logger.experiment.add_image(prefix+"preds_{}".format(i),fixed_preds,self.current_epoch)
            fixed_preds_class = torch.unsqueeze(classes[i],0)/self.n_classes
            self.logger.experiment.add_image(prefix+"preds_class_{}".format(i),fixed_preds_class,self.current_epoch)
        """
        pass
    
    def validation_epoch_end(self, outputs):
        
        """if hasattr(self, "sample_val"):
            X,Y = self.sample_val
            logits = self.forward(X.to(self.device))            
            self.image_grid(X, Y, logits, "val_")
        if hasattr(self, "sample_train"):
            X,Y = self.sample_train
            logits = self.forward(X.to(self.device))            
            self.image_grid(X, Y, logits, "train_")
        """
        combined_scores = {
            metric_key: np.mean([x[metric_key] for x in outputs])
            for metric_key in outputs[0]
        }
        for metric_key, metric_value in combined_scores.items():
            self.log(f"val_{metric_key}", metric_value, on_epoch=True)
        
        return {"log_scores":combined_scores}
    
    def get_preprocessing_fn(self):
        return id_callable()
        #return smp.encoders.get_preprocessing_fn(self.hparams.encoder_name, self.hparams.encoder_weights)

    def configure_optimizers(self):
        
        optimizer_params = self.hparams["OPTIMIZER"]["PARAMS"]; 
        if not optimizer_params: optimizer_params={};
        optimizer_params["lr"] = self.lr
        optimizer = optimizers_registery.get(self.hparams["OPTIMIZER"]["NAME"])(
            self.model.parameters(), **optimizer_params
        )

        schedulers = []
        if "SCHEDULER" in self.hparams and self.hparams["SCHEDULER"]["USE"]:
            sched = schedulers_registry.get(self.cfg["SCHEDULER"]["NAME"])
            if self.hparams["SCHEDULER"]["PARAMS"] is not None:
                scheduler_params = dict(self.hparams["SCHEDULER"]["PARAMS"])
                scheduler = sched(optimizer, **scheduler_params)
            else:
                scheduler = sched(optimizer)            
            schedulers.append(scheduler)        
        return [optimizer], schedulers

class id_callable:
    def __init__(self):
        pass
    def __call__(self, x):
        return x


 