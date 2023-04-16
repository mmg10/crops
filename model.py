import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchmetrics.classification import (MulticlassAccuracy, 
                                        MulticlassConfusionMatrix, MulticlassPrecision,
                                        MulticlassRecall, MulticlassF1Score)

from mlxtend.plotting import plot_confusion_matrix

class NNModule(pl.LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        class_names
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.class_names = class_names
        # loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        self.train_acc = MulticlassAccuracy(num_classes=30)
        self.val_acc = MulticlassAccuracy(num_classes=30)
        self.test_acc = MulticlassAccuracy(num_classes=30)

        self.train_pr = MulticlassPrecision(num_classes=30)
        self.val_pr = MulticlassPrecision(num_classes=30)
        self.test_pr = MulticlassPrecision(num_classes=30)

        self.train_rec = MulticlassRecall(num_classes=30)
        self.val_rec = MulticlassRecall(num_classes=30)
        self.test_rec = MulticlassRecall(num_classes=30)

        self.train_f1 = MulticlassF1Score(num_classes=30)
        self.val_f1 = MulticlassF1Score(num_classes=30)
        self.test_f1 = MulticlassF1Score(num_classes=30)
        self.conf_mat = MulticlassConfusionMatrix(num_classes=30)


                
    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        train_loss = self.loss(logits, y)
        train_acc = self.train_acc(preds, y)
        train_pr = self.train_pr(preds, y)
        train_rec = self.train_rec(preds, y)
        train_f1 = self.train_f1(preds, y)
        # self.log('train_acc_step', train_acc)
        # self.log('train_loss_step', train_loss)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        val_loss = self.loss(logits, y)
        val_acc = self.val_acc(preds, y)
        val_pr = self.val_pr(preds, y)
        val_rec = self.val_rec(preds, y)
        val_f1 = self.val_f1(preds, y)
        # self.log('val_acc_step', val_acc)
        # self.log('val_loss_step', val_loss)
        return {"loss": val_loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        test_loss = self.loss(logits, y)
        test_acc = self.test_acc(preds, y)
        test_pr = self.test_pr(preds, y)
        test_rec = self.test_rec(preds, y)
        test_f1 = self.test_f1(preds, y)
        # self.log('test_acc_step', test_acc)
        # self.log('test_loss_step', test_loss)
        return {"loss": test_loss, "test_preds": preds, "test_targ": y}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('valid/loss_epoch', avg_val_loss, sync_dist=True)
        self.log('valid/acc_epoch', self.val_acc.compute(), sync_dist=True)
        self.val_acc.reset()
        self.log('valid/pr_epoch', self.val_pr.compute(), sync_dist=True)
        self.log('valid/rec_epoch', self.val_rec.compute(), sync_dist=True)
        self.log('valid/f1_epoch', self.val_f1.compute(), sync_dist=True)
        self.val_pr.reset()
        self.val_rec.reset()
        self.val_f1.reset()

        
    def test_epoch_end(self, outputs):
        avg_test_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('test/loss_epoch', avg_test_loss, sync_dist=True)
        self.log('test/acc_epoch', self.test_acc.compute(), sync_dist=True)
        self.test_acc.reset()
        self.log('test/pr_epoch', self.test_pr.compute(), sync_dist=True)
        self.log('test/rec_epoch', self.test_rec.compute(), sync_dist=True)
        self.log('test/f1_epoch', self.test_f1.compute(), sync_dist=True)
        self.test_pr.reset()
        self.test_rec.reset()
        self.test_f1.reset()
        preds = torch.cat([x['test_preds'] for x in outputs])
        targs = torch.cat([x['test_targ'] for x in outputs])       
        confmat = self.conf_mat(preds, targs).cpu()

        # 3. Plot the confusion matrix
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat.numpy(), # matplotlib likes working with NumPy 
            class_names=self.class_names, # turn the row and column labels into class names
            figsize=(10, 7)
        );

        
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log('train/loss_epoch', avg_train_loss, sync_dist=True)
        self.log('train/acc_epoch', self.train_acc.compute(), sync_dist=True)
        self.train_acc.reset()
        self.log('train/pr_epoch', self.train_pr.compute(), sync_dist=True)
        self.log('train/rec_epoch', self.train_rec.compute(), sync_dist=True)
        self.log('train/f1_epoch', self.train_f1.compute(), sync_dist=True)
        self.train_pr.reset()
        self.train_rec.reset()
        self.train_f1.reset()


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return {
            "optimizer": torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9),
        }