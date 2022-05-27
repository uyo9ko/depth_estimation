import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models as models
from loss import *
from evaluate import *
from torchmetrics import StructuralSimilarityIndexMeasure
from collections import OrderedDict
import cv2
import os
import numpy as np


class UpSample(nn.Sequential):
    def __init__(self, input_channel, output_channel):
        super(UpSample, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, input):
        up_x = F.interpolate(
            input[0], size=input[1].size()[2:], mode="bilinear", align_corners=True
        )
        x = self.relu1(self.conv1(torch.cat((up_x, input[1]), dim=1)))
        return self.relu2(self.conv2(x))


class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width=0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(
            num_features, features, kernel_size=1, stride=1, padding=1
        )

        self.up1 = UpSample(features // 1 + 384, features // 2)
        self.up2 = UpSample(features // 2 + 192, features // 4)
        self.up3 = UpSample(features // 4 + 96, features // 8)
        self.up4 = UpSample(features // 8 + 96, features // 16)
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[3],
            features[4],
            features[6],
            features[8],
            features[11],
        )
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1([x_d0, x_block3])
        x_d2 = self.up2([x_d1, x_block2])
        x_d3 = self.up3([x_d2, x_block1])
        x_d4 = self.up4([x_d3, x_block0])
        x_d5 = F.interpolate(
            x_d4,
            size=(x_d4.size()[2] * 2, x_d4.size()[3] * 2),
            mode="bilinear",
            align_corners=True,
        )
        return self.conv3(self.relu3(x_d5))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet161(pretrained=True)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class DenseDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class MyModel(LightningModule):
    def __init__(self, lr, loss_alphas, weight_decay, min_depth, max_depth):
        super().__init__()
        # parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alphas = loss_alphas
        self.save_hyperparameters()

        # loss functions
        self.l1_criterion = nn.L1Loss()
        # self.SSIL = SSIL_Loss()
        # self.ssim = StructuralSimilarityIndexMeasure(data_range=self.max_depth - self.min_depth)
        # self.SiLog = SiLogLoss()

        self.model = DenseDepth()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        image, depth = batch
        train_out = self(image)
        train_loss_depth = self.alphas[0] * self.l1_criterion(train_out, depth)
        # train_SSIL_loss = self.alphas[1] * self.SSIL(train_out, depth)
        train_loss_ssim = self.alphas[2] * (
            1 - ssim(train_out, depth, val_range=self.max_depth - self.min_depth)
        )
        # train_Silog_loss = self.alphas[3] * self.SiLog(train_out.squeeze(), depth.squeeze())
        # train_loss_ssim = torch.clamp((1 - ssim(train_out, depth, val_range=self.max_depth - self.min_depth)) * 0.5, 0, 1)

        train_loss = train_loss_depth + train_loss_ssim
        self.log("l_d", train_loss_depth)
        self.log("l_ssim", train_loss_ssim)
        # self.log('l_ssil', train_SSIL_loss)
        self.log("l_t", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        image, val_depth = batch
        val_out = self(image)
        val_loss_depth = self.alphas[0] * self.l1_criterion(val_out, val_depth)
        # val_SSIL_loss = self.alphas[1] * self.SSIL(val_out, val_depth)
        val_loss_ssim = self.alphas[2] * (
            1 - ssim(val_out, val_depth, val_range=self.max_depth - self.min_depth)
        )
        # val_Silog_loss = self.alphas[3] * self.SiLog(val_out.squeeze(), val_depth.squeeze())
        # val_loss_ssim = torch.clamp((1 - ssim(val_out, val_depth, val_range=self.max_depth - self.min_depth)) * 0.5, 0, 1)

        val_loss = val_loss_depth + val_loss_ssim

        val_depth, val_out = process_depth(
            val_depth, val_out, self.min_depth, self.max_depth
        )
        val_acc = compute_acc(val_depth, val_out)
        self.log("val_acc", val_acc)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        gt, pred = np_process_depth(y, y_hat, self.min_depth, self.max_depth)
        metrics = compute_errors(gt, pred)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
            "monitor": "val_loss",
        }

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
