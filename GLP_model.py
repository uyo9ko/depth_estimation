import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models as models
from loss import *
from evaluate import *
from torchmetrics import StructuralSimilarityIndexMeasure
from mmcv.runner import load_checkpoint
from models.mit import mit_b4
from collections import OrderedDict
import cv2
import os
import numpy as np


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)

        return out


class SelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU())

        self.conv3 = nn.Conv2d(in_channels=int(in_channel / 2), 
                               out_channels=2, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + \
              x_global * attn[:, 1, :, :].unsqueeze(1)

        return out
    
    
class GLPDepth(nn.Module):
    def __init__(self, ckpt_path, max_depth=10.0, is_train=False):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        if is_train:            
            load_checkpoint(self.encoder, ckpt_path, logger=None)
            
        channels_in = [512, 320, 128]
        channels_out = 64
            
        self.decoder = Decoder(channels_in, channels_out)
    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):  
        with torch.no_grad():      
            conv1, conv2, conv3, conv4 = self.encoder(x)
        out = self.decoder(conv1, conv2, conv3, conv4)
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': out_depth}


class MyModel(LightningModule):
    def __init__(self, lr, loss_alphas, weight_decay, min_depth, max_depth, load_ckpt_paths): 
        super().__init__()
        #parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alphas = loss_alphas
        self.load_ckpt_paths = load_ckpt_paths
        self.save_path = '/content/drive/Shareddrives/szh/prediction/fda_0.2'
        self.save_hyperparameters()
        
        #loss functions
        # self.l1_criterion = nn.L1Loss()
        # self.SSIL = SSIL_Loss()
        # self.ssim = StructuralSimilarityIndexMeasure(data_range=self.max_depth - self.min_depth)
        self.SiLog = SiLogLoss()
        
        self.model = GLPDepth(self.load_ckpt_paths[0], max_depth=self.max_depth, is_train=True)
        # model_weight = torch.load(self.load_ckpt_paths[1])
        # if 'module' in next(iter(model_weight.items()))[0]:
        #     model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
        # self.model.load_state_dict(model_weight)
        # print('load GLP model success from {}'.format(self.load_ckpt_paths[1]))
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        image, depth = batch
        train_out = self(image)
        train_out = train_out['pred_d']
        # train_loss_depth = self.alphas[0] * self.l1_criterion(train_out, depth) 
        # train_SSIL_loss = self.alphas[1] * self.SSIL(train_out, depth) 
        # train_loss_ssim = self.alphas[2] * (1 - ssim(train_out, depth, val_range=self.max_depth - self.min_depth)) 
        train_Silog_loss = self.alphas[3] * self.SiLog(train_out.squeeze(), depth.squeeze()) 
        # train_loss_ssim = torch.clamp((1 - ssim(train_out, depth, val_range=self.max_depth - self.min_depth)) * 0.5, 0, 1)

        train_loss =   train_Silog_loss
        # self.log('l_d', train_loss_depth) 
        # self.log('l_ssim', train_loss_ssim)
        # self.log('l_ssil', train_SSIL_loss)
        self.log( 'l_t', train_loss)
        return train_loss
    
        
    def validation_step(self, batch, batch_idx):
        image, val_depth = batch
        val_out = self(image)
        val_out = val_out['pred_d']
        # val_loss_depth = self.alphas[0] * self.l1_criterion(val_out, val_depth) 
        # val_SSIL_loss = self.alphas[1] * self.SSIL(val_out, val_depth)
        # val_loss_ssim = self.alphas[2] * (1 - ssim(val_out, val_depth, val_range=self.max_depth - self.min_depth)) 
        val_Silog_loss = self.alphas[3] * self.SiLog(val_out.squeeze(), val_depth.squeeze())
        # val_loss_ssim = torch.clamp((1 - ssim(val_out, val_depth, val_range=self.max_depth - self.min_depth)) * 0.5, 0, 1)
        
        val_loss =  val_Silog_loss

        val_depth, val_out  = process_depth(val_depth,val_out,self.min_depth, self.max_depth)
        val_acc = compute_acc(val_depth, val_out)
        self.log('val_acc', val_acc)
        self.log("val_loss", val_loss)



    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat['pred_d']
        y = y.cpu().numpy().squeeze()
        y_hat = y_hat.cpu().numpy().squeeze()
        norm_y_hat = cv2.normalize(y_hat, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F) 
        im_color = cv2.applyColorMap(norm_y_hat.astype(np.uint8), cv2.COLORMAP_JET) 
        cv2.imwrite(os.path.join(self.save_path, '{}_pred.png'.format(batch_idx)), im_color)
        norm_y = cv2.normalize(y, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F) 
        im_color = cv2.applyColorMap(norm_y.astype(np.uint8), cv2.COLORMAP_JET) 
        cv2.imwrite(os.path.join(self.save_path, '{}_gt.png'.format(batch_idx)), im_color) 
        x = x.cpu().numpy().squeeze()*255
        x = x.transpose(1,2,0)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.save_path, '{}_img.png'.format(batch_idx)), x)          
        gt, pred = np_process_depth(y, y_hat, self.min_depth, self.max_depth)
        metrics = compute_errors(gt, pred)
        return metrics


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            verbose=True
        )
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }
    
    
    


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)

        return out


class SelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU())

        self.conv3 = nn.Conv2d(in_channels=int(in_channel / 2), 
                               out_channels=2, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + \
              x_global * attn[:, 1, :, :].unsqueeze(1)

        return out
