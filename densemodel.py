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
from cov_settings import CovMatrix_ISW
from instance_whitening import *
from dataset import MyDataModule
import tqdm as tqdm

class UpSample(nn.Sequential):
    def __init__(self, input_channel, output_channel):
        super(UpSample, self).__init__()        
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, input):
        up_x = F.interpolate(input[0], size=input[1].size()[2:], mode='bilinear', align_corners=True)
        x = self.relu1(self.conv1(torch.cat((up_x, input[1]), dim=1)))
        return self.relu2(self.conv2(x))


class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width=0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample(features//1 + 384, features//2)
        self.up2 = UpSample(features//2 + 192, features//4)
        self.up3 = UpSample(features//4 + 96, features//8)
        self.up4 = UpSample(features//8 + 96, features//16)
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1([x_d0, x_block3])
        x_d2 = self.up2([x_d1, x_block2])
        x_d3 = self.up3([x_d2, x_block1])
        x_d4 = self.up4([x_d3, x_block0])
        x_d5 = F.interpolate(x_d4, size=(x_d4.size()[2]*2, x_d4.size()[3]*2), mode='bilinear', align_corners=True)
        return self.conv3(self.relu3(x_d5))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet161(pretrained=True)

    def forward(self, x, isw=False):
        features = [x]
        w_arr = []
        for k, v in self.original_model.features._modules.items():
            if isw and (k == "norm0" or k == "denseblock1" or k == "denseblock2"):
                out = v(features[-1])
                out, w = InstanceWhitening(out.shape[1])(out)
                w_arr.append(w)
                features.append(out)
            else:
                features.append(v(features[-1]))
        if isw:
            return features, w_arr
        return features


class MyModel(LightningModule):
    def __init__(self, lr, loss_alphas, weight_decay, min_depth, max_depth, save_png_path): 
        super().__init__()
        self.automatic_optimization = False

        #parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alphas = loss_alphas
        self.save_png_path = save_png_path
        self.eps = 1e-5
        self.save_hyperparameters()

        #modules
        self.encoder = Encoder()
        self.decoder = Decoder()
        #loss functions
        self.l1_criterion = nn.L1Loss()

        self.cov_matrix_layer = []
        in_channel_list = [96, 384, 768]
        for i in range(3):
            self.cov_matrix_layer.append(CovMatrix_ISW(dim=in_channel_list[i], 
                                        relax_denom=0.0, 
                                        clusters=2))

        
    def forward(self, x, isw=False):
        if isw:
            x, w_arr= self.encoder(x, isw=True)
            return self.decoder(x), w_arr
        else:
            x = self.encoder(x)
            return self.decoder(x)
        
    
    def training_step(self, batch, batch_idx):
        if self.current_epoch == 5:
            images, depth = batch
            src_image = torch.chunk(images[0], depth.shape[0], dim=0)
            tgt_image = torch.chunk(images[1], depth.shape[0], dim=0)
            for i in range(depth.shape[0]):
                x = torch.cat([src_image[i],tgt_image[i]], dim=0)
                out, w_arr = self(x, isw=True)
                assert len(w_arr) == 3
                for index, f_map in enumerate(w_arr):
                    # Instance Whitening
                    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
                    HW = H * W
                    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
                    eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
                    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
                    off_diag_elements = f_cor * reverse_eye
                    #print("here", off_diag_elements.shape)
                    self.cov_matrix_layer[index].set_variance_of_covariance(torch.var(off_diag_elements, dim=0))
        else:
            opt = self.optimizers()
            images, depth = batch
            for image in images:
                if self.current_epoch < 5:
                    train_out= self(image)
                else:
                    train_out, w_arr = self(image, isw=True)
                
                    # calculate  wt_loss
                    wt_loss = torch.FloatTensor([0]).cuda()
                    for index, f_map in enumerate(w_arr):
                        eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                        loss = instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov)
                        wt_loss = wt_loss + loss
                    wt_loss = wt_loss / len(w_arr)

                train_loss_depth = self.alphas[0] * self.l1_criterion(train_out, depth) 
                train_loss_ssim = self.alphas[1] * (1 - ssim(train_out, depth, val_range=self.max_depth - self.min_depth)) 
                
                # train_Silog_loss = self.alphas[3] * self.SiLog(train_out.squeeze(), depth.squeeze()) 
                # train_loss_ssim = torch.clamp((1 - ssim(train_out, depth, val_range=self.max_depth - self.min_depth)) * 0.5, 0, 1)

                if self.current_epoch < 5:
                    train_loss =   train_loss_depth + train_loss_ssim
                else:
                    train_loss_wt = self.alphas[2] * wt_loss
                    train_loss =   train_loss_depth + train_loss_ssim + train_loss_wt
                    self.log('l_wt', train_loss_wt)

                self.log('l_d', train_loss_depth) 
                self.log('l_ssim', train_loss_ssim)
                self.log( 'l_t', train_loss)
                opt.zero_grad()
                self.manual_backward(train_loss)
                opt.step()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 5:
            for index in range(len(self.cov_matrix_layer)):
                self.cov_matrix_layer[index].reset_mask_matrix()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 5:
            for index in range(len(self.cov_matrix_layer)):
                self.cov_matrix_layer[index].set_mask_matrix()

       
    def validation_step(self, batch, batch_idx):
        
        images, depth = batch
        val_input = torch.cat(images, dim=0)
        val_depth = torch.cat([depth, depth], dim=0)
        val_out = self(val_input)

        val_loss_depth = self.alphas[0] * self.l1_criterion(val_out, val_depth) 
        val_loss_ssim = self.alphas[1] * (1 - ssim(val_out, val_depth, val_range=self.max_depth - self.min_depth)) 
        # val_Silog_loss = self.alphas[3] * self.SiLog(val_out.squeeze(), val_depth.squeeze())
        # val_loss_ssim = torch.clamp((1 - ssim(val_out, val_depth, val_range=self.max_depth - self.min_depth)) * 0.5, 0, 1)
        
        val_loss =  val_loss_depth + val_loss_ssim

        val_depth, val_out  = process_depth(val_depth, val_out, self.min_depth, self.max_depth)
        val_acc = compute_acc(val_depth, val_out)
        self.log('val_acc', val_acc)
        self.log("val_loss", val_loss)



    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        gt, pred = np_process_depth(y, y_hat, self.min_depth, self.max_depth)
        metrics = compute_errors(gt, pred)

        return metrics


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1,
            verbose=True
        )
 
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
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
        
    



    


