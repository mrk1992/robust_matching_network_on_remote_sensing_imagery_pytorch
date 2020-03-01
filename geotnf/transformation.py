from __future__ import print_function, division
import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F

class GeometricTnf(object):
    """
    
    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )        
    
    """
    def __init__(self, geometric_model='affine', out_h=240, out_w=240, use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h, out_w)
        elif geometric_model=='tps':
            self.gridGen = TpsGridGen(out_h, out_w, use_cuda=use_cuda)
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        b, c, h, w = image_batch.size()
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3)
            theta_batch = Variable(theta_batch,requires_grad=False)

        sampling_grid = self.gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data*padding_factor*crop_factor
        # sample transformed image, shape 1, 3, 1080, 1080
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)
        
        return warped_image_batch
    

class SynthPairTnf(object):
    """

    Generate a synthetically warped training pair using an affine transformation.

    """
    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9/16, output_size=(240,240), padding_factor = 0.5):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda=use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w,
                                         use_cuda = self.use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w,
                                         use_cuda = self.use_cuda)
        self.tilt = GeometricTnf(geometric_model, 240, 240,
                                         use_cuda = self.use_cuda)

    def __call__(self, batch):
        image_batch, image_batch2, theta_batch= batch['image'], batch['image2'], batch['theta']

        if self.use_cuda:
            image_batch = image_batch.cuda()
            image_batch2 = image_batch2.cuda()
            theta_batch = theta_batch.cuda()

        # convert to variables
        image_batch = Variable(image_batch,requires_grad=False)
        image_batch2 = Variable(image_batch2,requires_grad=False)
        theta_batch = Variable(theta_batch,requires_grad=False)

        # # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch,None,self.padding_factor,self.crop_factor)
        # # # get transformed image
        warped_image_batch = self.geometricTnf(image_batch2,theta_batch,
                                               self.padding_factor,self.crop_factor)

        # Origin
        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch, 'origin_image': image_batch}

    def symmetricImagePad(self,image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h*padding_factor), int(w*padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1))
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1))
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1))
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1))
        if self.use_cuda:
                idx_pad_left = idx_pad_left.cuda()
                idx_pad_right = idx_pad_right.cuda()
                idx_pad_top = idx_pad_top.cuda()
                idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,
                                 image_batch.index_select(3,idx_pad_right)),3)
        image_batch = torch.cat((image_batch.index_select(2,idx_pad_top),image_batch,
                                 image_batch.index_select(2,idx_pad_bottom)),2)
        return image_batch


class SynthSingleTnf(object):
    """

    Generate a synthetically warped training image in Session 2 using an affine transformation.

    """

    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9/16, output_size=(240,240), padding_factor=0.5):  # Original
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size

        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w,  # Original
                                          use_cuda=self.use_cuda)

    # I think not __call__, but forward
    def __call__(self, batch, theta1):
        image_batch, theta_batch = batch, theta1

        if self.use_cuda:
            theta_batch = theta_batch.cuda()

        # generate symmetrically padded image for bigger sampling region
        # convert to variables
        theta_batch = Variable(theta_batch, requires_grad=False)

        # For Demo
        warped_image_batch = self.geometricTnf(image_batch,
                                               theta_batch)  # Identity is not used as theta given # Original

        return warped_image_batch  # Original

    def symmetricImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))
        if self.use_cuda:
            idx_pad_left = idx_pad_left.cuda()
            idx_pad_right = idx_pad_right.cuda()
            idx_pad_top = idx_pad_top.cuda()
            idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), 3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), 2)
        return image_batch

class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)
