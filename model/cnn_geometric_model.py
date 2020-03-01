from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from geotnf.transformation import SynthSingleTnf
import numpy as np


class FeatureExtraction(torch.nn.Module):
    def __init__(self, use_cuda=True, feature_extraction_cnn='vgg', last_layer=''):
        super(FeatureExtraction, self).__init__()
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        # freeze parameters
        for param in self.model.parameters():
            # param.requires_grad = False
            param.requires_grad = True
        # move to GPU
        if use_cuda:
            self.model.cuda()

    def forward(self, image_batch):
        return self.model(image_batch)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class Feature2Pearson(torch.nn.Module):
    def __init__(self):
        super(Feature2Pearson, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        feature_mean = torch.mean(feature, 1, True)
        pearson = feature - feature_mean
        norm = torch.pow(torch.sum(torch.pow(pearson, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(pearson, norm)

class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        # Existed ver
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)

        # else:
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)

        return correlation_tensor


class FeatureMasking(torch.nn.Module):
    def __init__(self):
        super(FeatureMasking, self).__init__()
    #
    def forward(self, correlation_tensor):
        correlation_tensor = correlation_tensor.transpose(1, 2).transpose(2, 3)
        l = 11
        h = 15
        w = 15
        limit_region = np.zeros((w, h, w * h))
        for i in range(h):
            for j in range(w):
                for r_h in range(-1 * l, l + 1):
                    for r_w in range(-1 * l, l + 1):
                        temp_col = j + r_w
                        temp_raw = i + r_h
                        if temp_col in range(w) and temp_raw in range(h):
                            limit_region[i][j][w * (temp_col) + temp_raw] = 1
        cor_mask = torch.unsqueeze(Variable(torch.FloatTensor(limit_region), requires_grad=False), 0)
        cor_mask = cor_mask.cuda()
        correlation_tensor = correlation_tensor * cor_mask
        correlation_tensor = correlation_tensor.transpose(2, 3).transpose(1, 2)

        return correlation_tensor



class FeatureClassification(nn.Module):
    def __init__(self, output_dim=4, use_cuda=True):
        super(FeatureClassification, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(225, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(225, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class FeatureRegression2(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(FeatureRegression2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(225, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class CNNGeometricPearson(nn.Module):
    def __init__(self, geometric_model='affine', normalize_features=True, normalize_matches=True,
                 batch_normalization=True, use_cuda=True, feature_extraction_cnn='vgg'):
        super(CNNGeometricPearson, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.FeatureExtraction = FeatureExtraction(use_cuda=self.use_cuda,
                                                   feature_extraction_cnn=feature_extraction_cnn)
        self.FeatureExtraction2 = FeatureExtraction(use_cuda=self.use_cuda,
                                                   feature_extraction_cnn=feature_extraction_cnn)
        self.Feature2Pearson = Feature2Pearson()
        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureMasking = FeatureMasking()
        self.FeatureCorrelation = FeatureCorrelation()
        if geometric_model == 'affine':
            output_dim = 6
        self.FeatureClassification = FeatureClassification(8,use_cuda=self.use_cuda)
        self.FeatureRegression = FeatureRegression(output_dim, use_cuda=self.use_cuda)
        self.ReLU = nn.ReLU(inplace=True)

        self.single_generation_tnf = SynthSingleTnf(use_cuda=self.use_cuda, geometric_model=geometric_model, output_size = (240,240))

    def forward(self, tnf_batch):
        # do feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        # normalize
        if self.normalize_features:
            feature_A = self.Feature2Pearson(feature_A)
            feature_B = self.Feature2Pearson(feature_B)
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.ReLU(correlation))
        theta = self.FeatureClassification(correlation)

        _, predicted = torch.max(theta, 1)
        predicted = predicted.cpu().numpy()
        # 45 deg classification
        if predicted == 0:
            theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)
            # print ('middle result', theta)
        elif predicted == 1:
            theta = torch.tensor([0.70710678118, -0.70710678118, 0, 0.70710678118, 0.70710678118, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)
            # print ('middle result', theta)
        elif predicted == 2:
            theta = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)
            # print ('middle result', theta)
        elif predicted == 3:
            theta = torch.tensor([-0.70710678118, -0.70710678118, 0, 0.70710678118, -0.70710678118, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)
            # print ('middle result', theta)
        elif predicted == 4: # 180
            theta = torch.tensor([-1, 0, 0, 0, -1, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)
            # print ('middle result', theta)
        elif predicted == 5: # 225
            theta = torch.tensor([-0.70710678118, 0.70710678118, 0, -0.70710678118, -0.70710678118, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)
            # print ('middle result', theta)
        elif predicted == 6:
            theta = torch.tensor([0, 1, 0, -1, 0, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)
            # print ('middle result', theta)
        else:
            theta = torch.tensor([0.70710678118, 0.70710678118, 0, -0.70710678118, 0.70710678118, 0], dtype=torch.float).cuda()
            theta = Variable(theta, requires_grad=False)

        # Session 2
        theta1 = theta.view(-1, 2, 3)
        warped_image_batch2 = self.single_generation_tnf(tnf_batch['origin_image'], theta1)
        feature_A2 = self.FeatureExtraction2(warped_image_batch2)
        feature_B2 = self.FeatureExtraction2(tnf_batch['target_image'])

        feature_A2 = self.Feature2Pearson(feature_A2)
        feature_B2 = self.Feature2Pearson(feature_B2)

        correlation2 = self.FeatureCorrelation(feature_A2, feature_B2)
        if self.normalize_matches:
            correlation2 = self.FeatureL2Norm(self.ReLU(correlation2))
        correlation2 = self.FeatureMasking(correlation2)
        theta2 = self.FeatureRegression(correlation2)

        theta = theta.view(-1, 2, 3)
        theta2 = theta2.view(-1, 2, 3)
        theta_r1, theta_r2 = torch.chunk(theta, 2, dim=1)
        theta_r1 = theta_r1.type(torch.FloatTensor).cuda()
        theta_r2 = theta_r2.type(torch.FloatTensor).cuda()
        ho_last = torch.tensor([0,0,1]).type(torch.FloatTensor).cuda()
        theta_f = torch.cat([theta_r1, theta_r2, ho_last.expand_as(theta_r1)], dim = 1)

        theta2_r1, theta2_r2 = torch.chunk(theta2, 2, dim=1)
        theta2_r1 = theta2_r1.type(torch.FloatTensor).cuda()
        theta2_r2 = theta2_r2.type(torch.FloatTensor).cuda()
        ho_last = torch.tensor([0,0,1]).type(torch.FloatTensor).cuda()
        gt_f = torch.cat([theta2_r1, theta2_r2, ho_last.expand_as(theta2_r1)], dim = 1)

        result = torch.bmm(theta_f, gt_f)
        result = result[:,:2,:]

        return result
