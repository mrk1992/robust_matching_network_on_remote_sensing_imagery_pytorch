from __future__ import print_function, division
import torch
import os
from os.path import exists, join, basename
from skimage import io
import pandas as pd
import numpy as np
from numpy.linalg import inv
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable

class SynthDataset(Dataset):
    """
    
    Synthetically transformed pairs dataset for training with strong supervision
    
    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
            
    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation}
            
    """
    # Test : 240
    # Training : 1080
    # def __init__(self, csv_file, training_image_path, output_size=(240,240), geometric_model='affine', transform=None,
    # def __init__(self, csv_file, training_image_path, output_size=(1080,1080), geometric_model='affine', transform=None,
    def __init__(self, csv_file, training_image_path, output_size=(540,540), geometric_model='affine', transform=None,
    # def __init__(self, csv_file, training_image_path, output_size=(2160,2160), geometric_model='affine', transform=None,
                 random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1/6, random_t_tps=0.4): # Original random_s=0.5, random_alpha = 1/6
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(csv_file)
        self.img_names = self.train_data.iloc[:,0]
        self.img_names2 = self.train_data.iloc[:,1]
        self.theta_array = self.train_data.iloc[:, 2:].as_matrix().astype('float')
        # copy arguments
        self.training_image_path = training_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False)
        # Ready for distance
        grid_size = 20
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        X, Y = np.meshgrid(axis_coords, axis_coords)
        X = np.reshape(X, (1, self.N))
        Y = np.reshape(Y, (1, self.N))
        self.P = np.concatenate((X, Y))
        # Check tilt
        self.cosVal = [1, 0, -1, 0]
        self.sinVal = [0, 1, 0, -1]
        self.tilt = [1, 1 / np.cos(7/18 * np.pi)]
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # read image
        img_name = os.path.join(self.training_image_path, self.img_names[idx])
        image = io.imread(img_name)

        img_name2 = os.path.join(self.training_image_path, self.img_names2[idx])
        image2 = io.imread(img_name2)
        
        # read theta
        if self.random_sample==False:
            theta = self.theta_array[idx, :]

            if self.geometric_model=='affine':
                # reshape theta to 2x3 matrix [A|t] where 
                # first row corresponds to X and second to Y
                theta = theta[[3,2,5,1,0,4]].reshape(2,3)
            elif self.geometric_model=='tps':
                theta = np.expand_dims(np.expand_dims(theta,1),2)
        else:
            if self.geometric_model=='affine':
                alpha = (np.random.rand(1)-0.5)*2*np.pi*self.random_alpha
                theta = np.random.rand(6)
                theta[[2,5]]=(theta[[2,5]]-0.5)*2*self.random_t
                theta[0]=(1+(theta[0]-0.5)*2*self.random_s)*np.cos(alpha)
                theta[1]=(1+(theta[1]-0.5)*2*self.random_s)*(-np.sin(alpha))
                theta[3]=(1+(theta[3]-0.5)*2*self.random_s)*np.sin(alpha)
                theta[4]=(1+(theta[4]-0.5)*2*self.random_s)*np.cos(alpha)
                theta = theta.reshape(2,3)
            if self.geometric_model=='tps':
                theta = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
                theta = theta+(np.random.rand(18)-0.5)*2*self.random_t_tps

        theta_Discrete = {}
        cnt = 1

        for i in range(len(self.tilt)):
            for k in range(len(self.cosVal)):
                value = np.dot(np.array([[self.tilt[i], 0], [0, 1]]),
                               np.array([[self.cosVal[k], -self.sinVal[k]], [self.sinVal[k], self.cosVal[k]]]))
                theta_Discrete['case' + str(cnt)] = value
                cnt += 1

        min_error = 0
        flag = 0
        case_cnt = 1
        for k in theta_Discrete.keys():
            warped_points = np.dot(theta[:, :2], self.P)
            warped_points2 = np.dot(theta_Discrete[k][:, :2], self.P)

            error = np.sum((warped_points[0, :] - warped_points2[0, :]) ** 2 + (
                    warped_points[1, :] - warped_points2[1, :]) ** 2) / len(self.P[1])
            if case_cnt == 1:
                min_error = error
                flag = case_cnt
            else:
                if min_error > error:
                    min_error = error
                    flag = case_cnt
            case_cnt += 1

        # make arrays float tensor for subsequent processing
        image = torch.Tensor(image.astype(np.float32))
        image2 = torch.Tensor(image2.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))

        # permute order of image to CHW
        image = image.transpose(1,2).transpose(0,1)
        image2 = image2.transpose(1,2).transpose(0,1)

        # Resize image using bilinear sampling with identity affine tnf
        if image.size()[0]!=self.out_h or image.size()[1]!=self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0),requires_grad=False),None).data.squeeze(0)
            image2 = self.affineTnf(Variable(image2.unsqueeze(0),requires_grad=False),None).data.squeeze(0)

        sample = {'image': image, 'image2': image2, 'theta': theta}
        if self.transform:
            sample = self.transform(sample)

        return sample


class SynthDataset2(Dataset):
    """

    Synthetically transformed pairs dataset for training with strong supervision

    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation}

    """

    # Test : 240
    # Training : 1080
    def __init__(self, csv_file, training_image_path, output_size=(240, 240), geometric_model='affine', transform=None, # 1080, 1080
                 random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1 / 6,
                 random_t_tps=0.4):
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(csv_file)
        self.img_names = self.train_data.iloc[:, 0]
        self.img_names2 = self.train_data.iloc[:, 1]
        self.theta_array = self.train_data.iloc[:, 2:].as_matrix().astype('float')
        # copy arguments
        self.training_image_path = training_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)
        # Ready for distance
        grid_size = 20
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        X, Y = np.meshgrid(axis_coords, axis_coords)
        X = np.reshape(X, (1, self.N))
        Y = np.reshape(Y, (1, self.N))
        self.P = np.concatenate((X, Y))
        # Check tilt
        self.cosVal = [1, 0, -1, 0]
        self.sinVal = [0, 1, 0, -1]
        self.tilt = [1, 1 / np.cos(7 / 18 * np.pi)]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # read image
        # img_name = os.path.join()
        image = io.imread(self.training_image_path)
        # img_name2 = os.path.join()
        image2 = io.imread(self.training_image_path)

        # read theta
        if self.random_sample == False:
            theta = self.theta_array[idx, :]

            if self.geometric_model == 'affine':
                # reshape theta to 2x3 matrix [A|t] where
                # first row corresponds to X and second to Y
                theta = theta[[3, 2, 5, 1, 0, 4]].reshape(2, 3)
            elif self.geometric_model == 'tps':
                theta = np.expand_dims(np.expand_dims(theta, 1), 2)
        else:
            if self.geometric_model == 'affine':
                alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * self.random_alpha
                theta = np.random.rand(6)
                theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * self.random_t
                theta[0] = (1 + (theta[0] - 0.5) * 2 * self.random_s) * np.cos(alpha)
                theta[1] = (1 + (theta[1] - 0.5) * 2 * self.random_s) * (-np.sin(alpha))
                theta[3] = (1 + (theta[3] - 0.5) * 2 * self.random_s) * np.sin(alpha)
                theta[4] = (1 + (theta[4] - 0.5) * 2 * self.random_s) * np.cos(alpha)
                theta = theta.reshape(2, 3)
            if self.geometric_model == 'tps':
                theta = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
                theta = theta + (np.random.rand(18) - 0.5) * 2 * self.random_t_tps

        theta_Discrete = {}
        cnt = 1

        for i in range(len(self.tilt)):
            for k in range(len(self.cosVal)):
                value = np.dot(np.array([[self.tilt[i], 0], [0, 1]]),
                               np.array([[self.cosVal[k], -self.sinVal[k]], [self.sinVal[k], self.cosVal[k]]]))
                theta_Discrete['case' + str(cnt)] = value
                cnt += 1

        min_error = 0
        flag = 0
        case_cnt = 1
        for k in theta_Discrete.keys():
            warped_points = np.dot(theta[:, :2], self.P)
            warped_points2 = np.dot(theta_Discrete[k][:, :2], self.P)

            error = np.sum((warped_points[0, :] - warped_points2[0, :]) ** 2 + (
                    warped_points[1, :] - warped_points2[1, :]) ** 2) / len(self.P[1])
            if case_cnt == 1:
                min_error = error
                flag = case_cnt
            else:
                if min_error > error:
                    min_error = error
                    flag = case_cnt
            case_cnt += 1

        # make arrays float tensor for subsequent processing
        image = torch.Tensor(image.astype(np.float32))
        image2 = torch.Tensor(image2.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))

        # permute order of image to CHW
        image = image.transpose(1, 2).transpose(0, 1)
        image2 = image2.transpose(1, 2).transpose(0, 1)

        # Resize image using bilinear sampling with identity affine tnf
        if image.size()[0] != self.out_h or image.size()[1] != self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0), requires_grad=False), None).data.squeeze(0)
            image2 = self.affineTnf(Variable(image2.unsqueeze(0), requires_grad=False), None).data.squeeze(0)

        sample = {'image': image, 'image2': image2, 'theta': theta}

        if self.transform:
            sample = self.transform(sample)

        return sample