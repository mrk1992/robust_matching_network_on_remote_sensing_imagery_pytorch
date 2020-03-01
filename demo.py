from __future__ import print_function, division

import torch
from torch.autograd import Variable
from torchvision.transforms import Normalize

from model.cnn_geometric_model import CNNGeometricPearson
from image.normalization import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars, str_to_bool
# from util.checkboard import createCheckBoard
from geotnf.transformation import GeometricTnf
# from geotnf.point_tnf import *
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
import warnings
from collections import OrderedDict

import pickle
from functools import partial

import time
start_time = time.time()

warnings.filterwarnings('ignore')

# torch.cuda.set_device(1)

### Parameter
feature_extraction_cnn = 'resnet101'

if feature_extraction_cnn=='vgg':
    model_homo_path = ''
elif feature_extraction_cnn=='resnet101':
    model_aff_path = 'trained_models/resnet36_myproc_1_new_cor_fefr_4p5.pth.tar'
    model_aff_path2 = 'trained_models/resnet101_epo81_lr4p4_rm11.pth.tar'

target_image_path='datasets/tgt15.jpg'
source_image_path='datasets/src15.jpg'

### Load models
use_cuda = torch.cuda.is_available()
do_aff = not model_aff_path2 == ''\

# Create model
print('Creating CNN model...')
if do_aff:
    model_aff = CNNGeometricPearson(use_cuda=use_cuda, geometric_model='affine', feature_extraction_cnn=feature_extraction_cnn)\

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

# Load trained weights
print('Loading trained model weights...')
if do_aff:
    checkpoint = torch.load(model_aff_path, map_location=lambda storage, loc: storage)
    checkpoint2 = torch.load(model_aff_path2, map_location=lambda storage, loc: storage)
    model_dict = model_aff.FeatureExtraction.state_dict()
    for name, param in model_dict.items():
        model_dict[name].copy_(checkpoint['state_dict'][
                                   'FeatureExtraction.' + name])
    model_dict = model_aff.FeatureClassification.state_dict()
    for name, param in model_dict.items():
        model_dict[name].copy_(checkpoint['state_dict'][
                                   'FeatureClassification.' + name])
    model_dict = model_aff.FeatureExtraction2.state_dict()
    for name, param in model_dict.items():
        model_dict[name].copy_(checkpoint2['state_dict'][
                                   'FeatureExtraction.' + name])
    model_dict = model_aff.FeatureRegression.state_dict()
    for name, param in model_dict.items():
        model_dict[name].copy_(checkpoint2['state_dict'][
                                   'FeatureRegression.' + name])
### Create image transformers
affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)

### Load and preprocess images
resizeCNN = GeometricTnf(out_h=240, out_w=240, use_cuda=False)
affTnf_origin = GeometricTnf(out_h=1080, out_w=1080, use_cuda=False)
affTnf_Demo = GeometricTnf(out_h=540, out_w=540, use_cuda=False)
normalizeTnf = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def Im2Tensor(image):
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32) / 255.0)
    image_var = Variable(image, requires_grad=False)

    if use_cuda:
        image_var = image_var.cuda()
    return image_var

def preprocess_image(image):
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32) / 255.0)
    image_var = Variable(image, requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var

def preprocess_image_Demo(image):
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32) / 255.0)
    image_var = Variable(image, requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = affTnf_Demo(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var

def preprocess_image_Origin(image):
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32) / 255.0)
    image_var = Variable(image, requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = affTnf_origin(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var

source_image = io.imread(source_image_path)
target_image = io.imread(target_image_path)

source_image_var = preprocess_image(source_image)
source_image_var_orgin = preprocess_image_Origin(source_image)
source_image_var_demo = preprocess_image_Demo(source_image)
target_image_var = preprocess_image(target_image)
target_image = np.float32(target_image/255.)

if use_cuda:
    source_image_var = source_image_var.cuda()
    source_image_var_demo = source_image_var_demo.cuda()
    source_image_var_orgin = source_image_var_orgin.cuda()
    target_image_var = target_image_var.cuda()

batch = {'source_image': source_image_var, 'target_image':target_image_var, 'source_image_demo':source_image_var_demo, 'origin_image':source_image_var_orgin}

resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda = use_cuda)
resizeTgt_demo = GeometricTnf(out_h=540, out_w=540, use_cuda = use_cuda)

### Evaluate model
if do_aff:
    model_aff.eval()

# Evaluate models
if do_aff:
    theta_aff = model_aff(batch)
    warped_image_aff = affTnf(batch['source_image'], theta_aff.view(-1, 2, 3))

### Process result
if do_aff:
    result_aff = affTnf(Im2Tensor(source_image), theta_aff.view(-1,2,3))
    warped_image_aff_np = resizeTgt(result_aff).squeeze(0).transpose(0,1).transpose(1,2).cpu().detach().numpy()
    # io.imsave('results/aff.jpg', warped_image_aff_np)
    result_aff_demo = affTnf_Demo(Im2Tensor(source_image), theta_aff.view(-1,2,3))
    warped_image_aff_np_demo = resizeTgt_demo(result_aff_demo).squeeze(0).transpose(0,1).transpose(1,2).cpu().detach().numpy()
    io.imsave('aff_demo.jpg', warped_image_aff_np_demo)

print()
print("# ====================================== #")
print("#            <Execution Time>            #")
print("#            - %.4s seconds -            #" %(time.time() - start_time))
print("# ====================================== #")

# Create checkboard
if do_aff:
    aff_checkboard = createCheckBoard(warped_image_aff_np, target_image)
    io.imsave('aff_checkboard.jpg', aff_checkboard)


N_subplots = 3
fig, axs = plt.subplots(1, N_subplots)
axs[0].imshow(source_image)
axs[0].set_title('src')
axs[1].imshow(target_image)
axs[1].set_title('tgt')
axs[2].imshow(warped_image_aff_np)
axs[2].set_title('aff')
for i in range(N_subplots):
    axs[i].axis('off')
plt.show()
