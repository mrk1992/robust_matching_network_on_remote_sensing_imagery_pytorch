from __future__ import print_function, division
import argparse
from scipy.misc import imsave
from model.cnn_geometric_model import CNNGeometricPearson
from image.normalization import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars
from geotnf.transformation import GeometricTnf
from skimage import io
import torch
from torch.autograd import Variable
import numpy as np

class matching_demo(object):
    def __init__(self, geometric_model='affine'):
        # Argument parsing
        parser = argparse.ArgumentParser(description='Gradual Estimation for Aerial Image Matching demo script')
        # Paths
        parser.add_argument('--model-aff', type=str,
                            default='trained_models/resnet36_myproc_1_new_cor_fefr_4p5.pth.tar',
                            help='Trained affine model filename')
        parser.add_argument('--model-aff2', type=str,
                            default='trained_models/resnet101_epo81_lr4p4_rm11.pth.tar',
                            help='Trained affine model filename')
        parser.add_argument('--feature-extraction-cnn', type=str, default='resnet101',
                            help='Feature extraction architecture: vgg/resnet101')

        self.args = parser.parse_args()
        self.use_cuda = torch.cuda.is_available()

        self.do_aff = not self.args.model_aff2 == ''
        # Create model
        print('Creating CNN model...')
        if self.do_aff:
            self.model_aff = CNNGeometricPearson(use_cuda=self.use_cuda, geometric_model=geometric_model,
                                     feature_extraction_cnn=self.args.feature_extraction_cnn)

        # Load trained weights
        print('Loading trained model weights...')
        if self.do_aff:
            checkpoint = torch.load(self.args.model_aff, map_location=lambda storage, loc: storage)
            checkpoint2 = torch.load(self.args.model_aff2, map_location=lambda storage, loc: storage)
            model_dict = self.model_aff.FeatureExtraction.state_dict()
            for name, param in model_dict.items():
                model_dict[name].copy_(checkpoint['state_dict'][
                                           'FeatureExtraction.' + name])
            model_dict = self.model_aff.FeatureClassification.state_dict()
            for name, param in model_dict.items():
                model_dict[name].copy_(checkpoint['state_dict'][
                                           'FeatureClassification.' + name])
            model_dict = self.model_aff.FeatureExtraction2.state_dict()
            for name, param in model_dict.items():
                model_dict[name].copy_(checkpoint2['state_dict'][
                                           'FeatureExtraction.' + name])
            model_dict = self.model_aff.FeatureRegression.state_dict()
            for name, param in model_dict.items():
                model_dict[name].copy_(checkpoint2['state_dict'][
                                           'FeatureRegression.' + name])
        self.affTnf = GeometricTnf(geometric_model='affine', out_h=240, out_w=240, use_cuda=False)
        self.affTnf_demo = GeometricTnf(geometric_model='affine', out_h=338, out_w=338, use_cuda=False)
        self.affTnf_origin = GeometricTnf(geometric_model='affine', out_h=480, out_w=480, use_cuda=False)

        self.transform = NormalizeImageDict(['source_image', 'target_image', 'demo', 'origin_image'])
        self.rescalingTnf = GeometricTnf('affine', 240, 240,
                                         use_cuda=True)
        self.geometricTnf = GeometricTnf(geometric_model, 240, 240,
                                         use_cuda=True)


    def __call__(self, fname, fname2):

        image = io.imread(fname)
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)
        image_A = self.affTnf(image_var).data.squeeze(0)
        image_A_demo = self.affTnf_demo(image_var).data.squeeze(0)
        image_A_origin = self.affTnf_origin(image_var).data.squeeze(0)

        image2 = io.imread(fname2)
        image2 = np.expand_dims(image2.transpose((2, 0, 1)), 0)
        image2 = torch.Tensor(image2.astype(np.float32))
        image_var2 = Variable(image2, requires_grad=False)
        image_B = self.affTnf(image_var2).data.squeeze(0)

        sample = {'source_image': image_A, 'target_image': image_B, 'demo': image_A_demo, 'origin_image': image_A_origin}

        sample = self.transform(sample)

        batchTensorToVars = BatchTensorToVars(use_cuda=self.use_cuda)

        batch = batchTensorToVars(sample)
        batch['source_image'] = torch.unsqueeze(batch['source_image'],0)
        batch['target_image'] = torch.unsqueeze(batch['target_image'],0)
        batch['origin_image'] = torch.unsqueeze(batch['origin_image'],0)
        batch['demo'] = torch.unsqueeze(batch['demo'],0)

        if self.do_aff:
            self.model_aff.eval()

        # Evaluate models
        if self.do_aff:
            theta_aff = self.model_aff(batch)
            warped_image_aff_demo = self.affTnf_demo(batch['demo'], theta_aff.view(-1, 2, 3))

        if self.do_aff:
            warped_image_aff_demo = normalize_image(warped_image_aff_demo, forward=False)
            warped_image_aff_demo = warped_image_aff_demo.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()

        print("Done")
        imsave('result.jpg', warped_image_aff_demo)

        return warped_image_aff_demo

