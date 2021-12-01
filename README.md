## A Robust Matching Network for Gradually Estimating Geometric Transformation on Remote Sensing Imagery

We propose a matching network for gradually estimating the geometric transformation parameters between two aerial images taken in the same area but in different environments. To precisely matching two aerial images, there are important factors to consider such as different time, a variation of viewpoint, size, and rotation.

This paper has been accepted in SMC 2019. [[SMC version](https://ieeexplore.ieee.org/document/8913881)]


## Requirements
```
  python==3.6.8
  torch==1.0.1
  torchvision=0.2.2
  PyQt5==5.14 (for gui)
  opencv==3.4.1 (for gui)
```  
## Run
```
  python demo.py
  python gui.py (recommended)
```  
## Trained models

Save the files in ./trained_models folder

Download link : [ResNet models](https://drive.google.com/file/d/1au049oWWxio9Pgowo4Rias9knL_yiNth/view?usp=sharing, "trained models link")

## Citation
```
@inproceedings{kim2019robust,
  title={A Robust Matching Network for Gradually Estimating Geometric Transformation on Remote Sensing Imagery},
  author={Kim, Dong-Geon and Nam, Woo-Jeoung and Lee, Seong-Whan},
  booktitle={2019 IEEE International Conference on Systems, Man and Cybernetics (SMC)},
  pages={3889--3894},
  year={2019},
  organization={IEEE}
}
```  

## Contact

If you need more performance, further research or issues, please feel free to contact me anytime.

I currently research the NAS (Nerual Architecture Search) to create a network suitable for specific image domain.

E-mail : <dgkim0813@gmail.com>


## Screenshot for running gui.py 

Support real-time matching and overlay.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11848064/75682221-670fc700-5cd8-11ea-9691-ac2d33f92fe2.gif" />

