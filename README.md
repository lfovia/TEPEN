# TEPEN
TEPEN: Towards an Ensemble Model for Pixel-based Embodied Navigation
![Uploading image.png…]()

This work explores the possibility of depth modality to contribute towards pixel navigation. For RGB baseline we have taken the model weights shared in Pixel Navigation paper.
We train a Depth modality model, with similar dataset, as the dataset was not released.
The ensemble of both the models we were able to show improvement in performance in Habitatsim for an agent towards pixel navigation roll out evaluation and zero shot object goal navigation.

This is the official implementation of the paper. Please take a look at the paper for more technical details.

Dependency
-------------
Our project is based on the habitat-sim and habitat-lab. Please follow the guides to install them in your python environment. You can directly install the latest version of habitat-lab and habitat-sim. And make sure you have properly download the navigation scenes (HM3D, MP3D) and the episode dataset for object navigation. Besides, make sure you have installed the following dependencies on your python environment:
numpy, opencv-python, tqdm, openai, torch

Installation
---------------
Firstly, clone our repo as:
git clone https://github.com/lfovia/TEPEN.git
cd TEPEN
Our method depends on an open-vocalbulary detection module GroundingDINO and a segmentation module Segment-Anything. You can either follow their website and follow the installation guide or just enter our /thirdparty directory and install locally as:
cd third_party/GroundingDINO
pip install -e .
cd ../Segment-Anything/
pip install -e .

Pretrained checkpoints
------------------------
Checkpoints for Depth model() , RGBD ensemble model (). Download the D version weights for RGB from Pixel navigation git repo.

To evaluate the Pixel navigation model
-----------------------------------------
For depth model, 
For RGBD model,

To perform zero shot object goal navigation
-----------------------------------------
For depth model, 
For RGBD model,

BibTex
--------

Please cite our paper if you find it helpful :)

