import os
import pickle
from collections import OrderedDict

import torch
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from data.base_dataset import BaseDataset, get_params, get_transform

import numpy as np
from PIL import Image

with open('opts.pkl', 'rb') as f:
    opt = pickle.load(f)

class Muse():
    def __init__(self):
        self.model = Pix2PixModel(opt)
        self.model.eval()

    def post_process(self, img):
        img = img.detach().cpu().float().numpy().squeeze()
        image_numpy = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        return image_numpy.astype(np.uint8)
    
    def preprocess(self, label):
#         params = get_params(opt, label.size)
#         transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
#         label_tensor = transform_label(label) * 255.0
        label_tensor = torch.FloatTensor(label)[None]
        label_tensor[label_tensor == 255] = opt.label_nc  # 'unknown' is opt.label_nc

        # create one-hot label map
        label_map = label_tensor[None].long()
        bs, _, h, w = label_map.size()
        nc = 184
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        return input_semantics
        
    def infer(self, x):
        semantics = self.preprocess(x)
        with torch.no_grad():
            pred = self.model.netG(semantics, z=None)
            image_pred = self.post_process(pred)
            return image_pred