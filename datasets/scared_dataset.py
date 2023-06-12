from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image  
import random

class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):

        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, 
            # "02",
            # folder[0] + folder[7] + folder[9] + folder[-1],
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            "01",
            folder,
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class SCAREDNAIVEDataset(SCAREDRAWDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDNAIVEDataset, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        inputs = super().__getitem__(index)
        for idx in self.frame_idxs:
            for scale in (1,2,3):
                del inputs[("color", idx, scale)]
                del inputs[("color_aug",idx, scale)]
        for scale in (1,2,3):
            del inputs[('K', scale)]
            del inputs[("inv_K", scale)]

        # image_dpt = inputs[("color", 0, 0)]#torch.Size([3, 256, 320])
        # transform = transforms.Compose([
        #         # transforms.Resize((256, 320)), 
        #         transforms.Resize((640, 800)), 
        #         # transforms.ToTensor()    
        # ])
        # inputs[("color_dpt", 0, 0)]  = transform(image_dpt)
        # inputs[("color_dpt", 0, 0)] = transforms.ToPILImage()(image_dpt) 
        return inputs
