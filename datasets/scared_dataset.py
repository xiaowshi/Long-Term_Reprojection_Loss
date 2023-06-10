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
            # folder,
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


         
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            
class SCAREDNAIVEDataset(Dataset):
    # def __init__(self, *args, **kwargs):
    def __init__(self, data_path, filenames, transform=None, img_ext='.jpg', is_train=False):
        # , height, width, frame_ids, num_scales, is_train=True, img_ext='.jpg', transform=None):
        self.filenames = filenames
        self.data_path = data_path
        self.img_ext = img_ext
        self.loader = pil_loader
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # for k in list(inputs):
        #     frame = inputs[k]
        #     if "color" in k:
        #         # print(k)
        #         n, im, i = k # 
        #         for i in range(4):# num_scales
        #             print("resize:", k, (n, im, i - 1))
        #             inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                print("tensor:", k)
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def get_iamge_path(self, index):
        # get image path
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            # "02",
            # folder[0] + folder[7] + folder[9] + folder[-1],
            f_str)
        return image_path
    
    def __getitem__(self, index):
        inputs = {}
        self.resize = {}
        self.to_tensor = transforms.ToTensor()

        image_path = self.get_iamge_path(index)
        inputs[("color", 0, 0)] = self.loader(image_path)

        do_color_aug = self.is_train and random.random() > 0.5
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
                # self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        
        # for i in [0, -1, 1]: # self.frame_idxs:
        #     if i == "s":
        #         other_side = {"r": "l", "l": "r"}[side]
        #         inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
        #     else:
        #         inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        self.preprocess(inputs, color_aug)
        # for i in [0, -1, 1]:# self.frame_idxs:
        #     del inputs[("color", i, -1)]
        #     del inputs[("color_aug", i, -1)]

        if self.transform:
            inputs[("color", 0, 0)] = self.transform(inputs[("color", 0, 0)])
        print("asdfsadf", inputs.keys())
        return inputs
