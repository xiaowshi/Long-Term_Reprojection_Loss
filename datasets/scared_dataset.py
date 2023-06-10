from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset
from torch.utils.data import Dataset
from PIL import Image

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
            "02",
            folder[0] + folder[7] + folder[9] + folder[-1],
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            "01",
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class SCAREDNAIVEDataset(Dataset):
    def __init__(self, data_path, filenames,):
        self.filenames = filenames
        self.data_path = data_path

    def __getitem__(self, index):
        line = self.filenames[index].split()
        folder = line[0]
        inputs = {}

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        # get image path
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "02",
            folder[0] + folder[7] + folder[9] + folder[-1],
            f_str)
        inputs[("color", 0, 0)] = self.loader(image_path)
        return inputs

    # def get_image_path(self, folder, frame_index, side):
    #     f_str = "{:06d}{}".format(frame_index, self.img_ext)
    #     image_path = os.path.join(
    #         self.data_path,
    #         "02",
    #         folder[0] + folder[7] + folder[9] + folder[-1],
    #         f_str)
    #     return image_path
    
    # def get_depth(self, folder, frame_index, side, do_flip):
    #     f_str = "scene_points{:06d}.tiff".format(frame_index-1)

    #     depth_path = os.path.join(
    #         self.data_path,
    #         "01",
    #         folder,
    #         "image_0{}/data/groundtruth".format(self.side_map[side]),
    #         f_str)

    #     depth_gt = cv2.imread(depth_path, 3)
    #     depth_gt = depth_gt[:, :, 0]
    #     depth_gt = depth_gt[0:1024, :]
    #     if do_flip:
    #         depth_gt = np.fliplr(depth_gt)

    #     return depth_gt

