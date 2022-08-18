from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset
import imageio
import cv2
import os
from torch.utils.data import DataLoader


# -------------- Dataset ------------- #
# ------------------------------------ #


class TESTDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts
        np.random.seed(8)
        torch.manual_seed(8)
        self.full_img_dir = self.opts.test_dir

        self.pair_list = [
            ['3d930a4ba578463bbde6f7d53cb14e1e.jpg', '0c718004901643259a4fe8275a0b31d1.jpg'],
            ['Tree_Swallow_0002_136792.jpg','Savannah_Sparrow_0051_118574.jpg'],
            ['Brewer_Blackbird_0041_2653.jpg', 'Bewick_Wren_0067_184816.jpg'],
            ['Green_Jay_0114_65841.jpg', 'Elegant_Tern_0090_45924.jpg'],
            ['Purple_Finch_0030_27255.jpg', 'House_Wren_0110_187111.jpg'],
                ]

        self.num_imgs = len(self.pair_list)
        return

    def __len__(self):
        return self.num_imgs

    def get_image_nobbox(self,path):
        img_path = osp.join(self.full_img_dir, path)
        img = imageio.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        width, height, channel = img.shape
        top, bottom, left, right = 0, 0, 0, 0
        if width > height:
            left = (width - height) // 2
            right = left
        else:
            top = (height - width) // 2
            bottom = top
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 0)
        img = cv2.resize(img, (self.opts.img_size, self.opts.img_size))
        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        return img

    def __getitem__(self, index):
        source_name, target_name = self.pair_list[index]
        switch_sig = (np.random.rand(4)<0.5).astype(int)
        img = self.get_image_nobbox(source_name)
        img_t = self.get_image_nobbox(target_name)


        elem = {'img': img, 'index': index, 'name': source_name, 'switch_sig':switch_sig,
                    'img_t': img_t, 'index_t': index, 'name_t': target_name}
        return elem


#----------- Data Loader ----------#
#----------------------------------#

def test_loader(opts, shuffle=False):
    return DataLoader(TESTDataset(opts),
                      batch_size=opts.batch_size,
                      shuffle=shuffle,
                      num_workers=0,
                      drop_last=True)

