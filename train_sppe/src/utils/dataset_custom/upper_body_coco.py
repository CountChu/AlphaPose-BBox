import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt


class UpperBodyCoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/mydata/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 7
        self.nJoints = 7

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7)
        self.flipRef = ((2, 5), (3, 6), (4, 7))

        # create train/val split
        with h5py.File('../data/mydata/annot_data.h5', 'r') as annot:
            # train
            self.imgname_coco_train = annot['imgname'][:-8]
            self.bndbox_coco_train = annot['bndbox'][:-8]
            self.part_coco_train = annot['part'][:-8]
            # val
            self.imgname_coco_val = annot['imgname'][-8:]
            self.bndbox_coco_val = annot['bndbox'][-8:]
            self.part_coco_val = annot['part'][-8:]

        self.size_train = self.imgname_coco_train.shape[0]
        self.size_val = self.imgname_coco_val.shape[0]

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]

        imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
        img_path = os.path.join(self.img_folder, imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'coco', sf, self, train=self.is_train, nJoints_coco=self.nJoints_coco)

        inp, out, setMask = metaData

        return inp, out, setMask, 'coco'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
