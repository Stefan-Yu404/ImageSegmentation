import torch
import os
import numpy as np
import nibabel as nib
from numpy.random import randint
import random
import sys

class KneeDataset(torch.utils.data.Dataset):
    def __init__(self,patchSize,Path):
        self.x = patchSize[0]
        self.y = patchSize[1]
        self.z = patchSize[2]

        self.images = []
        self.masks = []
        self.Path = Path
        
        image_files = os.listdir(self.Path + "images")
        mask_files = os.listdir(self.Path + "masks")

        for file in image_files:
            rawImage = np.array(nib.load(self.Path +"images/"+file).get_fdata())
            rawImage = rawImage
            rawMask = np.array(nib.load(self.Path + "masks/" +file.replace("img", "msk")).get_fdata())
            rawMask[rawMask ==13] = 4
            rawMask[rawMask ==15] = 9
            rawMask[rawMask ==20] = 11
            if self.z>rawImage.shape[2]:
                rawImage = symmetric_pad_array(rawImage, (rawImage.shape[0], rawImage.shape[1], self.z), pad_value=random.randint(0, 255))
                rawMask = symmetric_pad_array(rawMask, (rawImage.shape[0], rawImage.shape[1], self.z), pad_value=0)

                chunks = rawImage.shape[0]*rawImage.shape[1]*rawImage.shape[2]//(self.x*self.y*self.z)
                row = randint(0, rawImage.shape[0] - self.x, chunks)
                col = randint(0, rawImage.shape[1] - self.y, chunks)
                dep = [0 for i in range(chunks)]
            else:
                row = randint(0, rawImage.shape[0] - self.x, chunks)
                col = randint(0, rawImage.shape[1] - self.y, chunks)
                dep = randint(0, rawImage.shape[2] - self.z, chunks)
            for r,c,d in zip(row, col, dep):
                subImg = rawImage[r: r + self.x, c: c + self.y, d: d + self.z]
                subMsk = rawMask[r: r + self.x, c: c + self.y, d: d + self.z]
                
                self.images.append(subImg)
                self.masks.append(subMsk)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = (self.images[idx]) # w, h, d
        mask = (self.masks[idx])
        
        return image, mask
    
def symmetric_pad_array(input_array: np.ndarray, target_shape: tuple, pad_value: int) -> np.ndarray:

    for dim_in, dim_target in zip(input_array.shape, target_shape):
        if dim_target < dim_in:
            raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

    pad_width = []
    for dim_in, dim_target  in zip(input_array.shape, target_shape):
        if (dim_in-dim_target)%2 == 0:
            pad_width.append((int(abs((dim_in-dim_target)/2)), int(abs((dim_in-dim_target)/2))))
        else:
            pad_width.append((int(abs((dim_in-dim_target)/2)), (int(abs((dim_in-dim_target)/2))+1)))
    
    return np.pad(input_array, pad_width, 'constant', constant_values=pad_value)