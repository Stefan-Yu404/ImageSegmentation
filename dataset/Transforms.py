import random
import numpy as np
import imgaug as ia
import torch
from imgaug import augmenters as iaa

def augGeo():
    return iaa.SomeOf((0,3),
                [  
                iaa.Crop(percent=(0, 0.2)),  
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, mode="constant", cval=0, order=[0,1]), 
                iaa.Affine(translate_percent={"x": (-0.35, 0.35), "y": (-0.35, 0.35)}, mode="constant", cval=0, order=[0,1]),
                iaa.Affine(rotate=(-90, 90), mode="constant", cval=0, order = [0,1]),
                iaa.Affine(shear=(-25, 25)),
                ], random_order = True)
    
def augInten():
    return iaa.SomeOf((0,3),[
                iaa.Add((-0.2, 0.2)),
                iaa.Multiply((0.75, 1.25), per_channel=0.5),
                iaa.OneOf([
                     iaa.LinearContrast((0.9,1.1),per_channel=0.5),
                     iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                     ]),
                iaa.OneOf([
                    iaa.Dropout((0.0,0.005)),
                    iaa.SaltAndPepper((0.0,0.005)),
                    ]),
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0.0, 2.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 5)),
                    ]),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.25*255), per_channel=0.5),    
                ], random_order = True)


def batchAugmentation(imgBatch, mskBatch, augment):
    """
    imgBatch: [img1, img2, img3, ...]
    mskBatch: [[msk11, msk12,...], [msk21, msk22,...], ...]
    augment: Boolean if augmentation is used.
    """
    ia.seed(random.randint(0, 50))
    # imgBatch = np.asarray(imgBatch) # shape: [nBatch, w, h, d]
    if augment:
        iaaI = augInten()
        iaaG = augGeo().to_deterministic()
        imgBatch = iaaG.augment_images(imgBatch)
        imgBatch = iaaI.augment_images(imgBatch.astype("int16"))
        
        imgBatch = np.expand_dims(imgBatch, axis=-1)
        # chanelBatchs =[]
        # for i in range(len(mskBatch[0])): # 10
        #     m_batch = []
        #     for j in range(len(mskBatch)): # batch=2
        #         m_batch.append(mskBatch[j][i])
        #     m_batch = np.asarray(m_batch)
        #     m_batch = iaaG.augment_images(m_batch)
        #     chanelBatchs.append(m_batch)
        # mskBatch = np.stack(chanelBatchs, axis=-1)
        mskBatch = iaaG.augment_images(mskBatch)

    else:
        imgBatch = np.expand_dims(imgBatch, axis=-1)
        # mskBatch = np.asarray([np.stack(msks,axis=-1) for msks in mskBatch])
        mskBatch = mskBatch
    return imgBatch, mskBatch


