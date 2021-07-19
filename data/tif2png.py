from PIL import Image
from PIL import ImageFile

import numpy as np
import skimage.io as io
import pylab
import json
import cv2
import os
import tifffile

TIF_DIV = 2^16 - 1

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True

def tif2ping(inDir, outDir):
    TIF_DIV = 2^16 - 1
    SAR_inDir = os.path.join(inDir, 'A')
    color_inDir = os.path.join(inDir, 'B')

    SAR_outDir = os.path.join(outDir, 'A')
    color_outDir = os.path.join(outDir, 'B')

    create_exp_dir(outDir)
    create_exp_dir(SAR_outDir)
    create_exp_dir(color_outDir)

    file_list = sorted(os.listdir(SAR_inDir))
    #color_files = sorted(os.listdir(color_inDir))
    print(file_list)
    #print(color_files)
    """
    img_names = []
    for file in SAR_files:
        if file.endswith(".tif"):
            img_names.append(file)
    print(img_names)
    """
    for img_name in file_list:
        #("["+i+"] "+ img_Dir)
        SARimg_Dir= '{}/{}'.format(SAR_inDir, img_name)
        colorimg_Dir= '{}/{}'.format(color_inDir, img_name)
        
        #SARimg = tifffile.imread(SARimg_Dir)
        #SARimg = SARimg.astype(np.float32)
        #SARimg = np.array(SARimg) / TIF_DIV

        SARimg = Image.open(SARimg_Dir)
        SARimg = np.array(SARimg)
        SAR_image=Image.fromarray(SARimg)
        SAR_image_resized = SAR_image.resize((256, 256))
        #SAR_image_resized.show()
        SARout_Dir = '{}/{}.png'.format(SAR_outDir, img_name[:-4])
        SAR_image_resized.save(SARout_Dir)

        colorimg = Image.open(colorimg_Dir)
        colorimg = np.array(colorimg)
        color_image=Image.fromarray(colorimg)
        color_image_resized = color_image.resize((256, 256))
        #SAR_image_resized.show()
        colorout_Dir = '{}/{}.png'.format(color_outDir, img_name[:-4])
        color_image_resized.save(colorout_Dir)

        # SAR_image.show()
        # print(SARimg.shape) (512, 512, 3)



    

        
        #print(out_Dir)
    #label = Image.open()


if __name__ == '__main__':
    # pass
    inDir = 'D:/Dataset/KOMPSAT5/TRAIN-small'
    outDir = 'D:/Dataset/KOMPSAT5/TRAIN-small-png'
    # C:\Users\kaist\Desktop\Dataset\COCO\testimg2017

    tif2ping(inDir, outDir)

    


    #img_names = crop_384(imgDir, outDir)
