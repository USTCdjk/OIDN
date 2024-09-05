import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import tifffile
import random
from Comparison.util.util import prctile_norm

class Dataset(Dataset):
    def __init__(self, datapath, datalist, mode='train', norm=True, input_num=9):
        self.datapath = datapath
        self.datalist = datalist
        self.mode=mode
        self.norm=norm
        self.input_num=input_num
        self.in_path,self.gt_path = self._read_datalist()

    def __len__(self):
        return len(self.in_path)

    def __getitem__(self, idx):
        in_path = self.in_path[idx]
        gt_path = self.gt_path[idx]
        if self.mode=='train':
           input_images=[]
           norm_inputs=[]
           for i in range(self.input_num):
               input_image = tifffile.imread(in_path+ '/'+str(i + 1) + '.tif')  #modified according to real data path
               input_image = np.array(input_image, dtype=np.float32)
               input_image = input_image/65535.
               if self.norm:
                  norm_input = prctile_norm(input_image)
               input_images.append(input_image)
               norm_inputs.append(norm_input)
           input_images = np.stack(input_images, 0)  
           norm_inputs = np.stack(norm_inputs, 0)    
           #print(np.shape(input_images))
        elif self.mode=='eval':
           input_images=[]
           norm_inputs=[]
           with tifffile.TiffFile(in_path) as tif:
                num_layers = len(tif.pages)
                for i in range(self.input_num):
                    input_image = tif.pages[i].asarray()
                    input_image = np.array(input_image, dtype=np.float32)
                    input_image = input_image/65535.
                    if self.norm:
                       norm_input = prctile_norm(input_image)
                    input_images.append(input_image)
                    norm_inputs.append(norm_input)
                input_images = np.stack(input_images, 0) 
                norm_inputs = np.stack(norm_inputs, 0)   

        elif self.mode=='test':
           input_images=[]
           norm_inputs=[]
           with tifffile.TiffFile(in_path) as tif:
                num_layers = len(tif.pages)
                for i in range(self.input_num):
                    input_image = tif.pages[i].asarray()
                    input_image = np.array(input_image, dtype=np.float32)
                    input_image = input_image/65535.
                    if self.norm:
                       norm_input = prctile_norm(input_image)
                    input_images.append(input_image)
                    norm_inputs.append(norm_input)
                input_images = np.stack(input_images, 0) 
                norm_inputs = np.stack(norm_inputs, 0) 
        else:
           print("Error! seleting the proper mode: train or test.")
           exit()
        
        gt_images=tifffile.imread(gt_path)
        gt_images = np.array(gt_images, dtype=np.float32)
        gt_images = gt_images/65535.
        if self.norm:
           norm_gt = prctile_norm(gt_images)
        gt_images= np.expand_dims(gt_images, axis=0)
        norm_gt= np.expand_dims(norm_gt, axis=0)

        return input_images, gt_images, norm_inputs, norm_gt

    
    def _read_datalist(self):
        
        f=open(self.datalist,'r')
        in_path=[]
        gt_path=[]
        for line in f:
            try:
                in_img, gt_img=line.strip("\n").split(' ')
            except ValueError:
                in_img = gt_img = line.strip("\n")
            in_path.append(self.datapath+in_img)
            gt_path.append(self.datapath+gt_img)
        return in_path, gt_path  
        
    def random_crop(self, input_image,patch_size=512): 
        _,width, height = input_image.shape
        width_start= random.randint(5, width - patch_size - 5)
        width_end=width_start+patch_size
        height_start= random.randint(5, height - patch_size - 5)
        height_end=height_start+patch_size
        croped_image=input_image[:, width_start:width_end,height_start:height_end]
        return croped_image

