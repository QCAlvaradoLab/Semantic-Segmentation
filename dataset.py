import rawpy
import imageio

import glob
import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import time
random.seed(int(time.time()))
import cv2

import local_transforms
'''
HIERARCHY:
Ventral side -> pectoral anal fin
Dorsal side -> dorsal fin
Head -> Eye + Operculum

INDEPENDENT:
Whole body

Humeral blotch
Pelvic fin
Caudal Fin
'''

INIT = ['whole_body']
HPARTS = [['ventral_side', 'anal_fin', 'pectoral_fin'], ['dorsal_side', 'dorsal_fin'], ['head', 'eye', 'operculum']]
INDEP = ['humeral_blotch', 'pelvic_fin', 'caudal_fin']

IMG_TYPES = ['jpg', 'png', 'arw']
IMG_TYPES.extend([x.upper() for x in IMG_TYPES])

JOINT_TRANSFORMS = ['CenterCrop', 'FiveCrop', 'Pad', 'RandomAffine', 'RandomCrop', 'RandomHorizontalFlip', 
                    'RandomVerticalFlip', 'RandomResizedCrop', 'RandomRotation', 'Resize', 'TenCrop']

IMG_TRANSFORMS = ['ColorJitter', 'Grayscale', 'RandomGrayscale']
IMG_FTRANSFORMS = ['adjust_gamma']

class FishDataset(Dataset):
    
    # folders: List of dataset folders with subfolders for body parts with naming convention as above.
    # dtype:
    #       all: Return all possible classes together for a single forward op
    #       full: whole_body semantic segmentation
    #       indep: independent parts finetuned using whole_body model
    #       ventral_side: assumes model for ventral_side already exists
    #       dorsal_side: assumes model for dorsal_side already exists
    #       head: assumes segmentation model for head already exists
    #       body: ventral_side, dorsal_side and head segmentations
    # split: train, test (test can be used for val)
    # shuffle: randomly shuffle dataset
    # transform: PIL Image transforms for image and annotation augmentation (rotate, flip etc.) [only augmentation transforms]
    # target_transform: PIL Image transforms for image augmentation only (brightness, contrast etc.) [only augmentation transforms]
    # name: Name identifying dataset to store cached RGB stats
    def __init__(self, folders, split='train', dtype='full', img_size=(1024,1024), transform=None, target_transform=None, name='fish'):
        
        self.name = name
        self.transform = transform
        self.target_transform = target_transform
        self.split_ratio = 0.9 # train-val split
        
        if dtype=='all':
            self.dataset = INIT + INDEP + [y for x in HPARTS for y in x]
        if dtype=='full':
            self.dataset = INIT
        elif dtype=='indep':
            self.dataset = INDEP
        elif dtype=='body':
            self.dataset = [x[0] for x in HPARTS]
        else:
            for i in range(len(HPARTS)):
                if dtype==HPARTS[i][0]:
                    self.dataset = HPARTS[i][1:]
        
        self.dtype = dtype

        self.img_size = img_size
        self.split = split

        # Dataset input images (x)
        self.img_files = {}
        # Dataset annotation images (y)
        self.ann_files = []
        for folder in folders:
            
            imgs = self.get_image_files(folder)
            self.img_files.update(imgs)
            
            for idx, fl in enumerate(self.dataset):
                dpath = os.path.join(folder, fl)
                anns = self.get_ann_files(dpath, self.img_files)

                if idx < len(self.ann_files):
                    self.ann_files[idx].update(anns)
                else:
                    self.ann_files.append(anns)
        
        N = len(list(self.img_files.keys()))
        n = int(N * self.split_ratio)
        
        num_del_imgs = N-n if split=='train' else n
        for _ in range(num_del_imgs):
            k = random.choice(list(self.img_files.keys()))
            del self.img_files[k]

            for ann in self.ann_files:
                if k in ann:
                    del ann[k]
       
        self.ordered_keys = list(self.img_files.keys())
        
        self.mean, self.std = self.calculate_rough_image_stats()

    def get_image_files(self, path):
        
        imgs = [y for x in [glob.glob(os.path.join(path, '*.'+e)) 
                  for e in IMG_TYPES] for y in x]

        img_dict = {}
        for path in imgs:
            sfx = '.'.join(path.split('/')[-1].split('.')[:-1])
            img_dict[sfx] = path

        return img_dict
    
    def calculate_rough_image_stats(self):
           
        if os.path.exists('cache/stats.txt'):   
            with open('cache/stats.txt', 'r') as f:
                mean, std = [np.array([np.float(y) for y in x.strip().split(' ')]) 
                                for x in f.readlines()]
            return mean, std

        mean = np.zeros(3)
        std = np.zeros(3)
        for key in self.img_files:
            img = np.array(self.get_image(self.img_files[key]))
            mean += np.mean(np.mean(img, axis=0), axis=0)
            std += np.std(np.std(img, axis=0), axis=0)
            
        nb_samples = len(self.img_files)
        mean /= nb_samples
        std /= nb_samples

        if not os.path.isdir('cache'):
            os.mkdir('cache')

        with open('cache/stats.txt', 'w') as f:
            f.write(' '.join(['%.5f'%(x) for x in mean])+'\n')
            f.write(' '.join(['%.5f'%(x) for x in std]))
 
        return mean, std

    def get_segmentation_mask(self, path):
        
        ANN = self.get_image(path)
        gray = ANN.convert('L')
        bw = gray.point(lambda x: 0 if x==255 else 1, '1')
        
        #bw.save('sample.jpg')
        #print (np.max(np.array(bw))) 
        #print (np.min(np.array(bw))) 
        
        return bw
    
    def one_hot_to_segmentation_map(self, ann, device=None):

        fflag = False
        seg = torch.zeros(ann.size()[-2:]).long()
        
        if device:
            seg = seg.to(device)

        for idx, a in enumerate(ann):
            seg += a*(idx+1)
            if torch.max(seg) > idx+1 and not fflag:
                print ('Warning: segmentation map overlap detected!')
                fflag = True
        return seg

    def get_ann_files(self, path, imgs):
        
        ann_dict = {}
        for img_key in imgs:
            
            annfile = glob.glob(os.path.join(path, '*'+img_key+' *'))
            annfile.extend(glob.glob(os.path.join(path, '*'+img_key+'.*')))
            assert len(annfile) <= 1
            if annfile:
                ann_dict[img_key] = annfile[0]
                
                #img = self.get_image(imgs[img_key])
                #img.save('sample2.jpg')
                
        return ann_dict
    
    def __len__(self):
        return len(self.img_files)

    def get_image(self, path):
        
        if 'arw' in path.lower():
            raw = rawpy.imread(path)
            img = raw.postprocess()
            img = Image.fromarray(img)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(img)

        return img

    def __getitem__(self, index):
        
        #key = random.choice(self.ordered_keys)
        key = self.ordered_keys[index]
        image = self.get_image(self.img_files[key])
        
        # List of PIL Image objects
        anns = []
        for idx, ann in enumerate(self.ann_files):
            segmask = self.get_segmentation_mask(ann[key]) 
            anns.append(segmask)
        
        random.seed(int(time.time()))
        if self.transform is not None and np.random.rand() > 0.3:
            idx, rn = random.choice(list(enumerate(self.transform)))
            transform = local_transforms.EnhancedCompose([
                                            local_transforms.Split(),
                                            rn,
                                            local_transforms.Merge()])
            grp = [image] + anns
            trans = transform(grp)
            image = trans[0]
            anns = trans[1:]

        if self.target_transform is not None and np.random.rand() > 0.3:
            idx, rn = random.choice(list(enumerate(self.transform)))
            image = rn(image)
        
        resize_transform = transforms.Resize(self.img_size)
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)])
        image = resize_transform(image)

        #image.save('sample_image_%s.jpg'%(str(index).zfill(2)))
        #[segmask.save('sample_ann_%s_%d.jpg'%(str(index).zfill(2), idx)) for idx, segmask in enumerate(anns)]
        
        image = transform(image)

        random.seed(int(time.time()))
        if self.target_transform:
            for idx, ann in enumerate(anns):
                anns[idx] = self.target_transform(ann)

        annvec = None
        for idx, ann in enumerate(anns):
            #ann.save('sample_ann_%s_%s.jpg'%(str(index).zfill(2), str(idx).zfill(2)))

            if len(self.dataset) == 1:
                ann_slice = transforms.ToTensor()(resize_transform(ann)).float()
            else:
                ann_slice = transforms.ToTensor()(resize_transform(ann)).long()
            ann_slice = ann_slice.squeeze(1)
            if annvec is None:
                annvec = ann_slice
            else:
                annvec = torch.cat((annvec, ann_slice))
        
        if len(anns) == 1:
            return image, annvec
        else:
            return image, self.one_hot_to_segmentation_map(annvec)
    
    def get_classes(self):
        return self.dataset
        
    def apply_transforms(self, img, seg):
       pass

if __name__=='__main__':
    
    types = ['full', 'body', 'indep', 'ventral_side', 'dorsal_side', 'head']
    
    types = types[0]

    for t in types:
        f = FishDataset(['/home/hans/Haplochromis-Burtoni-Study/Machine learning training set/Light-Dark/T0', 
                         '/home/hans/Haplochromis-Burtoni-Study/Machine learning training set/Light-Dark/T1',
                         '/home/hans/Haplochromis-Burtoni-Study/Machine learning training set/photos 1.30.2019'], 
                         split='test', dtype=t)
        
        for idx in range(len(f)):
            f.__getitem__(idx)
        
    '''
    FishDataset('x', 'body')
    FishDataset('x', 'indep')
    FishDataset('x', 'ventral_side')
    FishDataset('x', 'dorsal_side')
    FishDataset('x', 'head')
    '''
