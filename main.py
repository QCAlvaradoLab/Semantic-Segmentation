import os
from model import UNet
from dataset import FishDataset
from test_dataset import TestDataset
from config import Config
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import json
from scipy.misc import imsave
from torch.nn.functional import binary_cross_entropy, cross_entropy, softmax
from torchvision import transforms
import glob
import logging
import sys
from PIL import Image
logging.basicConfig(filename="training.log", filemode='a', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                                          
logger=logging.getLogger(__name__)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def train(config, dataloader, epoch, device, optimizer):
    
    config.net_class.train()

    running_loss = 0 
    for i, (image, seg) in enumerate(dataloader):
        
        image = image.to(device)
        seg[seg>0.5] = 1
        seg[seg<0.5] = 0
        seg = seg.to(device)
        
        optimizer.zero_grad()
        out = config.net_class(image)
        loss = config.loss(out, seg)
        
        running_loss += loss.item()
    
        logger.info('Epoch: %d; Batch: %5d; Loss: %.3f' %
                  (epoch, i + 1, running_loss / (i+1)))
        
        loss.backward()
        optimizer.step()

        imgbatch, segbatch = image.cpu().permute(0,2,3,1).data.numpy(), seg.cpu().data.numpy()
        
        '''
        for idx, img in enumerate(imgbatch):
            imsave('sample_img_%s.jpg'%(str(idx).zfill(2)), (img*255).astype(np.uint8))
        
        for idx, s in enumerate(seg):
            for jdx, a in enumerate(s):
                imsave('sample_ann_%s_%s.jpg'%(str(idx).zfill(2), str(jdx).zfill(2)), (a*255).data.cpu().numpy().astype(np.uint8))
        exit()
        '''
    
def validate(config, loader, epoch, device):
    
    config.net_class.eval()

    running_loss, ctx = 0, 0
    for image, seg in loader:
        
        ctx += image.size(0)
        image = image.to(device)
        seg = seg.to(device)

        out = config.net_class(image)
        loss = config.loss(out, seg)

        running_loss += loss.item()
    
    if out.size(1) == 1:
        ctx = 1

    logger.info('Epoch: %d; Val Loss: %.3f' %
              (epoch, running_loss/float(ctx)))
    
    return running_loss/float(ctx)

def test(config, data, device, img_folder=None, object_threshold=0.8):
    
    config.net_class.eval()
    
    try:
        os.mkdir('test_results')
    except Exception:
        pass

    with torch.no_grad():
        for idx, (img, seg) in enumerate(data):
            
            img = img.to(device)
            out = config.net_class(img)
            
            #labels = torch.argmax(out, dim=1)
            
            #print ('M', torch.max(out), torch.min(out), torch.std(out.float()))
            
            out[out<=object_threshold] = 0
            out[out>object_threshold] = 1
            
            #fg = out.squeeze(0)[1].unsqueeze(0).long()
            #print ('M', torch.max(labels), torch.min(labels))
            #mapping = data.dataset.one_hot_to_segmentation_map(fg, device)

            imsave('test_results/%s_ann.png'%(str(idx).zfill(4)), (out.view(out.size()[2:]).cpu().numpy()*255).astype(np.uint8))
            imsave('test_results/%s_img.png'%(str(idx).zfill(4)), (img.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))


if __name__=='__main__':
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true', help='Flag to test model')
    ap.add_argument('--model', help='Path of stored model')
    ap.add_argument('--img_folder', help='Path of directory of test images to run inference using trained model')
    ap.add_argument('--obj_threshold', help='Optional threshold for segmentation testing', default=0.8, type=float)
    ap.add_argument('--config_file', help='Path to training config file', default='train_config2.json')
    ap.add_argument('--logdir', help='Path to save model weights', default='data/')
    args = ap.parse_args()

    if args.test and args.model is None:
        ap.error("--test requires --model to be specified")
    
    config = Config(args.config_file)

    dataset = FishDataset(config.dataset_folders, img_size=config.img_size, 
                    split='train', dtype=config.ann_settings['type'])

    valdataset = FishDataset(config.dataset_folders, img_size=config.img_size,
                    split='val', dtype=config.ann_settings['type'])

    dataloader = DataLoader(dataset, batch_size=3,
                        shuffle=True, num_workers=8)
    
    valdataloader = DataLoader(valdataset, batch_size=32,
                        shuffle=False, num_workers=8)
    
    if args.img_folder is not None:
        tdataset = TestDataset(args.img_folder)
        testdataloader = DataLoader(tdataset, batch_size=1,
                                    shuffle=False, num_workers=1)
    else:
        testdataloader = DataLoader(valdataset, batch_size=1,
                                    shuffle=False, num_workers=1)

    N = len(dataset.get_classes())
    
    config.set_num_classes(N)

    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0') 
    cpu_device = torch.device('cpu')

    module = False
    if torch.cuda.device_count() > 1 and not args.test:
        logger.info('Using multiple GPUs')
        config.net_class = nn.DataParallel(config.net_class, device_ids=list(range(torch.cuda.device_count()))) 
        module = True
    elif torch.cuda.device_count() == 1:
        logger.info('Using 1 GPU')

    best_loss = 1000
    
    try:
        os.mkdir(args.logdir)
    except Exception:
        pass
    
    try:
        os.mkdir(os.path.join(args.logdir, '%s'%(config.dtype)))
    except Exception:
        pass
    
    config.net_class = config.net_class.to(device)
    
    location = 'cpu' if not torch.cuda.is_available() else None
    if args.test:
        config.load_model(args.model, module=module, location=location)
        logger.info('Loaded model from %s!'%(args.model))
    else:
        if location == 'cpu':
            logger.info('CPU training takes too long anyway! Exiting')
            exit()

        if config.log['load_from'] != "":
            config.load_model(config.log['load_from'], module=module)
            logger.info('Loaded model from %s!'%(config.log['load_from']))

    optimizer = config.get_optimizer(config.net_class.parameters())
    
    if args.test:
        test(config, testdataloader, device, args.img_folder, args.obj_threshold)
        exit()

    for epoch in range(config.log['start_epoch'], config.log['max_epochs']+1): 
        
        train(config, dataloader, epoch, device, optimizer)
        
        with torch.no_grad():
            val_loss = validate(config, valdataloader, epoch, device)
        
        if hasattr(config, 'lr_scheduler'):
            if config.decay_type == 'loss_plateau' and config.optimizer != 'adam':
                prev_lr = config.optim.param_groups[0]['lr']
                config.lr_scheduler.step(val_loss)
                cur_lr = config.optim.param_groups[0]['lr']
                if prev_lr != cur_lr:
                    logger.info('Lowering LR from %.8f to %.8f!'%(prev_lr, cur_lr))

        if epoch % config.log['save_every'] == 0:
            savepath = os.path.join(args.logdir, 
                            '%s/%s_%s_%s.pth'%(config.dtype, config.model, str(epoch).zfill(4), str("%.5f"%(val_loss)).replace('.', '_')))

            logger.info('Saving model to %s'%(savepath))
            config.save_model(savepath, val_loss, epoch, module)
