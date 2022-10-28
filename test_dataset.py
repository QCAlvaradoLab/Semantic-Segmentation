from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision import transforms
import torch
import cv2

TEST_MEAN = [174.11026, 166.63412, 150.50042]  
TEST_STD = [12.89034, 10.36813, 10.12681]

class TestDataset(Dataset):
    
    def __init__(self, img_dir):
        
        self.img_paths = glob.glob(os.path.join(img_dir, '*'))
    
        self.transform = transforms.Compose([transforms.Resize((512,512)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(TEST_MEAN, TEST_STD)])
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        path = self.img_paths[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(image)
        img = self.transform(img)

        return img, torch.zeros(1)

if __name__=='__main__':
    
    d = TestDataset('TEST/')
    print (d.__getitem__(0).size())
