import numpy as np
import pandas as pd
import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import os
import os.path
import csv
import torchvision
import torch
import pandas as pd
from PIL import Image
class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, path, label_path, resize_ratio, distributed=False, img_indx=None, batch_size=1, num_workers=8, test_dataset = 'kadid', istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers
        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop((384, 384)),
                torchvision.transforms.ToTensor(),
            ])

            self.pretraining_data = PreTrainingDataset(
                root=path, label_path=label_path, index=img_indx, transform=transforms)

            if distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.pretraining_data)
            else:
                self.train_sampler = None
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop((384, 384)),
                torchvision.transforms.ToTensor(),
            ])

            if test_dataset=='kadis':
                self.test_data = PreTrainingDataset(
                    root=path, label_path=label_path, index=img_indx, transform=transforms)


    def get_train_sampler(self):
        return self.train_sampler

    def get_data(self):
        if self.istrain:
            Dataloader = torch.utils.data.DataLoader(
                self.pretraining_data, batch_size=self.batch_size, shuffle=(self.train_sampler is None),
                num_workers=self.num_workers, pin_memory=True, sampler=self.train_sampler)
        else:
            Dataloader = torch.utils.data.DataLoader(
                self.test_data, batch_size=32, num_workers=self.num_workers, shuffle=False, pin_memory=True)
        return Dataloader






class PreTrainingDataset(data.Dataset):

    def __init__(self, root, index, transform, label_path):
        
        df = pd.read_csv(label_path)

      
        all_img_path = [os.path.join(root, filename) for filename in df.iloc[:, 0]]
        all_label = list(df.iloc[:, 1])

        
        self.samples = [(all_img_path[i], all_label[i]) for i in index]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path, label = self.samples[index]
        sample = RGB_Loader(img_path)
        sample = self.transform(sample)

        return sample, label

    def __len__(self):
        length = len(self.samples)
        return length



def RGB_Loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



if __name__ == '__main__':
   
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = PreTrainingDataset(root='/home/user/zed/pretrain-class', label_path='/home/user/zed/pretrain-class/labels.csv', index=range(10), transform=transforms)

    
    for i in range(10):
        img, label = dataset[i]
        print(f'Sample #{i+1}:')
        print(f'Image path: {img}')
        print(f'Label: {label}')



