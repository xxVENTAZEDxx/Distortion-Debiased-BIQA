import numpy as np
import pandas as pd
import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import os
import os.path
import csv
class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, path,ref_path,label_path,resize_ratio, distributed=False, img_indx=None,batch_size=1,num_workers=8, test_dataset = 'kadid',istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers
        # Train transforms
        if istrain:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.Resize((384, 512)),
                torchvision.transforms.CenterCrop((384, 384)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #                                  std=[0.5, 0.5, 0.5])
            ])

            self.pretraining_data = PreTrainingDataset(
                root=path, ref_path=ref_path, label_path=label_path, index=img_indx, transform=transforms)

            if distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.pretraining_data)
            else:
                self.train_sampler = None
        # Test transforms
        else:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((384, 512)),
                torchvision.transforms.CenterCrop((384, 384)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #                                  std=[0.5, 0.5, 0.5])
            ])

            if test_dataset=='kadis':
                self.test_data = PreTrainingDataset(
                    root=path, ref_path=ref_path, label_path=label_path, index=img_indx, transform=transforms)
            if test_dataset == 'kadid':
                self.test_data = TestKADIDDataset(
                    root=path, index=img_indx, transform=transforms)

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

    def __init__(self, root, index, transform, ref_path, label_path):
        self.transform = transform
        self.samples = []

        # Load all CSV files from label_path
        for file in os.listdir(label_path):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(label_path, file))
                for _, row in df.iterrows():
                    img_path = os.path.join(root, row[0])
                    label = row[1]
                    # Extract the base name without distortion type and level, and add .png extension
                    base_name = '_'.join(os.path.basename(img_path).split('_')[:-2]) + '.png'
                    # print(base_name)
                    target_path = os.path.join(ref_path, base_name)
                    # print(target_path)
                    self.samples.append((img_path, target_path, label))

    def __getitem__(self, index):
        img_path, target_path, label = self.samples[index]
        sample = RGB_Loader(img_path)
        sample = self.transform(sample)
        target = RGB_Loader(target_path)
        target = self.transform(target)

        return sample, target, label

    def __len__(self):
        return len(self.samples)

class TestKADIDDataset(data.Dataset):

    def __init__(self, root, index, transform):
        refpath = os.path.join(root, 'ref_imgs')
        # refname = getTIDFileName(refpath, '.png.PNG')
        # txtpath = os.path.join(root, 'dmos.txt')
        # fh = open(txtpath, 'r')

        imgnames = []
        target = []
        refnames_all = []
        all_dist_label = []
        csv_file = os.path.join(root, 'dmos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row['dist_img']
                imgnames.append(img)

                label = (int(img[-9:-7]) - 1) * 5 + int(img[-6:-4])
                all_dist_label.append(int(label - 1))
                refnames_all.append(row['ref_img'])
                # mos = np.array(float(row['dmos'])).astype(np.float32)
                # target.append(mos)
        # labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)
        # print(all_dist_label)
        refname= np.unique(refnames_all)
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):

                sample.append((os.path.join(root, 'distorted_imgs', imgnames[item]), os.path.join(refpath, refnames_all[item]), all_dist_label[item]))
        self.samples = sample
        # print(sample[:10], len(sample))
        self.transform = transform
        # all_refimg = []
        # all_img = []
        # all_label = []
        # dist_imgs = pd.read_csv(os.path.join(root, 'dmos.csv'))
        # dist_imgs = dist_imgs['dist_img']
        # ref_imgs = dist_imgs['ref_img']
        # for i, img in enumerate(dist_imgs):
        #     label = (int(img[-9:-7])-1) * 5 + int(img[-6:-4])
        #     all_label.append(int(label - 1))
        #     all_img.append(img)
        #     all_refimg.append(ref_imgs[i])
        #
        # sample = []
        # for i, item in enumerate(index):
        #         sample.append((os.path.join(root, 'images', all_img[item]), os.path.join(root,all_refimg[]), all_label[item]))
        #
        # self.samples = sample # 10125
        # self.transform = transform
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path, target_path, label = self.samples[index]
        sample = RGB_Loader(img_path)
        sample = self.transform(sample)
        target = RGB_Loader(target_path)
        target = self.transform(target)
        return sample, target, label

    def __len__(self):
        return len(self.samples)



def RGB_Loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

import torchvision.transforms as transforms

if __name__ == '__main__':
    root = '/home/user/zed/pretrained_data'  
    index = [0, 1, 2, 3, 4]  
    transform = transforms.Compose([transforms.ToTensor()])  
    ref_path = '/home/user/zed/50k'  
    label_path = '/home/user/zed/labels'  

    dataset = PreTrainingDataset(root, index, transform, ref_path, label_path)

    for i in range(min(3, len(dataset))):
        sample, target, label = dataset[i]
        print(f'Sample: {sample}, Target: {target}, Label: {label}')




