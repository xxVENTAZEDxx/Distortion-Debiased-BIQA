import os
import os.path
import random
from PIL import Image

import csv
import scipy.io
import numpy as np

import torch
import torchvision
import torch.utils.data as data
import pandas as pd

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, config, dataset, path, patch_num, img_indx=None, istrain=True):
        # config.dataset = dataset
        self.train_bs = config.train_bs
        self.eval_bs = config.eval_bs
        self.istrain = istrain
        self.num_workers = config.num_workers
        if istrain:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.RandomVerticalFlip(), 
                torchvision.transforms.RandomCrop((384, 384)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #                                  std=[0.5, 0.5, 0.5]) 
            ])
        else:
            transforms = torchvision.transforms.Compose([

                torchvision.transforms.RandomCrop((384, 384)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #                                  std=[0.5, 0.5, 0.5])
            ])



        if dataset == 'live':
            self.data = LIVEFolder(
            root=path, index=img_indx, transforms=transforms, patch_num=patch_num, img_crop=True)
        elif dataset=='csiq':
            self.data = CSIQFolder(
            root=path, index=img_indx, transforms=transforms, patch_num=patch_num, img_crop=True)
        elif dataset == 'tid2013':
            self.data = TID2013Folder(
            root=path, index=img_indx, transforms=transforms, patch_num=patch_num, img_crop=False)
        elif dataset == 'kadid-10k':
            self.data = KADID10KFolder(
            root=path, index=img_indx, transforms=transforms, patch_num=patch_num, img_crop=False)

        elif dataset == 'livec':
            self.data = LIVEChallengeFolder(
            root=path, index=img_indx, transforms=transforms, patch_num=patch_num, img_crop=True)
        elif dataset == 'koniq-10k':
            self.data = Koniq_10kFolder(
            root=path, index=img_indx, transforms=transforms, patch_num=patch_num, img_crop=True)
        elif dataset == 'spaq':
            self.data = SPAQ(root=path, index=img_indx, transforms=transforms, patch_num=patch_num, img_crop=False)
        else:
            print('Invalid dataset were provided.')

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.train_bs, shuffle=True, num_workers=self.num_workers,pin_memory=False)
        else:
            dataloader = torch.utils.data.DataLoader( self.data, batch_size=self.eval_bs,num_workers=self.num_workers, shuffle=False,pin_memory=False)
        return dataloader



class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transforms, patch_num, img_crop):
        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)
        
        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']
        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item], refpath + '/' + refname[index[i]])) 

        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target, ref_path = self.samples[index]  
        img = imgread(path)  
        img = imgprocess(img, patch_size=[384, 384])  
        sample = self.transforms(img)  

        ref_img = imgread(ref_path)  
        ref_img = imgprocess(ref_img, patch_size=[384, 384])  
        ref_sample = self.transforms(ref_img)  

        return sample, ref_sample, target  

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename



class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transforms, patch_num, img_crop):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)
        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # Create a folder 'all_imgs' and copy all distorted images into it.
                    sample.append((os.path.join(root, 'all_imgs', imgnames[item]), labels[item], os.path.join(refpath, refname[index[i]])))  # 添加参考图像的路径
        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target, ref_path = self.samples[index]  # 获取参考图像的路径
        img = imgread(path)  # 直接读取图像，不进行颜色空间转换
        img = imgprocess(img, patch_size=[384, 384])  # 对图像进行必要的预处理
        sample = self.transforms(img)  # 对图像应用转换

        ref_img = imgread(ref_path)  # 读取参考图像
        ref_img = imgprocess(ref_img, patch_size=[384, 384])  # 对参考图像进行必要的预处理
        ref_sample = self.transforms(ref_img)  # 对参考图像应用转换

        return sample, ref_sample, target  # 返回样本、目标和参考样本

    def __len__(self):
        length = len(self.samples)
        return length

class TID2013Folder(data.Dataset):
    def __init__(self, root, index, transforms, patch_num, img_crop):
        refpath = os.path.join(root, 'reference_images')
        refname = sorted(getTIDFileName(refpath, '.bmp.BMP'))
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)
        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item], os.path.join(refpath, 'I' + refname[index[i]] + '.BMP')))  # 添加参考图像的路径
        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target, ref_path = self.samples[index]  # 获取参考图像的路径
        img = imgread(path)  # 直接读取图像，不进行颜色空间转换
        img = imgprocess(img, patch_size=[384, 384])  # 对图像进行必要的预处理
        sample = self.transforms(img)  # 对图像应用转换

        ref_img = imgread(ref_path)  # 读取参考图像
        ref_img = imgprocess(ref_img, patch_size=[384, 384])  # 对参考图像进行必要的预处理
        ref_sample = self.transforms(ref_img)  # 对参考图像应用转换

        return sample, ref_sample, target  # 返回样本、目标和参考样本

    def __len__(self):
        length = len(self.samples)
        return length

class KADID10KFolder(data.Dataset):
    def __init__(self, root, index, transforms, patch_num, img_crop):
        imgname = []
        refnames_all = []
        labels = []
        csv_file = os.path.join(root, 'dmos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['dist_img'])
                refnames_all.append(row['ref_img'])
                mos = np.array(float(row['dmos']))
                labels.append(mos)
        im_ref = np.unique(refnames_all)
        refnames_all = np.array(refnames_all)
        sample = []
        for i, item in enumerate(index):
            train_sel = (im_ref[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'images', imgname[item]),
                                   os.path.join(root, 'ref_imgs', refnames_all[item]), labels[item]))

        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, refpath, target = self.samples[index]  
        img = imgread(path)  
        img = imgprocess(img, patch_size=[384, 384])  
        sample = self.transforms(img)  

        refimg = imgread(refpath)  
        refimg = imgprocess(refimg, patch_size=[384, 384])  
        refsample = self.transforms(refimg)  

        return sample, refsample, target  


    def __len__(self):
        length = len(self.samples)
        return length

class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transforms, patch_num, img_crop):
        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:]

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))
        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = imgread(path)  # 直接读取图像，不进行颜色空间转换
        img = imgprocess(img, patch_size=[384, 384])  # 对图像进行必要的预处理
        sample = self.transforms(img)  # 对图像应用转换
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class Koniq_10kFolder(data.Dataset):
    def __init__(self, root, index, transforms, patch_num, img_crop):
        imgname = []
        labels = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                labels.append(mos)
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '512x384', imgname[item]), labels[item]))

        self.img_crop = img_crop
        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = imgread(path)  # 直接读取图像，不进行颜色空间转换
        img = imgprocess(img, patch_size=[384, 384])  # 对图像进行必要的预处理
        sample = self.transforms(img)  # 对图像应用转换
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class SPAQ(data.Dataset):
    def __init__(self, root, index, transforms, patch_num, img_crop):
        spaq_info = pd.read_excel(
            os.path.join(root, 'Annotations/MOS and Image attribute scores.xlsx'),engine='openpyxl')
        imgnames = np.asarray(spaq_info['Image name'])
        labels = np.asarray(spaq_info['MOS']).astype(np.float32)
        sample = []

        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'TestImage_resize', imgnames[item]), labels[item]))

        self.samples = sample
        self.transforms = transforms

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = imgread(path)  # 直接读取图像，不进行颜色空间转换
        img = self.resize_image(img)
        img = imgprocess(img, patch_size=[384, 384])  # 对图像进行必要的预处理
        sample = self.transforms(img)

        # path, target = self.samples[index]
        # sample = pil_loader(path)
        # sample = self.resize_image(sample)
        # sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def resize_image(self, img, min_length=384):
        width, height = img.size

        if width < height:
            new_width = min_length
            new_height = int(height * (min_length / width))
        else:
            new_height = min_length
            new_width = int(width * (min_length / height))

        # resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_img


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename

def imgread(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def imgprocess(img, patch_size=[384, 384]):

    w, h = img.size
    w_ = np.random.randint(low=0, high=w - patch_size[1] + 1)
    h_ = np.random.randint(low=0, high=h - patch_size[0] + 1)
    img = img.crop((w_, h_, w_ + patch_size[1], h_ + patch_size[0]))

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


if __name__ == '__main__':
    pass



