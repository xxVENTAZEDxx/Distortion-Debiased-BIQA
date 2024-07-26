import torch

import torch.optim as optim
from tensorboardX import SummaryWriter

from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

from backbone import IQAModel
from scipy import stats


import time
import os
import argparse
import random


import data_loader


random.seed(12)

# 创建一个包含0到5999999的列表
img_sel = list(range(0, 6250000))
random.shuffle(img_sel)
val_img_num = 50000

train_index = img_sel[:-val_img_num]
val_index = img_sel[-val_img_num:]


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_data(args): 

    train_loader = data_loader.DataLoader(args.pretrain_path, args.ref_path, args.label_path, args.resize_ratio, img_indx=train_index,
                                          batch_size=args.batch_size, num_workers=args.num_workers, istrain=True)



    val_loader = data_loader.DataLoader(args.pretrain_path, args.ref_path, args.label_path, args.resize_ratio, img_indx=val_index,
                                        batch_size=args.batch_size, test_dataset='kadis', istrain=False)
    train_loader = train_loader.get_data()
    # test_kadid = test_loader.get_data()
    val_kadis = val_loader.get_data()

    return train_loader,val_kadis


writer = SummaryWriter('./log') 


def save_checkpoint(plcc, srocc, model, optimizer, args, epoch):
    print('Best Model Saving...')

    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'srocc': srocc,
        'plcc': plcc,
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))  # 模型保存路径为'checkpoints/checkpoint_model_best.pth'


best_srocc = 0.
best_plcc = 0.
global_step = 0
def _train(epoch, train_loader, val_loader, model, res_model, optimizer, criterion, lr_scheduler, args, crop_size = 224,):
    global best_srocc, best_plcc,global_step
    start = time.time()
    model.train()
    losses = 0.
    for idx, (img, target, label) in enumerate(train_loader):
        # if args.cuda:
        img, target , label= img.cuda(), target.cuda(), label.cuda()
        img_res = res_model(img)[0].cuda()

        image_height, image_width = img.shape[-2], img.shape[-1]

        crop_top = torch.randint(0, image_height - crop_size + 1, (1,))
        crop_left = torch.randint(0, image_width - crop_size + 1, (1,))

        img = img[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
        img_res = img_res[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
        target = target[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]


        iters_per_epoch = len(train_loader)
        optimizer.zero_grad()
        output, error_loss = model(img,img_res,target)


        loss1 = criterion(output,label)# 计算修复后的图像和原始图像之间的损失
        loss2 = error_loss

        loss = loss1 + loss2

        global_step += 1
        losses += loss.item()
        loss.backward()

        # if args.gradient_clip > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        lr_scheduler.step()
        writer.add_scalar('Training Loss',loss.item(), epoch * iters_per_epoch + idx)
        if (idx+1) % args.print_intervals == 0:
            print('[Epoch: {0:4d}], Loss: {1:.3f}'.format(epoch, losses / (idx + 1)))
        if (idx+1) % 5000 == 0:
            plcc, srocc = _eval(epoch, val_loader, model, res_model, args)
            print("Test PLCC: {}, SROCC: {}".format(plcc, srocc))
            if plcc > best_plcc and srocc > best_srocc:
                best_plcc = plcc
                best_srocc = srocc
                save_checkpoint(plcc, srocc, model, optimizer, args, epoch)
    end = time.time()
    print('Time:', end-start, 'Global_step:', global_step)
    return best_plcc,best_srocc

def _eval(epoch, test_loader, model, res_model, args, crop_size = 224):
    model.eval()
    total_plcc = 0.
    total_srocc = 0.
    with torch.no_grad():
        for idx, (img,_, label) in enumerate(test_loader):
            img, label = img.cuda(), label.cuda()
            img_res = res_model(img)[0].cuda()

            image_height, image_width = img.shape[-2], img.shape[-1]

            crop_top = torch.randint(0, image_height - crop_size + 1, (1,))
            crop_left = torch.randint(0, image_width - crop_size + 1, (1,))

            img = img[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
            img_res = img_res[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
            # target = target[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
            
            output= model(img,img_res)

            output = output.cpu().tolist()
            label = label.cpu().tolist()

            # 计算PSNR和SSIM
            plcc,_ = stats.pearsonr(output, label)
            srocc,_ = stats.spearmanr(output, label)
            print(plcc)
            total_plcc += plcc
            total_srocc += srocc


        writer.add_scalar('Testing PLCC', total_plcc / len(test_loader.dataset), epoch)
        writer.add_scalar('Testing SROCC', total_srocc / len(test_loader.dataset), epoch)

    return total_plcc / len(test_loader.dataset), total_srocc / len(test_loader.dataset)


def main(args):
    
    train_loader, val_loader= load_data(args)

    # 初始化模型
    model = IQAModel()
    model = model.cuda()

    res_model = SwinTransformerSys(img_size=(384,384), window_size=12)
    res_model = res_model.cuda()
    checkpoint = torch.load('checkpoint_model_best.pth')
    res_model.load_state_dict(checkpoint['model_state_dict'])
    res_model.eval()

    criterion = torch.nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', args.checkpoints))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch']
    else:
        start_epoch = 1

    # if args.cuda:
    model = model.cuda()
    model.train(True)
    if not args.evaluation:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.000001)  # 定义余弦退火重启学习率调度器


        for epoch in range(start_epoch, args.epochs + 1):
            plcc, srocc = _train(epoch, train_loader, val_loader, model, res_model, optimizer, criterion, lr_scheduler, args)


            # lr_scheduler.step()
            lr_ = lr_scheduler.get_last_lr()[0]
            print('Current Learning Rate: {}'.format(lr_))
            writer.add_scalar('Learning Rate', lr_, epoch)
        writer.close()

    else:
          _eval(epoch, val_loader, model, res_model, args)



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

dataset_path ={
        'pretrain_path': '/home/user/zed/pretrained_data',
        'test_path':'D:/HAT-main/datasets/kadid10k',
    'ref_path': '/home/user/zed/50k',
    'label_path': '/home/user/zed/labels'
}

def load_config():
    parser = argparse.ArgumentParser()  # 创建一个解析器对象
    parser.add_argument('--pretrain_path', type=str, default=dataset_path['pretrain_path'])  # 添加预训练数据集路径参数，默认为字典中的值
    parser.add_argument('--test_path', type=str, default=dataset_path['test_path'])  # 添加测试数据集路径参数，默认为字典中的值
    parser.add_argument('--ref_path', type=str, default=dataset_path['ref_path'])
    parser.add_argument('--label_path', type=str, default=dataset_path['label_path'])
    parser.add_argument('--batch_size', type=int, default=72)  # 添加批量大小参数，默认为64
    parser.add_argument('--resize_ratio', type=float, default=0.75)  # 添加图像缩放比例参数，默认为0.75
    parser.add_argument('--num_workers', type=int, default=8)  # 添加工作线程数参数，默认为8
    parser.add_argument('--lr', type=float, default=1e-4)  # 添加学习率参数，默认为0.05
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 添加权重衰减参数，默认为1e-4
    parser.add_argument('--momentum', type=float, default=0.9)  # 添加动量参数，默认为0.9
    parser.add_argument('--cuda', type=bool, default=True)  # 添加是否使用CUDA参数，默认为True
    parser.add_argument('--epochs', type=int, default=5)  # 添加训练轮数参数，默认为50
    parser.add_argument('--print_intervals', type=int, default=5000)  # 添加打印间隔参数，默认为1000
    parser.add_argument('--evaluation', type=bool, default=False)  # 添加是否评估模式参数，默认为False
    parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')  # 添加模型检查点路径参数，默认为None，带有帮助信息
    parser.add_argument('--device_num', type=int, default=1)  # 添加设备数参数，默认为1
    parser.add_argument('--gradient_clip', type=float, default=2.)  # 添加梯度裁剪参数，默认为2.
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for Adam optimizer')

    return parser.parse_args()

if __name__ == '__main__':
    args = load_config()
    main(args)