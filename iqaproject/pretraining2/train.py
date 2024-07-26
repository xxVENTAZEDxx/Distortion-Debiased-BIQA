import torch
import torch.nn as nn

import torch.optim as optim
from tensorboardX import SummaryWriter


from classmodel import ClassModel


import time
import os
import argparse
import random


import data_loader
import time
import random


random.seed(2024)


img_sel = list(range(0, 250000))
random.shuffle(img_sel)
random.shuffle(img_sel)
val_img_num = 50000

train_index = img_sel[:-val_img_num]
val_index = img_sel[-val_img_num:]





def load_data(args):
    train_loader = data_loader.DataLoader(args.pretrain_path, args.label_path, args.resize_ratio, img_indx=train_index,
                                          batch_size=args.batch_size, num_workers=args.num_workers, istrain=True)

    val_loader = data_loader.DataLoader(args.pretrain_path, args.label_path, args.resize_ratio, img_indx=val_index,
                                        batch_size=args.batch_size, test_dataset='kadis', istrain=False)

    train_loader = train_loader.get_data()
    # test_kadid = test_loader.get_data()
    val_kadis = val_loader.get_data()

    return train_loader,val_kadis


writer = SummaryWriter('./log')


def save_checkpoint(accuracy, model, optimizer, args, epoch):
    print('Best Model Saving...')

    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()


    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))  


global_step = 0
total_accuracy = 0.0

def _train(epoch, train_loader, val_loader, model, optimizer, criterion, lr_scheduler, args):
    global best_ssim, best_psnr,global_step
    start = time.time()
    model.train()
    losses = 0.
    accuracy = 0.0
    best_accuracy = 0.0
    for idx, (img, target) in enumerate(train_loader):
        # if args.cuda:
        img, target = img.cuda(), target.cuda()

        iters_per_epoch = len(train_loader)
        # print(len(train_loader))
        optimizer.zero_grad()
        
        # Calculate random crop positions
        image_height, image_width = img.shape[-2], img.shape[-1]
        crop_size = 224
        crop_top = torch.randint(0, image_height - crop_size + 1, (1,))
        crop_left = torch.randint(0, image_width - crop_size + 1, (1,))

        # Crop img and img_res
        img = img[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
        

        output = model(img)
        # print(output.argmax(dim=1))
        # print(target)
        
        loss = criterion(output, target)
        # print(loss)
        
        global_step += 1
        # print(global_step)
        losses += loss.item()
        loss.backward()

        # if args.gradient_clip > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        lr_scheduler.step()
        writer.add_scalar('Training Loss',loss.item(), epoch * iters_per_epoch + idx)
        if (idx+1) % args.print_intervals == 0:
            print('[Epoch: {0:4d}], Loss: {1:.3f}'.format(epoch, losses / (idx + 1)))
        if (idx+1) % 800 == 0:
            accuracy = _eval(epoch, val_loader, model, args)
            print("Test Accuracy: {}".format(accuracy))
            if accuracy > best_accuracy:
              best_accuracy = accuracy
              save_checkpoint(accuracy, model, optimizer, args, epoch)
    end = time.time()
    print('Time:', end-start, 'Global_step:', global_step)
    return best_accuracy

def _eval(epoch, test_loader, model, args):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader):
            img, target = img.cuda(), target.cuda()

            # Calculate random crop positions
            image_height, image_width = img.shape[-2], img.shape[-1]
            crop_size = 224
            crop_top = torch.randint(0, image_height - crop_size + 1, (1,))
            crop_left = torch.randint(0, image_width - crop_size + 1, (1,))
            
            # Crop img and img_res
            img = img[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]

            output = model(img)
            
            # Calculate accuracy
            correct = (output.argmax(dim=1) == target).sum().item()
            total_correct += correct
            total_samples += target.size(0)

        accuracy = total_correct / total_samples
        writer.add_scalar('Testing Accuracy', accuracy, epoch)

    return accuracy
        

def main(args):
    train_loader, val_loader= load_data(args)
    # print(len(train_loader.dataset))
    # print(len(val_loader.dataset))
    model = ClassModel()
    model = model.cuda()


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', args.checkpoints), map_location=device)
        model.load_state_dict(checkpoints['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch'] + 1
    else:
        start_epoch = 1


    # if args.cuda:
    model = model.cuda()
    model.train(True)
    if not args.evaluation:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.000001)  # 定义余弦退火重启学习率调度器


        for epoch in range(start_epoch, args.epochs + 1):
            _train(epoch, train_loader, val_loader, model, optimizer, criterion, lr_scheduler, args)

            # lr_scheduler.step()
            lr_ = lr_scheduler.get_last_lr()[0]
            print('Current Learning Rate: {}'.format(lr_))
            writer.add_scalar('Learning Rate', lr_, epoch)
        writer.close()

    else:
          _eval(start_epoch, val_loader, model, args)



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

dataset_path ={
        'pretrain_path': '/home/user/zed/pretrain-class',
        'test_path':'/home/user/zed/iqadataset/kadid10k',
    'label_path':'/home/user/zed/pretrain-class/labels.csv'
}

def load_config():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--pretrain_path', type=str, default=dataset_path['pretrain_path'])  
    parser.add_argument('--test_path', type=str, default=dataset_path['test_path'])  
    parser.add_argument('--label_path', type=str, default=dataset_path['label_path'])
    parser.add_argument('--batch_size', type=int, default=72)  
    parser.add_argument('--resize_ratio', type=float, default=0.75) 
    parser.add_argument('--num_workers', type=int, default=8)  
    parser.add_argument('--lr', type=float, default=1e-4)  
    parser.add_argument('--weight_decay', type=float, default=1e-4) 
    parser.add_argument('--momentum', type=float, default=0.9)  
    parser.add_argument('--cuda', type=bool, default=True)  
    parser.add_argument('--epochs', type=int, default=20)  
    parser.add_argument('--print_intervals', type=int, default=40)  
    parser.add_argument('--evaluation', type=bool, default=False)  
    parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')  
    parser.add_argument('--device_num', type=int, default=1)  
    parser.add_argument('--gradient_clip', type=float, default=2.)  
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for Adam optimizer')

    return parser.parse_args()

if __name__ == '__main__':
    args = load_config()   
    main(args)
