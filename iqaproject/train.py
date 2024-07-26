import os
import random
import logging
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

import torch
import torch.optim

from configs import Configs
from models.networks import IQAModel
from data_loader import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
os.makedirs('./result', exist_ok=True)

# def lr_lambda(epoch):
#     t = epoch // 10
#     if epoch > 200:
#         return 1.0
#     else:
#         return 1.0 / (10 ** t)

def main(config):
    best_srocc, best_plcc, best_epoch = 0, 0, 0

    # define data loaders
    train_data = DataLoader(config, dataset=config.dataset, path=config.path, img_indx=config.train_index,
                            patch_num=config.train_patch_num, istrain=True)
    train_loader = train_data.get_data()
    test_data = DataLoader(config, dataset=config.dataset, path=config.path, img_indx=config.test_index,
                           patch_num=config.test_patch_num, istrain=False)
    test_loader = test_data.get_data()

    model = IQAModel()

   
    pretrained_model_path = 'class_model_best.pth'
    pretrained_dict = torch.load(pretrained_model_path)
    model_dict = model.state_dict()

  
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'class_branch' in k or 'fc' in k}

    
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict, strict=True)

    # for name, param in model.named_parameters():
    #     if 'class_branch' in name or 'fc' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    model = model.to(device)

    res_model = SwinTransformerSys(img_size=(384, 384), window_size=12)
    res_model = res_model.to(device)
    checkpoint = torch.load('checkpoint_model_best.pth')
    res_model.load_state_dict(checkpoint['model_state_dict'])
    res_model.eval()

    # Define the optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.learning_rate * 0.001,
                                                     max_lr=config.learning_rate,
                                                     step_size_up=100, step_size_down=100,
                                                     mode='triangular', scale_mode='cycle',
                                                     cycle_momentum=False)
    
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    # Define the loss function
    criterion = torch.nn.L1Loss().to(device)

    # Training loop
    logger.info('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC\tTest_SRCC\tTest_PLCC')
    for epoch in range(config.start_epoch, config.epochs + 1):
        # train one epoch
        train_srocc, train_plcc, loss = train(train_loader, test_loader, model, res_model, optimizer, lr_scheduler,
                                              criterion, epoch, config)
        lr_scheduler.step()

        if (epoch) % 5 == 0:
            test_srocc, test_plcc, all_preds, all_labels = validate(config, model, res_model, test_loader)

            logger.info('\t%d\t%6.5f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' % (
            epoch, sum(loss) / len(loss), train_srocc, train_plcc, test_srocc, test_plcc))

            # save the best results
            if best_srocc < test_srocc and best_plcc < test_plcc:
                best_srocc = test_srocc
                best_plcc = test_plcc
                # Save the trained model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_criterion': best_srocc,
                    'optimizer': optimizer.state_dict(),
                }, config)
                best_epoch = epoch

        # Early Stopping for over-fitting
        # if (epoch > (best_epoch + 75)) and (test_srocc < best_srocc) and epoch > int(config.epochs * 0.5):
        #     break

    logger.info('Best Performance: {}, {}'.format(best_srocc, best_plcc))
    logger.info("End Training!")

    if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
        return best_srocc, best_plcc
    else:
        return best_srocc, best_plcc


def train(train_loader, test_loader, model, res_model, optimizer, lr_scheduler, criterion, epoch, config,
          crop_size=224):
    model.train()
    pred_scores, gt_scores, epoch_loss = [], [], []


    # for i, (img, ref, labels) in enumerate(train_loader):
    for i, (img, labels) in enumerate(train_loader):
        # img, ref, labels= img.cuda(), ref.cuda(), labels.cuda()
        img, labels = img.to(device), labels.to(device)
        img_res = res_model(img)[0].to(device)
        
       

        image_height, image_width = img.shape[-2], img.shape[-1]

        # Calculate random crop positions
        crop_top = torch.randint(0, image_height - crop_size + 1, (1,))
        crop_left = torch.randint(0, image_width - crop_size + 1, (1,))

        # Crop img and img_res
        img = img[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
        img_res = img_res[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
        # ref = ref[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]

        optimizer.zero_grad()
        # preds ,error_loss= model(img,img_res,ref)
        preds = model(img, img_res)
        # print(preds.shape)

        loss = criterion(preds, labels)  
        # loss2 = error_loss

        # loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        epoch_loss.append(loss.item())

        pred_scores = pred_scores + preds.cpu().tolist()
        gt_scores = gt_scores + labels.cpu().tolist()

    train_srocc, _ = stats.spearmanr(pred_scores, gt_scores)
    train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    return train_srocc, train_plcc, epoch_loss


def validate(config, model, res_model, test_loader, crop_size=224):
    logger.info("***** Running validation *****")
    model.eval()
    all_preds, all_labels = [], []

    # for step, (img, _, labels) in enumerate(test_loader):
    for step, (img, labels) in enumerate(test_loader):
        img, labels = img.to(device), labels.to(device)
        img_res = res_model(img)[0].to(device)

        image_height, image_width = img.shape[-2], img.shape[-1]

        # Calculate random crop positions
        crop_top = torch.randint(0, image_height - crop_size + 1, (1,))
        crop_left = torch.randint(0, image_width - crop_size + 1, (1,))

        # Crop img and img_res
        img = img[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
        img_res = img_res[:, :, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]

        preds = model(img, img_res)

        all_preds = all_preds + preds.cpu().tolist()
        all_labels = all_labels + labels.cpu().tolist()

    all_preds = np.mean(np.reshape(np.array(all_preds), (-1, config.test_patch_num)), axis=1)
    all_labels = np.mean(np.reshape(np.array(all_labels), (-1, config.test_patch_num)), axis=1)

    test_srocc = test_protocol(all_preds, all_labels, protocol='srocc')
    test_plcc = test_protocol(all_preds, all_labels, protocol='plcc')
    # test_rmse = test_protocol(all_preds, all_label, protocol='rmse')
    model.train()
    return test_srocc, test_plcc, all_preds, all_labels


def test_protocol(preds, labels, protocol='srocc'):
    if protocol == 'srocc':
        result = stats.spearmanr(preds, labels)[0]
    elif protocol == 'plcc':
        result = stats.pearsonr(preds, labels)[0]
    elif protocol == 'krcc':
        result = stats.kendalltau(preds, labels)[0]
    elif protocol == 'rmse':
        result = np.sqrt(mean_squared_error(preds, labels))
    else:
        result = None
        logger.info('Invalid evaluation criteria were provided.')
    return result


def save_checkpoint(state, config):
    model_checkpoint = os.path.join(config.output_dir, "%s_checkpoint.pth.tar" % (config.dataset))
    torch.save(state, model_checkpoint)
    logger.info("Saved model checkpoints to [DIR: %s]", config.output_dir)


if __name__ == '__main__':

    dataset_path = {
        'live': '/home/zlz/iqadataset/LIVE',
        'csiq': '/home/zlz/iqadataset/CSIQ',
        'tid2013': '/home/zlz/iqadataset/tid2013',
        'kadid-10k': '/home/zlz/iqadataset/kadid10k',
        'livemd': 'D:\iqadataset\LIVEMD',
        'livec': '/home/zlz/iqadataset/ChallengeDB_release/ChallengeDB_release',
        'koniq-10k': '/home/user/zed/iqadataset/koniq10k',
        'live_livec': {
            'live': '/home/zlz/iqadataset/LIVE',
            'livec': '/home/zlz/iqadataset/ChallengeDB_release/ChallengeDB_release'
        },
        'kadid_koniq': {
            'kadid-10k': '/home/zlz/iqadataset/kadid10k',
            'koniq-10k': '/home/zlz/iqadataset/koniq10k'
        },
        'spaq': '/home/user/zed/iqadataset/SPAQ'
    }

    dataset_split_img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid-10k': list(range(0, 81)),

        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),

        'live_livec': [('live', i) for i in range(29)] + [('livec', i) for i in range(1162)],
        'kadid_koniq': [('kadid-10k', i) for i in range(81)] + [('koniq-10k', i) for i in range(10073)],
        'spaq': list(range(0, 11125))
    }

    config = Configs()
    logger.info('Experimental Configurations : %s ', config)
    # Setup logging
    logger.setLevel(level=logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./result/spaq_log.txt', mode='a', encoding='utf-8')
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    console_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("\033[91m" + "=" * 40)
    logger.info("== Training on {} for {} epochs ==".format(config.dataset, config.epochs))
    logger.info("=" * 40 + "\033[0m")

    config.path = dataset_path[config.dataset]

    srocc_all = np.zeros(config.train_test_round, dtype=np.float32)
    plcc_all = np.zeros(config.train_test_round, dtype=np.float32)

    if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
        dist_type_num = 24 if config.dataset == 'tid2013' else 25
        type_results = np.zeros((config.train_test_round, dist_type_num), dtype=np.float32)

    for i in range(1, config.train_test_round + 1):
        # Update the seed for each experiment
        seed = config.seed + i - 1
        logger.info('Using the seed = {} for {}-th experiment'.format(seed, i))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        total_num_images = dataset_split_img_num[config.dataset]
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(total_num_images)
        config.train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
        config.test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

        if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
            srocc_all[i - 1], plcc_all[i - 1] = main(config)
        else:
            srocc_all[i - 1], plcc_all[i - 1] = main(config)

        

    logger.info('{}: all srocc: {}'.format(config.dataset, srocc_all))
    logger.info('{}: all plcc: {}'.format(config.dataset, plcc_all))
    srocc_median, plcc_median = np.median(srocc_all), np.median(plcc_all)
    logger.info('%s : Testing SRCC %4.4f,\t PLCC %4.4f' % (config.dataset, srocc_median, plcc_median))
