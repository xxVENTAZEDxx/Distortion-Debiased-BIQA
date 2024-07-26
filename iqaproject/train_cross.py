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
from train import validate

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
os.makedirs('./results', exist_ok=True)


def lr_lambda(epoch):
    t = epoch // 30
    if epoch > 70:
        return 1.0
    else:
        return 1.0 / (10 ** t)


def main(config):
    train_data = DataLoader(config, config.train_dataset, path=dataset_path[config.train_dataset],
                            img_indx=dataset_split_img_num[config.train_dataset], patch_num=config.train_patch_num,
                            istrain=True)
    train_loader = train_data.get_data()
    # first_batch = next(iter(train_loader))
    # print(first_batch)

    test_data1 = DataLoader(config, config.test_dataset1, path=dataset_path[config.test_dataset1],
                            img_indx=dataset_split_img_num[config.test_dataset1], patch_num=config.test_patch_num,
                            istrain=False)
    test_loader1 = test_data1.get_data()

    test_data2 = DataLoader(config, config.test_dataset2, path=dataset_path[config.test_dataset2],
                            img_indx=dataset_split_img_num[config.test_dataset2], patch_num=config.test_patch_num,
                            istrain=False)
    test_loader2 = test_data2.get_data()

    test_data3 = DataLoader(config, config.test_dataset3, path=dataset_path[config.test_dataset3],
                            img_indx=dataset_split_img_num[config.test_dataset3], patch_num=config.test_patch_num,
                            istrain=False)
    test_loader3 = test_data3.get_data()

    test_data4 = DataLoader(config, config.test_dataset4, path=dataset_path[config.test_dataset4],
                            img_indx=dataset_split_img_num[config.test_dataset4], patch_num=config.test_patch_num,
                            istrain=False)
    test_loader4 = test_data4.get_data()

    # Create an instance of VIPNet model
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
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Define the loss function
    criterion = torch.nn.L1Loss().to(device)

    logger.info('Epoch\tTrain_Loss\tTrain_SROCC\tTest_on_{}_SROCC\tTest_on_{}_SROCC'.format(config.test_dataset1,
                                                                                            config.test_dataset2,
                                                                                            config.test_dataset3,
                                                                                            config.test_dataset4))
    all_test_srocc1 = []
    all_test_plcc1 = []
    all_test_srocc2 = []
    all_test_plcc2 = []
    all_test_srocc3 = []
    all_test_plcc3 = []
    all_test_srocc4 = []
    all_test_plcc4 = []

    for epoch in range(config.start_epoch, config.epochs + 1):
        # train one epoch
        train_srocc, train_plcc = train(
            train_loader, model, res_model, optimizer,lr_scheduler, criterion, epoch, config)
        lr_scheduler.step()

        if epoch % 3 == 0:
            test_srocc1, test_plcc1, _, _ = validate(config, model, res_model, test_loader1)
            test_srocc2, test_plcc2, _, _ = validate(config, model, res_model, test_loader2)
            test_srocc3, test_plcc3, _, _ = validate(config, model, res_model, test_loader3)
            test_srocc4, test_plcc4, _, _ = validate(config, model, res_model, test_loader4)

            logger.info(
                'Epoch\tTrain_Loss\tTrain_SROCC\tTest_on_{}_SROCC\tTest_on_{}_SROCC'.format(config.test_dataset1,
                                                                                            config.test_dataset2,
                                                                                            config.test_dataset3,
                                                                                            config.test_dataset4))

            test_srocc1, test_plcc1 = np.abs(test_srocc1), np.abs(test_plcc1)
            test_srocc2, test_plcc2 = np.abs(test_srocc2), np.abs(test_plcc2)
            test_srocc3, test_plcc3 = np.abs(test_srocc3), np.abs(test_plcc3)
            test_srocc4, test_plcc4 = np.abs(test_srocc4), np.abs(test_plcc4)
            all_test_srocc1.append(test_srocc1)
            all_test_plcc1.append(test_plcc1)
            all_test_srocc2.append(test_srocc2)
            all_test_plcc2.append(test_plcc2)
            all_test_srocc3.append(test_srocc3)
            all_test_plcc3.append(test_plcc3)
            all_test_srocc4.append(test_srocc4)
            all_test_plcc4.append(test_plcc4)
            best_srocc1 = np.max(all_test_srocc1)
            best_plcc1 = np.max(all_test_plcc1)
            best_srocc2 = np.max(all_test_srocc2)
            best_plcc2 = np.max(all_test_plcc2)
            best_srocc3 = np.max(all_test_srocc3)
            best_plcc3 = np.max(all_test_plcc3)
            best_srocc4 = np.max(all_test_srocc4)
            best_plcc4 = np.max(all_test_plcc4)
            logger.info('Epoch: {}\tBest_SROCC1: {}\tBest_PLCC1: {}\tBest_SROCC2: {}\tBest_PLCC2: {}\tBest_SROCC3: {}\tBest_PLCC3: {}\tBest_SROCC4: {}\tBest_PLCC4: {}'.format(
            epoch, best_srocc1, best_plcc1, best_srocc2, best_plcc2, best_srocc3, best_plcc3, best_srocc4, best_plcc4))

    test_srocc1, test_plcc1 = np.max(all_test_srocc1), np.max(all_test_plcc1)
    test_srocc2, test_plcc2 = np.max(all_test_srocc2), np.max(all_test_plcc2)
    test_srocc3, test_plcc3 = np.max(all_test_srocc3), np.max(all_test_plcc3)
    test_srocc4, test_plcc4 = np.max(all_test_srocc4), np.max(all_test_plcc4)
    logger.info('Training on {} and test on {}. SROCC: {}, PLCC: {}'.format(config.train_dataset, config.test_dataset1,
                                                                            test_srocc1, test_plcc1))

    logger.info('Training on {} and test on {}. SROCC: {}, PLCC: {}'.format(config.train_dataset, config.test_dataset2,
                                                                            test_srocc2, test_plcc2))
    logger.info('Training on {} and test on {}. SROCC: {}, PLCC: {}'.format(config.train_dataset, config.test_dataset3,
                                                                            test_srocc3, test_plcc3))
    logger.info('Training on {} and test on {}. SROCC: {}, PLCC: {}'.format(config.train_dataset, config.test_dataset4,
                                                                            test_srocc4, test_plcc4))

    return test_srocc1, test_plcc1, test_srocc2, test_plcc2, test_srocc3, test_plcc3, test_srocc4, test_plcc4


def train(train_loader, model, res_model, optimizer,lr_scheduler, criterion, epoch, config, crop_size=224):
    model.train()
    pred_scores, gt_scores, epoch_loss = [], [], []
    # for i, (img, ref, labels) in enumerate(train_loader):
    for i, (img, labels) in enumerate(train_loader):
        # img, ref, labels= img.cuda(), ref.cuda(), labels.cuda()
        # print(labels.shape)
        # print(img.shape)
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

    return train_srocc, train_plcc


if __name__ == '__main__':
    dataset_path = {
        'live': '/home/zlz/iqadataset/databaserelease2',
        'csiq': '/home/zlz/iqadataset/CSIQ',
        'tid2013': '/home/zlz/iqadataset/tid2013',
        'kadid-10k': '/home/zlz/iqadataset/kadid10k',
        'livemd': 'D:\iqadataset\LIVEMD',
        'livec': '/home/zlz/iqadataset/ChallengeDB_release/ChallengeDB_release',
        'koniq-10k': '/home/zlz/iqadataset/koniq10k',
        'livec_tid2013': {
        'livec': '/home/zlz/iqadataset/ChallengeDB_release/ChallengeDB_release',
        'tid2013': '/home/zlz/iqadataset/tid2013'
        },
    }

    dataset_split_img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid-10k': list(range(0, 81)),

        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'livec_tid2013': [('livec', i) for i in range(1162)] + [('tid2013', i) for i in range(25)],

    }
    config = Configs()
    # config.train_dataset = 'live'
    # config.test_dataset1 = 'csiq'
    # config.test_dataset2 = 'tid2013'

    logger.info('Experimental Configurations : %s ', config)
    # Setup logging
    logger.setLevel(level=logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./results/train_on_{}_test_on_{}_and_{}.txt'.format(config.train_dataset,
                                                                                            config.test_dataset1,
                                                                                            config.test_dataset2),
                                       mode='a', encoding='utf-8')
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')
    console_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("***** Training on {}, Testing on {} and {} *****".format(config.train_dataset, config.test_dataset1,
                                                                          config.test_dataset2, config.epochs))

    all_cross_performance = np.zeros((config.train_test_round, 8), dtype=np.float32)

    for i in range(1, config.train_test_round + 1):
        seed = config.seed + i - 1
        logger.info('Using the seed = {} for {}-th experiment'.format(config.seed * i, i))
        torch.manual_seed(i * config.seed)
        torch.cuda.manual_seed(i * config.seed)
        np.random.seed(i * config.seed)
        random.seed(i * config.seed)

        all_cross_performance[i - 1, :] = main(config)
    results_median = np.median(all_cross_performance, axis=0)
    logger.info('Testing on {} and Testing on {}'.format(config.test_dataset1, config.test_dataset2))
    logger.info('Results: {}'.format(results_median))
