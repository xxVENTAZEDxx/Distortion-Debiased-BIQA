import argparse

def Configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', type=str, default='spaq',
                        help='synthetic distortion (|live|csiq|tid2013|kadid-10k) and authentic distortion (livec|koniq-10k)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='the path to save results')
    parser.add_argument('--checkpoints', type=str, default=None,
                        help='The path to the checkpoints file')

    
    # optimizer setup
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--betas', dest='betas', type=tuple, default=(0.9, 0.999),
                        help='betas for AdamW optimizer')
    parser.add_argument('--eps', dest='eps', type=float, default=1e-8,
                        help='epsilon for AdamW optimizer')




    # dataloader setup
    parser.add_argument('--train_bs', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('--eval_bs',  type=int, default=8,
                        help='batch size for validation')
    parser.add_argument('--train_patch_num', type=int, default=1,
                        help='the number of patch for training stage')
    parser.add_argument('--test_patch_num', type=int, default=30,
                        help='the number of patch for testing stage')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    # training setup
    parser.add_argument('--epochs', dest='epochs', type=int, default=150,
                        help='totall epochs for training model')
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=1,
                        help='start epoch')
    parser.add_argument('--seed', dest='seed', type=int, default=2035,
                        help='Set random seeds for replication')
    parser.add_argument('--train_test_round', dest='train_test_round', type=int, default=2,
                        help='train-test times')
    parser.add_argument('--cuda', default=True, action='store_false',
                        help='using gpu for training model')

    # cross-dataset test settings
    parser.add_argument('--train_dataset', type=str, default='live',
                        help='the training dataset in cross-dataset test')
    parser.add_argument('--test_dataset1', type=str, default='csiq',
                        help='the 1st test dataset')
    parser.add_argument('--test_dataset2', type=str, default='tid2013',
                        help='The 2nd test dataset')
    return parser.parse_args()

if __name__ == '__main__':
    config = Configs()
    for arg in vars(config):
        print(arg, getattr(config, arg))
