import os
os.environ['CUDA_VISIBLE_DEVICES']= '2'
import torch
import tifffile
from torch.utils.data import DataLoader
from BioSR_dataset import Dataset
from argparse import ArgumentParser
from train import train
from test import test



def build_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OIDN", help="selecting the training model")
    parser.add_argument("--resume", type=str, default="", help="Resume path (default: none)")
    parser.add_argument("--chinkpoint_for_test", type=str, default="real path to your checkpoint", help="Resume path (default: none)")
    parser.add_argument("--datapath", type=str, default='./../', help="data path")
    parser.add_argument("--datalist", type=str, default='datalist/train_F-actin_list1.txt', help="train list") #data path to the input
    parser.add_argument("--evallist", type=str, default='datalist/test_F-actin/test_F-actin_level_05_list.txt', help="eval list")
    parser.add_argument("--testlist", type=str, default='datalist/test_502X502_F-actin_low.txt', help="test list")
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=10, help="Number of epochs to train")

    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--weight', type=float, default=0.9, help='weight')
        
    parser.add_argument("--log_pth", type=str, default='./logs/log_test.txt')
    parser.add_argument("--results_path", type=str, default='Output/results/')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help="data path")
 
    parser.add_argument("--test_frequency", type=int, default=1000)
    parser.add_argument("--model_save_frequency", type=int, default=1)
    
    parser.add_argument("--train_or_test", type=str, default='train')


    args = parser.parse_args()
    return args


def main(args):
    if args.train_or_test == 'train':
       train_data=Dataset(args.datapath,args.datalist,mode='train')
       eval_data=Dataset(args.datapath,args.evallist,mode='eval')
       train_loader=DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
       eval_loader=DataLoader(eval_data, batch_size=1, shuffle=False)
       train(args, train_loader, eval_loader)
    elif args.train_or_test == 'test':
       test_data=Dataset(args.datapath,args.testlist,mode='test') 
       test_loader=DataLoader(test_data, batch_size=1, shuffle=False)
       test(args, test_loader)
    else:
       print("Error: wrong args for train_or_test!")
       exit()
if __name__ == "__main__":
    args=build_args()
    main(args)
    print("over!")


