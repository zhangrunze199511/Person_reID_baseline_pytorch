from train import train
from test import test
from evaluate import eva
from evaluate_gpu import eva_gpu
import argparse

import logging


try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='PCB', type=str, help='output model name')
parser.add_argument('--data_dir',default='E:/workplace/data/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
parser.add_argument('--test_dir',default='E:/workplace/data/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query' )


opt = parser.parse_args()

def generateResult(name, fp16, PCB):
    opt.name = name
    opt.fp16 = fp16
    opt.PCB = PCB

    print(opt)

    print("training start")
    train(opt=opt)
    print("training finish")
    print("extracting features")
    test(opt)
    print("feature extracted, calculating results")
    eva_gpu()

if  __name__ == "__main__":
    logging.basicConfig(filename="running.log", level=logging.DEBUG, format='%(asctime)s %(message)s')
    generateResult(name='PCB', fp16= False, PCB= True)

    generateResult(name='ft_ResNet50', fp16= False, PCB= True)

    generateResult(name='ft_net_dense', fp16= False, PCB= False)



