from train import train
from test import test
from evaluate import eva
from evaluate_rerank import eva_rerank
from evaluate_gpu import eva_gpu
import argparse
import yaml

import logging


try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

MARKET_DIR = '../data/market/pytorch'
DUKE_DIR = '../data/duke/pytorch'


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='PCB', type=str, help='output model name')
parser.add_argument('--data_dir',default='../data/market/pytorch',type=str, help='training dir path')
parser.add_argument('--test_dir',default='../data/duke/pytorch',type=str, help='./test_data')
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

def generateResult(name, fp16, PCB, batchSize, train_dir, test_dir):
    opt.name = name
    opt.fp16 = fp16
    opt.batchsize = batchSize
    opt.PCB = PCB
    opt.data_dir = train_dir
    print(opt)

    print("training start")
    train(opt=opt)
    print("training finish")

    opt.test_dir = test_dir
    for dir in test_dir:
        opt.test_dir = dir
        print("extracting features")
        test(opt)
        print("feature extracted, calculating results")
        eva_gpu(opt)
        eva_rerank(opt)

if  __name__ == "__main__":
    logging.basicConfig(filename="running.log", level=logging.DEBUG, format='%(asctime)s %(message)s')

    file_name = './evaluateResult.txt'
    with open(file_name, 'a+') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)


    generateResult(name='PCB', fp16= False, PCB= True, batchSize=16, train_dir=MARKET_DIR, test_dir= [MARKET_DIR, DUKE_DIR])

    generateResult(name='ft_ResNet50', fp16= False, PCB= True, batchSize=16, train_dir=MARKET_DIR, test_dir = [MARKET_DIR, DUKE_DIR])

    generateResult(name='ft_net_dense', fp16= False, PCB= False,batchSize=16, train_dir=MARKET_DIR, test_dir = [MARKET_DIR, DUKE_DIR])



