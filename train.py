import  torch
from    utils.tools import load_vocab, show_info
from    utils.trainer import trainer
from    utils.args import get_train_parser
from    math import inf
from    os import environ
import  importlib
from    preprocess import data_loader
import  random
import  warnings
warnings.filterwarnings("ignore")


def train():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    args = get_train_parser()
    _, tgt_index2word = load_vocab(args.vocab)
    vocab_size = len(tgt_index2word)

    setattr(args, 'vocab_size', vocab_size)
    environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_num)

    show_info(epoch=args.epoch,
              vocab_size=vocab_size,
              USE_CUDA=len(args.cuda_num) != 0)

    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = '8878'
    mp = importlib.import_module('torch.multiprocessing')
    seed = random.randint(0, 2048)
    setattr(args, 'world_size', len(args.cuda_num))
    mp.spawn(trainer, nprocs=args.world_size, args=(args, seed)) 


if __name__ == '__main__':
    train()
