import  argparse
from    utils import constants
from    utils.tools import load_vocab
import  os
from    math import inf, ceil
import  torch
from    torch import LongTensor, FloatTensor
from    torch.utils.data import TensorDataset, DataLoader
import  random
import  pickle
from    tqdm import tqdm
from    copy import deepcopy
from    utils.penalty import cal_penalty
from    utils.args import get_preprocess_config


def data_process(file, word2index, split_sent):
    '''
    Change word to index.
    '''
    def prepare_sequence(seq):
        return list(map(lambda word: word2index[constants.UNK_WORD] 
                        if word2index.get(word) is None else word2index[word], seq))
    if split_sent:
        with open(file, 'r', encoding='utf-8') as f:
            data = [[seq.strip().split() for seq in line.strip('\n').lower().split('\t')]
                    for line in f.readlines()]

        return [[prepare_sequence(seq) for seq in line] for line in tqdm(data)]
    else:
        with open(file, 'r', encoding='utf-8') as f:
            data = [line.strip('\n').strip(' ').replace('\t', ' ').lower().split()
                    for line in f.readlines()]

        return [prepare_sequence(line) for line in tqdm(data)]


def graph_process(file):

    with open(file, 'r') as f:
        graph = [[float(num) for num in line.strip('\n').strip().split()]
                 for line in f.readlines()]
    return graph


def key_func(x):
    max_src_len = max([len(seq) for seq in x[0]]) * len(x[0])
    if len(x) > 2:
        return max(max_src_len, len(x[2]))
    else:
        return max_src_len


def sort_data(data):
    return [(index, value) for index, value in sorted(list(enumerate(data)),
            key=lambda x: key_func(x[1]), reverse=True)]


class data_loader:

    def __init__(self, source, graph, target=None, penalty=None,
                 PAD_index=0, BOS_index=None, EOS_index=None):
        self.source = source
        self.graph = graph
        self.train_flag = False
        if target is not None:
            self.train_flag = True
            self.target_input = []
            self.target_output = []
            self.penalty = penalty
            for line in target:
                self.target_input.append([BOS_index] + line)
                self.target_output.append(line + [EOS_index])

        self.PAD_index = PAD_index
        self.process_flag = True

    def _get_tokens(self, shuffle, args):
        if self.train_flag:
            assert len(args.gram_penalty) == 4
            self.penalty = avg_penalty(self.penalty, args.gram_penalty)
        if self.process_flag:
            if self.train_flag:
                data = list(zip(self.source, self.graph, self.target_input, self.target_output, self.penalty))
                src_index, data = zip(*sort_data(data))
                self.source, self.graph, self.target_input, self.target_output, self.penalty = zip(*data)
            else:
                data = list(zip(self.source, self.graph))
                self.rank, data = zip(*sort_data(data))
                self.source, self.graph = zip(*data)

        self.process_flag = False
        st = 0
        total_len = len(self.source)
        index_pair = []
        while st < total_len:
            if self.train_flag:
                max_length = key_func([self.source[st], 0, self.target_input[st]])
            else:
                max_length = key_func([self.source[st]])
            ed = min(st + args.max_tokens // max_length, total_len)
            if ed == st:
                ed += 1
            index_pair.append((st, ed))
            st = ed
        data = []
        total_cnt = 0
        for (st, ed) in tqdm(index_pair):
            graph = (FloatTensor(self.graph[st:ed]) > args.graph_eps).long()
            graph_size = int(graph.size(-1) ** 0.5)
            graph = graph.view(ed - st, graph_size, graph_size)
            for i in range(ed - st):
                for j in range(graph_size):
                    graph[i][j][j] = 0
            for g in graph:
                for i in range(graph_size):
                    flag = True
                    for j in range(graph_size):
                        if (g[i][j] != 0 or g[j][i] != 0):
                            flag = False
                            break
                    if flag:
                        total_cnt += 1
            # graph += torch.eye(graph_size).view(1, graph_size, graph_size).long()
            # graph = (graph != 0).long()
            if self.train_flag:
                target_output = LongTensor(pad_batch(self.target_output[st:ed], self.PAD_index, dim=2))
                penalty = FloatTensor(self.penalty[st:ed])
                if penalty.size(1) >= target_output.size(1):
                    penalty = penalty[:, :target_output.size(1)]
                else:
                    length = target_output.size(1) - penalty.size(1)
                    penalty = torch.cat((penalty, torch.ones(penalty.size(0), length)), dim=-1)
                data.append((LongTensor(pad_batch(self.source[st:ed], self.PAD_index, dim=3)), graph,
                             LongTensor(pad_batch(self.target_input[st:ed], self.PAD_index, dim=2)),
                             target_output, penalty))
            else:
                data.append((LongTensor(pad_batch(self.source[st:ed], self.PAD_index, dim=3)), graph))
        print("Ioslated Node: %d" % total_cnt)
        if shuffle:
            random.shuffle(data)
        return data
  
    def restore_rank(self, data):
        rank_data = []
        rank = [(index, value) for index, value in 
                sorted(list(enumerate(self.rank)), key=lambda x: x[1], reverse=False)]
        self.rank, _ = zip(*rank)
        for index in self.rank:
            rank_data.append(data[index])
        return rank_data

    def set_param(self, shuffle, args, train_flag=True, seed=None):
        self.train_flag = train_flag
        if seed is not None:
            random.seed(seed)
        if args.max_tokens is not None:
            self.index = 0
            return self._get_tokens(shuffle, args), 1

        else:
            self.st_index = 0
            self.ed_index = 1
            return self._get_batchs, self.batch_size


def pad_batch(batch, pad_index, dim):
    assert dim == 2 or dim == 3
    if dim == 2:
        max_len = max([len(seq) for seq in batch])
        return [list(seq) + [pad_index] * (max_len - len(seq)) for seq in batch]
    else:
        max_len = max([len(seq) for line in batch for seq in line])
        return [[list(seq) + [pad_index] * (max_len - len(seq)) for seq in line] for line in batch]


def save_data_loader(dataloader, save_file):

    save_path = os.path.join(*os.path.split(save_file)[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_file, 'wb') as f:
        pickle.dump(dataloader, f)

def avg_penalty(penalty, val):
    for i in range(len(penalty[0])):
        for j in range(len(penalty[0][0])):
            tmp = 0
            for k in range(len(val)):
                tmp += penalty[k][i][j] * val[k]
            penalty[0][i][j] = tmp + 1
    return penalty[0]

def preprocess():

    args = get_preprocess_config()
    word2index, _ = load_vocab(args.vocab)
    print("===> Create source dataset.")
    source = data_process(
        file=args.source,
        word2index=word2index,
        split_sent=True
    )
    target = None
    src_line = None
    penalty = None
    if args.target is not None:
        print("===> Create target dataset.")
        target = data_process(
            file=args.target,
            word2index=word2index,
            split_sent=False
        )
        print("===> Create penalty coefficient.")
        src_line = data_process(
            file=args.source,
            word2index=word2index,
            split_sent=False
        )
        penalty = cal_penalty(src_line, target)
    print("===> Create graph dataset.")
    graph = graph_process(
        file=args.graph
    )

    assert len(source) == len(graph) 
    if target:
        assert len(source) == len(target)

    dataloader = data_loader(
            source=source,
            penalty=penalty,
            target=target,
            graph=graph,
            PAD_index=constants.PAD_index,
            BOS_index=constants.BOS_index,
            EOS_index=constants.EOS_index
        )

    save_data_loader(dataloader, args.save_file)


if __name__ == '__main__':

    preprocess()
