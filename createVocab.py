from    utils import constants
import  argparse
from    collections import Counter
from    math import inf
from    utils.tools import save_vocab


def create_vocab(file_list, vocab_num):
    def create_corpus(file):
        with open(file, 'r') as f:
            corpus = [word.lower() for line in f.readlines() for word in line.strip('\n').split()]
        return corpus
    corpus = []
    for file in file_list:
        corpus.extend(create_corpus(file))
    
    word2index = {}; index2word = {}
    word2index[constants.PAD_WORD] = constants.PAD_index
    index2word[constants.PAD_index] = constants.PAD_WORD
    word2index[constants.UNK_WORD] = constants.UNK_index
    index2word[constants.UNK_index] = constants.UNK_WORD
    word2index[constants.BOS_WORD] = constants.BOS_index
    index2word[constants.BOS_index] = constants.BOS_WORD
    word2index[constants.EOS_WORD] = constants.EOS_index
    index2word[constants.EOS_index] = constants.EOS_WORD
    
    if vocab_num != -1:
        w_count = [pair[0] for pair in Counter(corpus).most_common(vocab_num)]
    else:
        w_count = set(corpus)
    for word in w_count:
        word2index[word] = len(word2index)
        index2word[len(index2word)] = word
    return word2index, index2word


def main_vocab():
    parser = argparse.ArgumentParser(description='Create vocabulary.',
                                     prog='creat_vocab')

    parser.add_argument('-f', '--file', type=str, nargs='+',
                        help='File list to generate vocabulary.')
    parser.add_argument('--vocab_num', type=int, nargs='?', default=-1, 
                        help='Total number of word in vocabulary.')
    parser.add_argument('--save_path', type=str, default='./',
                        help='Path to save vocab.')
                                                
    args = parser.parse_args()
    word2index, index2word = create_vocab(file_list=args.file,
                                          vocab_num=args.vocab_num)
    print('Vocabulary Number: %d' % len(word2index))
    save_vocab(word2index, index2word, args.save_path)


if __name__ == '__main__':
    main_vocab()