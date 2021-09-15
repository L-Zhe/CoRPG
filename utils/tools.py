import  os
import  pickle
import  torch

def save_vocab(word2index, index2word, save_path):
    vocab = {'word2idx':  word2index,
             'idx2word':  index2word}
    
    file_path = os.path.join(*os.path.split(save_path)[:-1])
    if os.path.exists(file_path) == False:
        os.makedirs(file_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('===> Save Vocabulary Successfully.')

def load_vocab(save_path):
    with open(save_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab['word2idx'], vocab['idx2word']

def show_info(epoch, vocab_size, USE_CUDA):
    print('+' * 22)
    print('EPOCH: \t\t%d' % epoch)
    print('Vocab Size: \t%d' % vocab_size)
    print('USE_CUDA:\t', USE_CUDA)
    print("Let's use", torch.cuda.device_count(), "GPUs!")


def save2file(data, file):
    path = os.path.join(*os.path.split(file)[:-1])
    
    if not os.path.exists(path):
        os.makedirs(path)
    flag = False
    with open(file, 'w') as f:
        for line in data:
            if flag:
                f.write('\n')
            flag = True
            f.write(line)