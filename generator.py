import  torch
from    utils.lang import translate2word
import  argparse
from    utils import constants
from    tqdm import tqdm
import  importlib
from    torch.utils.data import DataLoader
from    utils.args import get_generate_config
from    utils.tools import load_vocab, save2file
from    utils.makeModel import make_model
from    utils.checkpoint import process_state_dict, load_model
from    utils.eval import Eval
from    preprocess import data_process, data_loader
from    utils import constants
import  os
from    os import environ
import  pickle


class generator:
    def __init__(self, *args, **kwargs):
        self.data = kwargs['data']
        self.index2word = kwargs['index2word']
        self.max_length = kwargs['max_length']
        self.model = kwargs['model']

    def _batch(self, st, ed):
        try:
            output = self.model(source=self.source[st:ed].cuda(),
                                graph=self.graph[st:ed].cuda(),
                                mode='test',
                                max_length=self.max_length)
            output = output.tolist()
            for i in range(len(output)):
                output[i] = output[i][1:]
                if constants.EOS_index in output[i]:
                    end_index = output[i].index(constants.EOS_index)
                    output[i] = output[i][:end_index]
                print(len(output[i]))
                
        except RuntimeError:
            if ed - st == 1:
                raise RuntimeError
            print('==>Reduce Batch Size')
            torch.cuda.empty_cache()
            output = []
            length = max(int((ed - st) / 4), 1)
            while st < ed:
                _ed = min(st + length, ed)
                output.extend(self._batch(st, _ed))
                st = _ed
        return output

    @torch.no_grad()
    def __call__(self):
        outputs = []
        self.model.eval()
        print('===>Start Generate.')
        for source, graph in tqdm(self.data):

            self.source = source[0]
            self.graph = graph[0]
            outputs.extend(self._batch(0, self.source.size(0)))
            print(translate2word(outputs[-1:], self.index2word))
        return translate2word(outputs, self.index2word)


def _main():

    args = get_generate_config()
    setattr(args, 'PAD_index', constants.PAD_index)
    setattr(args, 'BOS_index', constants.BOS_index)
    setattr(args, 'EOS_index', constants.EOS_index)
    assert (args.file is None) ^ (args.raw_file is None)
    _, tgt_index2word = load_vocab(args.vocab)
    vocab_size = len(tgt_index2word)
    setattr(args, 'vocab_size', vocab_size)

    environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_num)
    model_state_dict = load_model(args.model_path)
    model = make_model(args)
    model.load_state_dict(model_state_dict)
    model.cuda()
    if args.file is not None:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
    
    else:
        src_word2index, _ = load_vocab(args.src_vocab)
        source = data_process(filelist=[args.raw_file],
                              word2index=src_word2index)

        data = data_loader(source=source,
                           PAD_index=constants.PAD_index)

    dataset, batch_size = data.set_param(False, args, False)
    dataset = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    generate = generator(data=dataset,
                         index2word=tgt_index2word,
                         max_length=args.max_length,
                         model=model)
    
    outputs = generate()
    outputs = data.restore_rank(outputs)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    save_file = os.path.join(args.output_path, 'result.txt')
    save2file(outputs, save_file)

    if args.ref_file is not None:
        eval = Eval(reference_file=args.ref_file)
        eval(save_file)

if __name__ == '__main__':
    _main()