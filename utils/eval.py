from    nltk.translate.bleu_score import corpus_bleu

class Eval:

    def __init__(self, reference_file):
        self.reference = self.read_file(reference_file, reference=True)

    def read_file(self, file, reference=False):
        with open(file, 'r') as f:
            if reference:
                data = [[seq.strip('\n').lower().split() for seq in line.strip('\n').split('\t')]
                          for line in f.readlines()]
            else:
                data = [[word.lower() for word in line.strip('\n').split()] for line in f.readlines()]
        return data

    def __call__(self, candidate_file):
        candidate = self.read_file(candidate_file, reference=False)
        bleu4 = corpus_bleu(self.reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)) * 100
        print(corpus_bleu(self.reference, candidate, weights=(1, 0, 0, 0)) * 100)
        print(corpus_bleu(self.reference, candidate, weights=(0, 1, 0, 0)) * 100)
        print(corpus_bleu(self.reference, candidate, weights=(0, 0, 1, 0)) * 100)
        print(corpus_bleu(self.reference, candidate, weights=(0, 0, 0, 1)) * 100)
        print('BLEU4: %.2f' % bleu4)
        return bleu4, str(bleu4)

if __name__ == '__main__':
    source = '/home/linzhe/document-level-paraphrase/data/eval_data/test.src'
    target = '/home/linzhe/document-level-paraphrase/data/eval_data/test.tgt'
    # target = '/home/linzhe/document-level-paraphrase/baseline/outptus/transformer.tgt'
    # source = '/home/linzhe/paraphrase/data/paraNMT_small/0.7-0.8-div2/test.src'
    # target = '/home/linzhe/paraphrase/data/paraNMT_small/0.7-0.8-div2/test.tgt'
    eval = Eval(source)
    eval(target)
