from    nltk.translate.bleu_score import corpus_bleu


def read_file(file, reference=False):
    with open(file, 'r') as f:
        if reference:
            data = [[seq.strip('\n').lower().split() for seq in line.strip('\n').split('\t')]
                      for line in f.readlines()]
        else:
            data = [[word.lower() for word in line.strip('\n').split()] for line in f.readlines()]
    return data


def cal_bleu(ref_file, cand_file):
    reference = read_file(ref_file, reference=True)
    candidate = read_file(cand_file, reference=False)
    bleu4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    print('BLEU4: %.2f' % bleu4, '(', end='')
    print(corpus_bleu(reference, candidate, weights=(1, 0, 0, 0)) * 100, '/', end='')
    print(corpus_bleu(reference, candidate, weights=(0, 1, 0, 0)) * 100, '/', end='')
    print(corpus_bleu(reference, candidate, weights=(0, 0, 1, 0)) * 100, '/', end='')
    print(corpus_bleu(reference, candidate, weights=(0, 0, 0, 1)) * 100, ')')


if __name__ == '__main__':
    ref_file = '/home/linz/data/eval_data/test.src'
    cand_file = '/home/linz/coherence3_gru/data/output/result.txt'
    cal_bleu(ref_file, cand_file)