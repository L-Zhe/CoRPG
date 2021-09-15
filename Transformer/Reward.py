from    bert_score import score
from    transformers import GPT2LMHeadModel, GPT2TokenizerFast
from    nltk.translate.bleu_score import corpus_bleu
import  torch
from    torch import nn
from    tqdm import tqdm
from    math import inf
bert_score_model = '/home/linzhe/tool/bert-score'

def cal_self_bleu(cands, refs):
    batch_size = len(cands)
    score = torch.FloatTensor(batch_size)
    for i in range(batch_size):
        cand = [cands[i].split()]
        ref = [[refs[i].split()]]
        score[i] = corpus_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25))
    return score

class PPL_GPT2:

    def __init__(self, model_file, device):
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_file).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_file)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, sents):
        ppl = torch.FloatTensor(len(sents)).to(self.device)
        max_length = self.model.config.n_positions
        for index, sent in tqdm(enumerate(sents)):
            encodings = self.tokenizer(sent, return_tensors='pt')
            input_ids = encodings.input_ids[:, :max_length].to(self.device)
            target_ids = input_ids.clone()
            outputs = self.model(input_ids, labels=target_ids)
            ppl[index] = torch.exp(outputs[0]).tolist()
        return ppl


class Reward:

    def __init__(self, ref_file, device):
        with open(ref_file, 'r') as f:
            self.source = [line.strip('\n').strip() for line in f.readlines()]
        self.bert_score = score(lang='en', rescale_with_baseline=False, device=device)
        self.ppl = PPL_GPT2('/home/linzhe/tool/transformers/gpt2_large', device=device)
        self.device = device
        
    def get_index_batch(self, index):
        outputs = []
        for i in index:
            outputs.append(self.source[i])
        return outputs

    def cal_div_r(self, sent, ref):
        _, _, bert_score = self.bert_score(sent, ref)
        bert_score = bert_score.cuda(self.device)
        bert_score.masked_fill_((bert_score < 0), 0)
        self_bleu = cal_self_bleu(sent, ref).cuda(self.device)
        return torch.exp(0.8 * torch.log(bert_score) + 0.2 * torch.log(self_bleu))

    @torch.no_grad()
    def __call__(self, cand, ref, index):
        batch_ref = self.get_index_batch(index)
        r_c = self.cal_div_r(cand, batch_ref)
        r_f = self.cal_div_r(ref, batch_ref)
        ppl_c = self.ppl(cand).cuda(self.device)
        ppl_f = self.ppl(ref).cuda(self.device)

        # return (r_c - r_f) + (ppl_c - ppl_f)
        return r_c + ppl_c - 0.8

if __name__ == '__main__':
    # cands = ['trump seems to think that kim can be swayed not simply by threats and pressure , but by flattery and promises as well . the white house released a four-minute video that showcased kim as someone who could be a great historical figure if only he would fundamentally change . the video also went to great lengths to show what north korea could gain economically were it to meet us demands . the president even spoke of the north ’s potential as a venue for real-estate development and tourism . what seems not to have occurred to trump is that such a future holds more peril than promise to someone whose family has ruled with an iron grip for three generations .']
    # ref = ['trump thinks Kim can not only be swayed by threats and pressure , but also flattery and promises . the White House released a four-minute video , showing Kim as someone who might have been a great historical figure if he had changed his mind . the video has also taken a great deal of effort to show what North Korea can gain from an economic point of view if it is to meet our demands . the president even talked about the potential of the North as a place for real estate development and tourism . it seems to me that such a future is more dangerous than a promise to someone whose family has ruled for three generations .']
    # bert_score = score(lang='en', rescale_with_baseline=True)
    # # print(bert_score)
    # print(bert_score(cands, ref))
    PPL = PPL_GPT2('/home/linzhe/tool/transformers/gpt2_large', 0)
    with open('/home/linzhe/document-level-paraphrase/data/eval_data/test.src', 'r') as f:
        cands = [line.strip('\n').strip().lower() for line in f.readlines()]
    import random
    random.shuffle(cands)
    print(PPL(cands).mean().item())