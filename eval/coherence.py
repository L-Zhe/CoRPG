from    transformers import AlbertTokenizer, AlbertForPreTraining
import  torch
from    torch import nn
from    torch.nn import functional as F
from    tqdm import tqdm
import  random
import  time
import  torch_optimizer as optim
import  argparse
from    nltk.tokenize import sent_tokenize


def split_sent(sent):
    ans = list(sent_tokenize(sent))
    return ans

def read(file, split=False):
    with open(file, 'r') as f:
        if split:
            return [[seq for seq in split_sent(line.strip('\n').strip())] for line in f.readlines()]
        else:
            return [line.strip('\n').strip().split('\t') for line in f.readlines()]

def write(data, file):
    with open(file, 'w') as f:
        for line in data:
            f.write(line)
            f.write('\n')

@torch.no_grad()
def cal_graph(data=None, pretrain_model=None):
    tokenizer = AlbertTokenizer.from_pretrained(pretrain_model)
    model = AlbertForPreTraining.from_pretrained(pretrain_model).cuda()
    model.eval()
    graph = []
    prompt = []
    next_sentence = []
    for seq in tqdm(data):
        for i in range(len(seq)):
            for j in range(len(seq)):
                prompt.append(seq[i])
                next_sentence.append(seq[j])
        encoding = move2cuda(tokenizer(prompt, next_sentence, return_tensors='pt', 
                             add_special_tokens=True, padding=True))
        outputs = model(**encoding, labels=torch.LongTensor([1] * len(prompt)).cuda())
        logits = outputs.sop_logits
        prob = F.softmax(logits, dim=-1)[:, 1].view(-1, len(seq) ** 2)
        _tmp = [' '.join([str(num) for num in line]) for line in prob.tolist()]
        graph.extend(_tmp)
        prompt = []
        next_sentence = []
    # if len(prompt) != 0:
    #     encoding = move2cuda(tokenizer(prompt, next_sentence, return_tensors='pt', 
    #                          add_special_tokens=True, padding=True))
    #     outputs = model(**encoding, labels=torch.LongTensor([1] * len(prompt)).cuda())
    #     logits = outputs.sop_logits
    #     prob = F.softmax(logits, dim=-1)[:, 1].view(-1, 25)
    #     _tmp = [' '.join([str(num) for num in line]) for line in prob.tolist()]
    #     graph.extend(_tmp)
    return graph

@torch.no_grad()
def COH(data, pretrain_model):
    tokenizer = AlbertTokenizer.from_pretrained(pretrain_model)
    model = AlbertForPreTraining.from_pretrained(pretrain_model).cuda()
    model.eval()
    total_prob = 0
    total_cnt = 0
    prompt = []
    next_sentence = []
    cnt = 0
    for seq in tqdm(data):
        for i in range(len(seq) - 1):
            prompt.append(seq[i])
            next_sentence.append(seq[i + 1])
        # print(len(seq))
        if len(seq) > 1:
            encoding = move2cuda(tokenizer(prompt, next_sentence, return_tensors='pt', 
                                 add_special_tokens=True, padding=True))
            outputs = model(**encoding, labels=torch.LongTensor([1] * len(prompt)).cuda())
            logits = outputs.sop_logits
            prob = F.softmax(logits, dim=-1)[:, 1].view(-1, len(seq) - 1)
            total_prob += prob.mean(dim=1).item()
            total_cnt += (prob > 0.5).float().mean(dim=-1).item()
        else:
            cnt += 1
        prompt = []
        next_sentence = []
    print("COH-p: %f\tCOH: %f" % (total_prob / (len(data) - cnt), total_cnt / (len(data) - cnt)))

def data_process(file, train_num=30000, test_num=1000):
    with open(file, 'r') as f:
        corpus = [[seq.lower().strip() for seq in line.strip('\n').strip().split('\t')] for line in f.readlines()]
    data = []
    for line in corpus:
        for i in range(len(line) - 1):
            data.append((line[i], line[i + 1]))
    random.shuffle(data)
    return data[:train_num], data[train_num:test_num + train_num]

def data_loader(data, batch_size):
    st = 0
    ed = batch_size
    random.shuffle(data)
    while st < len(data):
        _data = data[st:ed]
        st = ed
        ed = min(ed + batch_size, len(data))
        yield _data

def move2cuda(data):
    for key in data.keys():
        if data[key].size(1) > 512:
            data[key] = data[key][:, :512]
        data[key] = data[key].cuda()
    return data

def train(model,
          tokenizer,
          criterion,
          optim,
          train_data,
          test_data,
          batch_size,
          epoch,
          checkpoint_path,
          grad_accum_step):
    model.train()
    best_score = 0
    for e in range(epoch):
        total_loss = 0
        total_cnt = 0
        accum_cnt = 0
        st_time = time.time()
        for i, data in enumerate(data_loader(train_data, batch_size)):
            optim.zero_grad()
            prompt, next_sentence = zip(*data)
            label_true = torch.ones(len(data)).long().cuda()
            label_false = torch.zeros(len(data)).long().cuda()
            input_true = tokenizer(prompt, next_sentence, return_tensors='pt',
                                   padding=True, add_special_tokens=True)
            output_true = model(**move2cuda(input_true)).sop_logits
            loss_true = criterion(output_true, label_true) / 2
            total_loss += loss_true.item()
            loss_true.backward()
            input_false = tokenizer(next_sentence, prompt, return_tensors='pt',
                                    padding=True, add_special_tokens=True)
            output_false = model(**move2cuda(input_false)).sop_logits
            loss_false = criterion(output_false, label_false) / 2
            total_loss += loss_false.item()
            loss_false.backward()
            accum_cnt += 1
            if accum_cnt == grad_accum_step:
                optim.step()
                accum_cnt = 0
            total_cnt += 1
            if i % (10 * grad_accum_step) == 0:
                print("epoch: %d\tbatch: %d\tloss: %f\ttime: %d\tbest_acc: %f%%" 
                    % (e, i, total_loss / total_cnt, time.time() - st_time, best_score * 100))
                total_loss = total_cnt = 0
                st_time = time.time()
        if  accum_cnt != 0:
            optim.step()
        with torch.no_grad():
            model.eval()
            total_true = 0
            total_cnt = 0
            for data in tqdm(data_loader(test_data, batch_size)):
                prompt, next_sentence = zip(*data)
                input_true = tokenizer(prompt, next_sentence, return_tensors='pt', 
                                       padding=True, add_special_tokens=True)
                output_true = model(**move2cuda(input_true)).sop_logits
                total_true += (F.softmax(output_true, dim=-1)[:, 0] < 0.5).long().sum().item()
                input_false = tokenizer(next_sentence, prompt, return_tensors='pt', 
                                        padding=True, add_special_tokens=True)
                output_false = model(**move2cuda(input_false)).sop_logits
                total_true += (F.softmax(output_false, dim=-1)[:, 0] > 0.5).long().sum().item()
                total_cnt += 2 * len(data)
            acc = total_true / total_cnt
            print("valid acc: %f" % acc)
            if best_score <= acc:
                best_score = acc
                torch.save(model.state_dict(), checkpoint_path)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--coh', action='store_true')
    parser.add_argument('--pretrain_model', type=str)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--text_file', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    if args.train:
        tokenizer = AlbertTokenizer.from_pretrained(args.pretrain_model)
        model = AlbertForPreTraining.from_pretrained(args.pretrain_model).cuda()
        train_data, test_data = data_process(args.text_file)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Lamb(model.parameters(), lr=0.00176)
        train(model=model,
              tokenizer=tokenizer,
              criterion=criterion,
              optim=optimizer,
              train_data=train_data,
              test_data=test_data,
              batch_size=128,
              epoch=100,
              checkpoint_path=args.save_file,
              grad_accum_step=2)
    else if args.inference:
    # inference:
        file = args.text_file
        write(cal_graph(read(file), args.pretrain_model), file + '.graph')
    # COH score
    else if args.coh:
        COH(read(args.text_file, split=True), args.pretrain_model)
