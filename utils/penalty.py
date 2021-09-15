from    tqdm import tqdm

def sent_n_gram(sent, n):
    gram = []
    for line in sent:
        _gram = []
        for i in range(len(line) - n):
            _gram.append(' '.join(line[i:i+n]))
        gram.append(_gram)
    return gram

def word_n_gram(sent, n):
    gram = []
    for line in sent:
        _gram = []
        for i in range(len(line)):
            token = []
            for j in range(i - n + 1, i + 1):
                st = j
                ed = j + n
                if st >= 0 and ed <= len(line):
                    token.append(' '.join(line[st:ed]))
            _gram.append(token)
        gram.append(_gram)
    return gram

def penalty_flag(source, target, n):
    batch_size = len(target)
    sent_gram = sent_n_gram(source, n)
    word_gram = word_n_gram(target, n)
    flag = [[0 for _ in line] for line in target]
    for i in tqdm(range(batch_size)):
        for j in range(len(word_gram[i])):
            _flag = True
            for gram in word_gram[i][j]:
                if gram in sent_gram[i]:
                    _flag = False
                    break
            if _flag:
                flag[i][j] = 1
    return flag


def cal_penalty(source, target):
    max_length = max([len(line) for line in target])
    total_penalty = []
    source = [[str(word) for word in line] for line in source]
    target = [[str(word) for word in line] for line in target]
    for n in [1, 2, 3, 4]:
        print("===> Create %d-gram penalty." % n)
        penalty = [[0] * max_length for _ in range(len(target))]
        flag = penalty_flag(source, target, n)
        for i in range(len(flag)):
            for j in range(len(flag[i])):
                _flag = True
                for _n in range(n - 1):
                    if total_penalty[_n][i][j] == 1:
                        _flag = False
                        break
                if _flag:
                    penalty[i][j] = flag[i][j]
        total_penalty.append(penalty)
    return total_penalty