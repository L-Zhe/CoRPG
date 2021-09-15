def combine(file, save_file):
    with open(file, 'r') as f:
        data = [line.strip('\n').lower().strip() for line in f.readlines()]
    src = []
    cnt = 0
    tmp = []
    for line in data:
        cnt += 1
        tmp.append(line)
        if cnt == 5:
            cnt = 0
            src.append(' '.join(tmp))
            tmp = [] 

    with open(save_file, 'w') as f:
        for line in src:
            f.write(line)
            f.write('\n')
file = '/home/linzhe/document-level-paraphrase/baseline/tran_sent/outputs/result.txt'
save_file = '/home/linzhe/document-level-paraphrase/baseline/outputs/tran_sent.txt'
combine(file, save_file)