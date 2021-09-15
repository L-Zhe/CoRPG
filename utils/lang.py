def translate2word(sequence, index2word):
    return [' '.join([index2word[index] for index in seq]).replace('@@', ' ').replace('  ', '')
            for seq in sequence]