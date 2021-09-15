import  torch
from    .Module import triu_mask
from    math import inf
from    torch.nn import functional as F


class SearchMethod:

    def __init__(self, search_method, BOS_index, EOS_index, beam=5):
        assert search_method in ['greedy', 'beam']
        self.BOS_index = BOS_index
        self.EOS_index = EOS_index
        self.search_method = search_method
        self.beam = 5
        self.return_prob = False

    def __call__(self, *args, **kwargs):

        if self.search_method == 'greedy':
            return self.greedy_search(*args, **kwargs)
        else:
            return self.beam_search(*args, **kwargs)

    @torch.no_grad()
    def greedy_search(self, decoder, tgt_embed, src_pad_mask,
                     encoder_outputs, max_length, source):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        sentence = torch.LongTensor([self.BOS_index] * batch_size).reshape(-1, 1).to(device)
        end_flag = torch.BoolTensor(batch_size).fill_(False).to(device)
        i = 0
        memory = None
        while i < max_length:
            embed = tgt_embed(sentence[:, -1:], i)
            seq_mask = triu_mask(i + 1).cuda()
            prob, memory = decoder(embed=embed, 
                                   encoder_outputs=encoder_outputs, 
                                   src_pad_mask=src_pad_mask,
                                   memory=memory,
                                   source=source)
            prob, word = prob.topk(5, dim=-1, largest=True)
            index = prob.squeeze(1).multinomial(1)
            index = index.repeat(1, 5)
            word = word.squeeze(1).gather(dim=-1, index=index)[:, :1]
            sentence = torch.cat((sentence, word), dim=1)
            mask = (word == self.EOS_index).view(-1).masked_fill_(end_flag, False)
            end_flag |= mask
            if (end_flag == False).sum() == 0:  
                break
            i += 1
        del memory, embed, mask, end_flag
        return sentence

    # def beam_search(self, decoder, tgt_embed, src_pad_mask, 
    #                 encoder_outputs, max_length):

    #     batch_size = encoder_outputs.size(0)
    #     device = encoder_outputs.device
    #     srcLen = encoder_outputs.size(1)
    #     # generate first word.
    #     sentence = torch.LongTensor(batch_size, 1).fill_(self.BOS_index).to(device)
    #     embed = tgt_embed(sentence)
    #     seq_mask = triu_mask(1).to(device)
    #     embed = decoder(embed, 
    #                     encoder_outputs, 
    #                     src_pad_mask, 
    #                     seq_mask)
    #     prob = F.log_softmax(embed[:, -1, :], dim=-1)
    #     bos_mask = torch.BoolTensor(1, prob.size(-1)).cuda().fill_(False)
    #     bos_mask[0, self.EOS_index] = True
    #     bos_mask = bos_mask.repeat(batch_size, 1)
    #     prob.masked_fill_(bos_mask, -inf)
    #     totalProb, word = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True)
        
    #     sentence = sentence.unsqueeze(1).repeat(1, self.beam, 1).view(batch_size * self.beam, -1)
    #     sentence = torch.cat((sentence, word.view(-1, 1)), dim=-1)
    #     eos_flag = (word == self.EOS_index)
    #     # generate other word
    #     encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, self.beam, 1, 1)
    #     encoder_outputs = encoder_outputs.view(batch_size * self.beam, srcLen, -1)
    #     src_pad_mask = src_pad_mask.unsqueeze(1).repeat(1, self.beam, 1, 1)
    #     src_pad_mask = src_pad_mask.view(batch_size * self.beam, 1, -1)

    #     i = 1
    #     while i < max_length:
           
    #         embed = tgt_embed(sentence)    # [Batch*beam, 1, hidden]
        
    #         seq_mask = triu_mask(i + 1).to(device)
    #         embed = decoder(embed, 
    #                         encoder_outputs, 
    #                         src_pad_mask, 
    #                         seq_mask)
    #         prob = F.log_softmax(embed[:, -1, :], dim=-1)
    #         flatten_eos_index = eos_flag.view(batch_size * self.beam, 1)
    #         vocab_size = prob.size(-1)
    #         mask = flatten_eos_index.repeat(1, vocab_size)
    #         prob.masked_fill_(mask, -inf)
            
    #         for j in range(mask.size(0)):
    #             if flatten_eos_index[j, 0]:
    #                 mask[j, 1:] = False
    #         prob.masked_fill_(mask, 0)
    #         prob += totalProb.view(-1, 1)
    #         totalProb, index = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True, sorted=True)
    #         # prob, index: [batch_size, beam]
    #         word = index % vocab_size
    #         index = index // vocab_size
    #         # print(index)
    #         eos_flag = eos_flag.gather(dim=-1, index=index)
    #         eos_flag |= (word == self.EOS_index)
    #         if eos_flag.sum() == batch_size * self.beam or \
    #            eos_flag[:, 0].sum() == batch_size:
    #             break
    #         sentence = sentence.view(batch_size, self.beam, -1)
    #         sentence = sentence.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i + 1))
    #         sentence = torch.cat((sentence, word.unsqueeze(-1)), dim=-1)
    #         sentence = sentence.view(batch_size * self.beam, -1)
    #         i += 1
    #     index = totalProb.max(1)[1]
    #     sentence = sentence.view(batch_size, self.beam, -1)
    #     outputs = torch.LongTensor().to(device)
    #     for i in range(batch_size):
    #         sent = sentence[i:i+1, index[i], :]
    #         outputs = torch.cat((outputs, sent), dim=0)
    #     return outputs

    @torch.no_grad()
    def beam_search(self, decoder, tgt_embed, src_pad_mask,
                    encoder_outputs, max_length, source):

        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        srcLen = encoder_outputs.size(1)
        # generate first word.
        sentence = torch.LongTensor(batch_size, 1).fill_(self.BOS_index).to(device)
        embed = tgt_embed(sentence, 0)
        prob, memory = decoder(embed=embed, 
                               encoder_outputs=encoder_outputs, 
                               src_pad_mask=src_pad_mask,
                               source=source)
        prob = torch.log(prob)
        # prob = F.log_softmax(embed[:, -1, :], dim=-1)
        prob = prob.squeeze(1)
        vocab_size = prob.size(-1)
        bos_mask = torch.BoolTensor(1, prob.size(-1)).cuda().fill_(False)
        bos_mask[0, self.EOS_index] = True
        bos_mask = bos_mask.repeat(batch_size, 1)
        prob.masked_fill_(bos_mask, -inf)
        totalProb, word = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True)
        
        sentence = sentence.unsqueeze(1).repeat(1, self.beam, 1).view(batch_size * self.beam, -1)
        sentence = torch.cat((sentence, word.view(-1, 1)), dim=-1)
        eos_flag = (word == self.EOS_index)
        # generate other word
        # memory: [batch, beam, num_layers, num_head, seq_length, dim, 2]
        num_layers, num_heads, _, dimension = memory.size()[1:5]
        memory = memory.unsqueeze(1).repeat(1, self.beam, 1, 1, 1, 1, 1)\
                       .view(batch_size * self.beam, num_layers, num_heads, 1, dimension, 2)
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, self.beam, 1, 1)
        encoder_outputs = encoder_outputs.view(batch_size * self.beam, srcLen, -1)
        source = source.unsqueeze(1).repeat(1, self.beam, 1)
        source = source.view(batch_size * self.beam, -1)
        # if src_pad_mask:
        src_pad_mask = src_pad_mask.unsqueeze(1).repeat(1, self.beam, 1, 1)
        src_pad_mask = src_pad_mask.view(batch_size * self.beam, 1, -1) 
        i = 1
        # z = z.unsqueeze(1).repeat(1, self.beam, 1, 1).view(batch_size * self.beam, 1, -1)
        while i < max_length:
            embed = tgt_embed(sentence[:, -1:], i)   # [Batch*beam, 1, hidden]
        
            # seq_mask = triu_mask(i + 1).to(device)
            prob, memory = decoder(embed=embed, 
                                   encoder_outputs=encoder_outputs, 
                                   src_pad_mask=src_pad_mask,
                                   memory=memory,
                                   source=source)
            # prob = F.log_softmax(embed[:, 0, :], dim=-1)
            prob = torch.log(prob)
            prob = prob.squeeze(1)
            flatten_eos_index = eos_flag.view(batch_size * self.beam, 1)
            mask = flatten_eos_index.repeat(1, vocab_size)
            prob.masked_fill_(mask, -inf)
            
            for j in range(mask.size(0)):
                if flatten_eos_index[j, 0]:
                    mask[j, 1:] = False
            prob.masked_fill_(mask, 0)
            prob += totalProb.view(-1, 1)
            totalProb, index = prob.view(batch_size, -1).topk(self.beam, dim=-1, largest=True, sorted=True)
            # prob, index: [batch_size, beam]
            word = index % vocab_size
            index = index // vocab_size
            eos_flag = eos_flag.gather(dim=-1, index=index)
            eos_flag |= (word == self.EOS_index)
            if eos_flag.sum() == batch_size * self.beam or \
               eos_flag[:, 0].sum() == batch_size:
                break
            sentence = sentence.view(batch_size, self.beam, -1)
            sentence = sentence.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, i + 1))
            sentence = torch.cat((sentence, word.unsqueeze(-1)), dim=-1)
            sentence = sentence.view(batch_size * self.beam, -1)
            memory = memory.view(batch_size, self.beam, num_layers, num_heads, i + 1, dimension, 2)
            index = index.view(batch_size, self.beam, 1, 1, 1, 1, 1)\
                         .repeat(1, 1, num_layers, num_heads, i + 1, dimension, 2)
            memory = memory.gather(dim=1, index=index)\
                           .view(batch_size * self.beam, num_layers, num_heads, i + 1, dimension, 2)
            i += 1
        # index = totalProb.max(1)[1]
        sentence = sentence.view(batch_size, self.beam, -1)
        outputs = torch.LongTensor().to(device)
        for i in range(batch_size):
            sent = sentence[i:i+1, 0, :]
            outputs = torch.cat((outputs, sent), dim=0)
        return outputs
