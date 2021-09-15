import  torch
from    torch import nn
from    torch.nn import functional as F
from    numpy import inf
import  math
from    copy import deepcopy
from    torch.autograd import Variable


def pad_mask(inputs, PAD):
    return (inputs == PAD).unsqueeze(-2)


def triu_mask(length):
    mask = torch.ones(length, length).triu(1)
    return mask.unsqueeze(0) == 1


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_qk, d_v, num_head, return_weight=False):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_qk = d_qk
        self.d_v = d_v
        self.W_Q = Linear(d_model, num_head * d_qk)
        self.W_K = Linear(d_model, num_head * d_qk)
        self.W_V = Linear(d_model, num_head * d_v)
        self.W_out = Linear(d_v * num_head, d_model)
        self.return_weight = return_weight

    def ScaledDotProductAttention(self, query, keys, values, mask=None):
        score = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.d_model)
        if mask is not None:
            score.masked_fill_(mask.unsqueeze(1), -inf)   
        weight = F.softmax(score, dim=-1)
        return torch.matmul(weight, values), weight

    def forward(self, Q, K, V, mask=None, ):
        batch_size = Q.size(0)
        query = self.W_Q(Q).view(batch_size, Q.size(1), self.num_head, self.d_qk)
        keys = self.W_K(K).view(batch_size, K.size(1), self.num_head, self.d_qk)
        values = self.W_V(V).view(batch_size, V.size(1), self.num_head, self.d_v)
        query.transpose_(1, 2)
        keys.transpose_(1, 2)
        values.transpose_(1, 2)

        outputs, weight = self.ScaledDotProductAttention(query, keys, values, mask)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v*self.num_head)
        if self.return_weight:
            return self.W_out(outputs), weight
        else:
            return self.W_out(outputs)

    def cal_one_vector(self, vector, memory, memory_new, i):
        # memory: [batch * beam, num_layer, num_head, seq_len, dimension, 2]
        batch_size = vector.size(0)
        query = self.W_Q(vector).view(batch_size, vector.size(1), self.num_head, self.d_qk)
        key = self.W_K(vector).view(batch_size, vector.size(1), self.num_head, self.d_qk)
        value = self.W_V(vector).view(batch_size, vector.size(1), self.num_head, self.d_v)

        query.transpose_(1, 2)
        key.transpose_(1, 2)
        value.transpose_(1, 2)
        outputs = torch.cat((key.unsqueeze(-1), value.unsqueeze(-1)), dim=-1)
        if memory is not None:
            if memory_new is None:
                memory_new = torch.cat((memory[:, i, ...], outputs), dim=2).unsqueeze(1)
            else:
                _m = torch.cat((memory[:, i, ...], outputs), dim=2)
                memory_new = torch.cat((memory_new, _m.unsqueeze(1)), dim=1)
        else:
            if memory_new is None:
                memory_new = outputs.unsqueeze(1)
            else:
                memory_new = torch.cat((memory_new, outputs.unsqueeze(1)), dim=1)

        outputs, weight = self.ScaledDotProductAttention(
                query, 
                memory_new[:, i, ..., 0],
                memory_new[:, i, ..., 1]
            )
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v * self.num_head)
        if self.return_weight:
            return self.W_out(outputs), memory_new, weight
        else:
            return self.W_out(outputs), memory_new


class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0., max_length=512, padding_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.register_buffer('PE', self.PositionalEncoding(0, max_length, embed_size))
        self.max_length = max_length
        self.d_model = embed_size
        self.embed_scare = math.sqrt(embed_size)

        nn.init.normal_(self.embed.weight, mean=0, std=embed_size ** -0.5)
        nn.init.constant_(self.embed.weight[padding_idx], 0)

    def PositionalEncoding(self, st, ed, embedding_dim):

        position = torch.arange(st, ed).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * 
                             -(math.log(10000.0) / embedding_dim))
        tmp = position * div_term
        pe = torch.zeros(ed - st, embedding_dim)
        pe[:, 0::2] = torch.sin(tmp)
        pe[:, 1::2] = torch.cos(tmp)  

        return pe.detach_()

    def forward(self, inputs, add_item=None):
        seq_length = inputs.size(1)
        if seq_length < self.max_length:
            outputs = self.embed(inputs) * self.embed_scare + self.PE[:seq_length]
        else:
            pe = self.PositionalEncoding(self.max_length, seq_length, self.d_model).to(inputs.device)
            pe = torch.cat((self.PE, pe), dim=0)
            outputs = self.embed(inputs) * self.embed_scare + pe
        if add_item is not None:
            outputs += add_item
        return self.dropout(outputs)

    def single_embed(self, inputs, i):
        if i < self.max_length:
            outputs = self.embed(inputs) * self.embed_scare + self.PE[i:i + 1]
        else:
            pe = self.PositionalEncoding(i, i + 1, self.d_model).to(inputs.device)
            outputs = self.embed(inputs) * self.embed_scare + pe
        return outputs


class PositionWiseFeedForwardNetworks(nn.Module):

    def __init__(self, input_size, output_size, d_ff):
        super().__init__()
        self.W_1 = Linear(input_size, d_ff, bias=True)
        self.W_2 = Linear(d_ff, output_size, bias=True)
        nn.init.constant_(self.W_1.bias, 0.)
        nn.init.constant_(self.W_2.bias, 0.)

    def forward(self, inputs):
        outputs = F.relu(self.W_1(inputs), inplace=True)
        return self.W_2(outputs)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, normalize_before):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.normalize_before = normalize_before

    def forward(self, x, sublayer, return_weight=False):
        "Apply residual connection to any sublayer with the same size."
        if return_weight:
            if self.normalize_before:
                x_tmp, weight = sublayer(self.norm(x))
                return x + self.dropout(x_tmp), weight
            else:
                x_tmp, weight = sublayer(x)
                return self.norm(x + self.dropout(x_tmp)), weight
        else:
            if self.normalize_before:
                return x + self.dropout(sublayer(self.norm(x)))
            else:
                return self.norm(x + self.dropout(sublayer(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class EncoderCell(nn.Module):

    def __init__(self, d_model, attn, FFNlayer, dropout, normalize_before):
        super().__init__()
        self.attn = deepcopy(attn)
        self.FFN = deepcopy(FFNlayer)
        self.sublayer = clones(SublayerConnection(d_model, dropout, normalize_before), 2)

    def forward(self, inputs, pad_mask):
        inputs = self.sublayer[0](inputs, lambda x: self.attn(x, x, x, pad_mask))
        return self.sublayer[1](inputs, self.FFN)


class DecoderCell(nn.Module):
    def __init__(self, d_model, attn, FFNlayer, dropout, normalize_before):
        super().__init__()
        self.cross_attn = deepcopy(attn)
        self.self_attn = deepcopy(attn)
        self.cross_attn.return_weight = True
        self.FFN = deepcopy(FFNlayer)
        self.sublayer = clones(SublayerConnection(d_model, dropout, normalize_before), 3)

    def forward(self, inputs, encoder_outputs, pad_mask, seq_mask):
        m = encoder_outputs
        inputs = self.sublayer[0](inputs, lambda x: self.self_attn(x, x, x, seq_mask))
        inputs, weight = self.sublayer[1](inputs, lambda x: self.cross_attn(x, m, m, pad_mask), return_weight=True)
        return self.sublayer[2](inputs, self.FFN), weight

    def memory_decode(self, inputs, memory, memory_new, encoder_outputs, pad_mask, i):
        m = encoder_outputs
        if self.sublayer[0].normalize_before:
            outputs = self.sublayer[0].norm(inputs)
            outputs, memory_new = self.self_attn.cal_one_vector(outputs, memory, memory_new, i)
            outputs = self.sublayer[0].dropout(outputs) + inputs
        else:
            outputs, memory_new = self.self_attn.cal_one_vector(inputs, memory, memory_new, i)
            outputs = self.sublayer[0].dropout(outputs) + inputs
            outputs = self.sublayer[0].norm(outputs)
        outputs, weight = self.sublayer[1](outputs, lambda x: self.cross_attn(x, m, m, pad_mask), return_weight=True)
        return self.sublayer[2](outputs, self.FFN), memory_new, weight


class Encoder(nn.Module):

    def __init__(self, d_model, num_layer, layer, normalize_before=False):
        super().__init__()
        self.encoder = nn.ModuleList([deepcopy(layer) for _ in range(num_layer)])    
        self.normalize_before = normalize_before
        if normalize_before:
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, pad_mask):
        for encoder_cell in self.encoder:
            inputs = encoder_cell(inputs, pad_mask)
        if self.normalize_before:
            inputs = self.layer_norm(inputs)
        return inputs


class Decoder(nn.Module):

    def __init__(self, d_model, num_layer, layer, normalize_before=False):
        super().__init__()
        self.num_layer = num_layer
        self.decoder = nn.ModuleList([deepcopy(layer) for _ in range(num_layer)])
        self.normalize_before = normalize_before
        if normalize_before:
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, encoder_outputs, pad_mask, seq_mask):
        for decoder_cell in self.decoder:
            inputs, weight = decoder_cell(inputs, encoder_outputs, pad_mask, seq_mask)
        if self.normalize_before:
            inputs = self.layer_norm(inputs)

        return inputs, weight.mean(dim=1)

    def generate(self, embed, encoder_outputs, src_pad_mask, memory=None):
        memory_new = None
        for (i, decoder) in enumerate(self.decoder):
            embed, memory_new, weight = decoder.memory_decode(inputs=embed,
                                                              memory=memory,
                                                              memory_new=memory_new,
                                                              encoder_outputs=encoder_outputs,
                                                              pad_mask=src_pad_mask,
                                                              i=i)
        if self.normalize_before:
            embed = self.layer_norm(embed)

        return embed, memory_new, weight.mean(dim=1)


class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0., ignore_index=None):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, penalty=None):
        # inputs = F.log_softmax(inputs, dim=-1)
        inputs = torch.log(inputs)
        batch_size = inputs.size(0)
        vocab_size = inputs.size(-1)
        if self.ignore_index is not None:
            norm = (targets != self.ignore_index).sum(dim=-1, keepdim=True)
        else:
            norm = torch.FloatTensor(batch_size, 1).fill_(targets.size(1))
        norm = norm.cuda(targets.device)
        if self.ignore_index is not None:
            mask = (targets == self.ignore_index)

        targets = F.one_hot(targets, num_classes=inputs.size(-1))
        targets = targets * (1 - self.smoothing) + self.smoothing / vocab_size
        loss = self.criterion(inputs.view(-1, vocab_size), 
                              targets.view(-1, vocab_size).detach()).sum(dim=-1)
        loss = loss.view(batch_size, -1)
        if penalty is not None:
            loss = loss * penalty
        loss = loss / norm
        if self.ignore_index is not None:
            return loss.masked_fill(mask, 0.).sum(dim=-1).mean()
        else:
            return loss.sum(dim=-1).mean()


class WarmUpOpt:

    def __init__(self, optimizer, d_model, warmup_steps, 
                 min_learning_rate, factor=1, state_dict=None):
        self.step_num = 0
        self.factor = factor
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.min_learning_rate = min_learning_rate
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def updateRate(self):
        return self.factor * (self.d_model**(-0.5) * 
               min(self.step_num**(-0.5), self.step_num * self.warmup_steps**(-1.5)))

    def step(self):
        self.step_num += 1
        for param in self.optimizer.param_groups:
            param['lr'] = max(self.updateRate(), self.min_learning_rate)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            'optim_state_dict': self.optimizer.state_dict(),
            'step': self.step_num
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(
            state_dict['optim_state_dict']
        )
        self.step_num = state_dict['step']
