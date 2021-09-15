import  torch
from    torch import nn
from    torch.nn import functional as F
from    .Module import Linear, SublayerConnection
from    copy import deepcopy
from    math import inf


class gru(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_z = Linear(2 * d_model, d_model, bias=True)
        self.w_r = Linear(2 * d_model, d_model, bias=True)
        self.w_h = Linear(2 * d_model, d_model, bias=True)

    def forward(self, x, h):
        z_t = torch.sigmoid(self.w_z(torch.cat((x, h), dim=-1)))
        r_t = torch.sigmoid(self.w_r(torch.cat((x, h), dim=-1)))
        h_hat = torch.tanh(self.w_h(torch.cat((r_t * h, x), dim=-1)))
        return (1 - z_t) * h + z_t * h_hat


class graphLayer(nn.Module):
    def __init__(self, d_model, num_head, FFN, dropout, normalize_before):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** 0.5
        self.num_head = num_head
        # self.FFN = deepcopy(FFN)
        # self.sublayer = SublayerConnection(d_model, dropout, normalize_before)
        self.head_dim = d_model // num_head
        self.w_head = Linear(d_model, d_model)
        self.w_tail = Linear(d_model, d_model)
        self.w_info = Linear(d_model, d_model)
        self.w_comb = Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)
        # self.gru = gru(d_model)
        
        # self.w_z = Linear(2 * d_model, d_model)
        # self.w_r = Linear(2 * d_model, d_model)
        # self.w_h = Linear(2 * d_model, d_model)

    def forward(self, inputs, memory, graph, gru):
        '''
        inputs: [batch_size, num_seq, hidden_dim]
        graph: [batch_size, num_seq, num_seq]
        '''
        batch_size = inputs.size(0)
        num_seq = inputs.size(1)
        head = self.w_head(inputs).view(batch_size, num_seq, self.num_head, -1)
        tail = self.w_tail(memory).view(batch_size, num_seq, self.num_head, -1)
        info = self.w_info(memory).view(batch_size, num_seq, self.num_head, -1)
        head.transpose_(-2, -3).contiguous()
        tail.transpose_(-2, -3).contiguous()
        info.transpose_(-2, -3).contiguous()
        score = torch.matmul(head, tail.transpose_(-1, -2).contiguous()) / self.scale
        graph = (graph.unsqueeze(1) == 0)
        score.masked_fill_(graph, -inf)
        attn_weight = F.softmax(score, dim=-1)
        attn_weight = attn_weight.masked_fill(graph, 0.)
        attn_vector = torch.matmul(attn_weight, info)
        attn_vector.transpose_(-2, -3)
        attn_vector = self.w_comb(attn_vector.contiguous().view(batch_size, num_seq, -1))
        inputs = inputs.view(batch_size * num_seq, 1, -1)
        attn_vector = attn_vector.view(1, batch_size * num_seq, -1)
        _, memory = gru(inputs, attn_vector)
        memory = self.norm(self.dropout(memory.view(batch_size, num_seq, -1)))
        # z_t = torch.sigmoid(self.w_z(torch.cat((inputs, attn_vector), dim=-1)))
        # r_t = torch.sigmoid(self.w_r(torch.cat((inputs, attn_vector), dim=-1)))
        # h_hat = torch.tanh(self.w_h(torch.cat((r_t * attn_vector, inputs), dim=-1)))
        # inputs = self.norm(self.dropout((1 - z_t) * attn_vector + z_t * h_hat))
        # inputs = self.norm(self.dropout(self.gru(inputs, attn_vector)))
        # inputs = self.norm(inputs + self.dropout(attn_vector))
        return memory


class GAT(nn.Module):
    def __init__(self, d_model, num_head, num_layer, dropout):
        super().__init__()
        self.num_layer = num_layer
        self.gru = nn.GRU(input_size=d_model,
                          hidden_size=d_model,
                          num_layers=1,
                          batch_first=True)
        self.d_model = d_model
        self.scale = d_model ** 0.5
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.w_head = Linear(d_model, d_model)
        self.w_tail = Linear(d_model, d_model)
        self.w_info = Linear(d_model, d_model)
        self.w_comb = Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)
    
    def forward(self, inputs, graph):
        batch_size = inputs.size(0)
        num_seq = inputs.size(1)
        _, memory = self.gru(inputs.view(batch_size * num_seq, 1, -1))
        memory = memory.view(batch_size, num_seq, -1)
        memory = self.norm(self.dropout(memory))
        graph = (graph.unsqueeze(1) == 0)
        for _ in range(self.num_layer):
            # inputs = inputs.view(batch_size, num_seq, -1)
            head = self.w_head(inputs).view(batch_size, num_seq, self.num_head, -1)
            tail = self.w_tail(memory).view(batch_size, num_seq, self.num_head, -1)
            info = self.w_info(memory).view(batch_size, num_seq, self.num_head, -1)
            head.transpose_(-2, -3).contiguous()
            tail.transpose_(-2, -3).contiguous()
            info.transpose_(-2, -3).contiguous()
            score = torch.matmul(head, tail.transpose_(-1, -2).contiguous()) / self.scale
            score.masked_fill_(graph, -inf)
            attn_weight = F.softmax(score, dim=-1)
            attn_weight = attn_weight.masked_fill(graph, 0.)
            attn_vector = torch.matmul(attn_weight, info)
            attn_vector.transpose_(-2, -3)
            attn_vector = self.w_comb(attn_vector.contiguous().view(batch_size, num_seq, -1))
            attn_vector = attn_vector.view(1, batch_size * num_seq, -1)
            _, memory = self.gru(inputs.view(batch_size * num_seq, 1, -1), attn_vector)
            memory = self.norm(self.dropout(memory.view(batch_size, num_seq, -1)))

        return memory
