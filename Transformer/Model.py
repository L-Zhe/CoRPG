import  torch
from    torch import nn
from    torch.nn import functional as F
from    .Module import (Encoder, Decoder, Embedding,
                        PositionWiseFeedForwardNetworks,
                        MultiHeadAttention, EncoderCell,
                        DecoderCell, pad_mask, triu_mask,
                        SublayerConnection, Linear, clones)
from    .Graph import GAT, graphLayer
from    copy import deepcopy
from    .SearchStrategy import SearchMethod
from    .Reward import Reward
from    utils.lang import translate2word
from    math import inf


class transformer(nn.Module):
    def __init__(self, config):
            
        d_model  = getattr(config, 'embedding_dim', 512)
        num_head = getattr(config, 'num_head', 8)
        num_layer_encoder = getattr(config, 'num_layer_encoder', 6)
        num_layer_decoder = getattr(config, 'num_layer_decoder', 6)
        num_layer_graph_encoder = getattr(config, 'num_layer_grah_encoder', 6)
        d_ff = getattr(config, 'd_ff', 2048)
        dropout_embed = getattr(config, 'dropout_embed', 0.1)
        dropout_sublayer = getattr(config, 'dropout_sublayer', 0.1)
        BOS_index   = config.BOS_index
        EOS_index   = config.EOS_index
        PAD_index   = config.PAD_index

        super().__init__()
        
        assert d_model % num_head == 0, \
            ("Parameter Error, require embedding_dim % num head == 0.")

        d_qk = d_v = d_model // num_head
        attention = MultiHeadAttention(d_model, d_qk, d_v, num_head)
        FFN = PositionWiseFeedForwardNetworks(d_model, d_model, d_ff)

        vocab_size = getattr(config, 'vocab_size')
        self.src_embed = Embedding(vocab_size, 
                                   d_model, 
                                   dropout=dropout_embed,
                                   padding_idx=PAD_index)
        self.tgt_embed = self.src_embed

        normalize_before = getattr(config, 'normalize_before', False)

        self.sent_Encoder = Encoder(d_model=d_model, 
                                    num_layer=num_layer_encoder,
                                    layer=EncoderCell(d_model=d_model, 
                                                      attn=attention, 
                                                      FFNlayer=FFN, 
                                                      dropout=dropout_sublayer,
                                                      normalize_before=normalize_before),
                                    normalize_before=normalize_before)

        self.graph_Encoder_f = GAT(d_model=d_model,
                                   num_layer=num_layer_graph_encoder,
                                   num_head=num_head,
                                   dropout=dropout_sublayer)

        self.graph_Encoder_r = GAT(d_model=d_model,
                                   num_layer=num_layer_graph_encoder,
                                   num_head=num_head,
                                   dropout=dropout_sublayer)

        # self.sent_pos = nn.Parameter(torch.randn(1, 5, 1, d_model))

        self.w_comb = Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout_sublayer, inplace=True)
        self.norm = nn.LayerNorm(d_model)

        self.Decoder = Decoder(d_model=d_model, 
                               layer=DecoderCell(d_model=d_model,
                                                 attn=attention,
                                                 FFNlayer=FFN,
                                                 dropout=dropout_sublayer,
                                                 normalize_before=normalize_before),
                               num_layer=num_layer_decoder,
                               normalize_before=normalize_before)

        beam = getattr(config, 'beam', 5)
        search_method = getattr(config, 'decode_method', 'greedy')
        self.decode_search = SearchMethod(search_method=search_method, 
                                          BOS_index=BOS_index,
                                          EOS_index=EOS_index,
                                          beam=beam)
        self.PAD_index = PAD_index
        self.BOS_index = BOS_index
        self.EOS_index = EOS_index
        self.project = Linear(d_model, vocab_size)
        self.w_p_g = Linear(d_model, 1)

    def generate(self, embed, encoder_outputs, src_pad_mask, source, memory=None):
        outputs, memory, attn_weight = self.Decoder.generate(embed=embed,
                                                             encoder_outputs=encoder_outputs,
                                                             src_pad_mask=src_pad_mask,
                                                             memory=memory)
        p_g = torch.sigmoid(self.w_p_g(outputs))
        outputs = self.project(outputs)
        prob = F.softmax(outputs, dim=-1)
        prob = self.copy(prob, attn_weight, p_g, source)
        return prob, memory

    def copy(self, prob, attn_weight, p_g, src_index):
        
        src_index = src_index.unsqueeze(1).repeat(1, prob.size(1), 1)
        attn_weight = (1 - p_g) * attn_weight
        prob = prob * p_g
        return prob.scatter_add(2, src_index, attn_weight)

    def forward(self, **kwargs):
        '''
            source: [batch_size, num_seq, seq_len]
            graph: [batch_size, num_seq, num_seq]
            ground_truth: [batch_size, total_seq_len]
        '''
        assert kwargs['mode'] in ['train', 'test']
        source = kwargs['source']
        graph = kwargs['graph']
        batch_size = source.size(0)
        num_seq = source.size(1)
        seq_len = source.size(2)
        source = source.view(batch_size * num_seq, seq_len)
        src_embed = self.src_embed(source)
        src_pad_mask = pad_mask(source, self.PAD_index)
        src_outputs = self.sent_Encoder(src_embed, src_pad_mask)
        src_outputs = src_outputs.view(batch_size, num_seq, seq_len, -1)
        # encoder_outputs = src_outputs.view(batch_size, num_seq * seq_len, -1)
        sent_rep = src_outputs.mean(dim=-2)
        graph_outputs = self.graph_Encoder_f(sent_rep, graph) + self.graph_Encoder_r(sent_rep, graph.transpose_(-1, -2))
        source = source.view(batch_size, num_seq * seq_len)
        encoder_mask = pad_mask(source, self.PAD_index)
        # encoder_outputs = torch.cat((src_outputs, self.sent_pos.repeat(batch_size, 1, seq_len, 1)), dim=-1)
        encoder_outputs = torch.cat((src_outputs, graph_outputs.unsqueeze(2).repeat(1, 1, seq_len, 1)), dim=-1)
        encoder_outputs = self.w_comb(encoder_outputs.view(batch_size, num_seq * seq_len, -1))
        encoder_outputs = self.norm(self.dropout(F.relu(encoder_outputs)))

        if kwargs['mode'] == 'train':
            ground_truth = kwargs['ground_truth']
            tgt_embed = self.tgt_embed(ground_truth)
            tgt_mask = triu_mask(ground_truth.size(1)).to(ground_truth.device) \
                 | pad_mask(ground_truth, self.PAD_index)
            outputs, attn_weight = self.Decoder(inputs=tgt_embed, 
                                                encoder_outputs=encoder_outputs,
                                                pad_mask=encoder_mask,
                                                seq_mask=tgt_mask)
            p_g = torch.sigmoid(self.w_p_g(outputs))
            outputs = self.project(outputs)
            prob = F.softmax(outputs, dim=-1)
            return self.copy(prob, attn_weight, p_g, source)
        else:
            max_length = kwargs['max_length']
            return self.decode_search(decoder=self.generate,
                                      tgt_embed=self.tgt_embed.single_embed, 
                                      src_pad_mask=encoder_mask, 
                                      encoder_outputs=encoder_outputs,
                                      max_length=max_length,
                                      source=source)
