import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from math import sqrt

class Encoder(nn.Module):
    def __init__(
            self,
            ninp=8,#=self.state_length=8
            ntimestep=9,
            in_dim=128,#256
            nhead=2,#8
            fdim=256,#512
            nlayers=6,
            noutput=32,
            dropout=0.2,
            low_dim="sum"):
        super().__init__()

        self.model_type = 'Transformer'
        self.input_fc = nn.Linear(ninp, in_dim)
        self.pos_embedding = PositionalEncoding(in_dim, dropout)
        self.value_embedding = TokenEmbedding(in_dim, in_dim)
        
        # encoder_layers = TransformerEncoderLayer(
        #     in_dim, nhead, fdim, dropout, batch_first=True)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        e_layers=[6,5,3,2,1]
        distil=True
        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here        
        encoders = [
            InformerEncoder(
                [
                    EncoderLayer(
                        AttentionLayer(FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False), 
                                    d_model=256, n_heads=8, mix=False),
                        d_model=256,
                        d_ff=256,
                        dropout=0.1,
                        activation='gelu'
                    ) for l in range(el)
                ],
                [
                    ConvLayer(256) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(256)
            )for el in e_layers]
        
        self.encoder = EncoderStack(encoders, inp_lens)

        self.output_fc = nn.Linear(in_dim, noutput)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, src, robot, src_mask, src_key_padding_mask, robot_padding_mask, robot_src_mask):
        # print(src_mask)
        # print(src_key_padding_mask)
        input = torch.nan_to_num(src)
        input = self.input_fc(input)
        input = self.pos_embedding(input) + self.value_embedding(input) + input
        memory, attns = self.encoder(input, attn_mask=src_mask)
        if robot != None:
            robot = torch.nan_to_num(robot)
            robot = self.input_fc(robot)
            robot = self.pos_embedding(robot) + self.value_embedding(robot) + robot
            robot, attns = self.encoder(robot, attn_mask=robot_src_mask)
            robot = self.output_fc(robot)
            robot = robot.permute(0,2,1)
            robot = self.avg_pool(robot) # (B,C,L) 通过avg=》 (B,C,1)
            robot = robot.permute(0,2,1)
        
        # input = torch.nan_to_num(src)
        # input = self.input_fc(input)
        # input = self.pos_embedding(input)
        # memory = self.transformer_encoder(input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # if robot != None:
        #     robot = torch.nan_to_num(robot)
        #     robot = self.input_fc(robot)
        #     robot = self.pos_embedding(robot)
        # robot = self.transformer_encoder(robot, mask=robot_src_mask, src_key_padding_mask=robot_padding_mask)
        # robot = self.output_fc(robot)
        # robot = robot.permute(0,2,1)
        # robot = self.avg_pool(robot) # (B,C,L) 通过avg=》 (B,C,1)
        # robot = robot.permute(0,2,1)

        return memory, robot

class Edge_encoder(nn.Module):
    def __init__(
            self,
            ninp=14,#=self.state_length=8
            ntimestep=9,
            in_dim=128,#256
            nhead=2,#8
            fdim=256,#512
            nlayers=6,
            noutput=32,
            dropout=0.2,
            low_dim="sum"):
        super().__init__()

        self.model_type = 'Transformer'
        self.input_fc = nn.Linear(ninp, in_dim)
        self.pos_embedding = PositionalEncoding(in_dim, dropout)
        self.value_embedding = TokenEmbedding(in_dim, in_dim)
        
        # encoder_layers = TransformerEncoderLayer(
        #     in_dim, nhead, fdim, dropout, batch_first=True)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
       
        e_layers=[6,5,3,2,1]
        distil=True
        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here        
        encoders = [
            InformerEncoder(
                [
                    EncoderLayer(
                        AttentionLayer(FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False), 
                                    d_model=256, n_heads=8, mix=False),
                        d_model=256,
                        d_ff=256,
                        dropout=0.1,
                        activation='gelu'
                    ) for l in range(el)
                ],
                [
                    ConvLayer(256) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(256)
            )for el in e_layers]
        
        self.encoder = EncoderStack(encoders, inp_lens)

        self.output_fc = nn.Linear(in_dim, noutput)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, src, src_mask, src_key_padding_mask):
        input = torch.nan_to_num(src)
        input = self.input_fc(input)
        input = self.pos_embedding(input) + self.value_embedding(input) + input
        memory, attns = self.encoder(input, attn_mask=src_mask)
        memory = self.output_fc(memory)
        memory = memory.permute(0,2,1)
        b, c, l = memory.size() # (B,C,L)
        memory = self.avg_pool(memory) # (B,C,L) 通过avg=》 (B,C,1)
        memory = memory.reshape(b, c) 

        # input = torch.nan_to_num(src)
        # input = self.input_fc(input)
        # input = self.pos_embedding(input)
        # memory = self.transformer_encoder(input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # memory = self.output_fc(memory)
        # memory = memory.permute(0,2,1)
        # b, c, l = memory.size() # (B,C,L)
        # memory = self.avg_pool(memory) # (B,C,L) 通过avg=》 (B,C,1)
        # memory = memory.reshape(b, c) 

        return memory

class Decoder(nn.Module):
    def __init__(
            self,
            nlayers,
            ninp,
            in_dim,
            nhead,
            fdim,
            noutput,
            dropout=0.2):
        super().__init__()

        self.model_type = 'Transformer'
        self.input_fc = nn.Linear(ninp, in_dim)
        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        self.pos_embedding = PositionalEncoding(in_dim, dropout)
        self.value_embedding = TokenEmbedding(in_dim, in_dim)
        # d_layers=1      
        # self.decoder = InformerDecoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(FullAttention(True, factor=5, attention_dropout=0.1, output_attention=False), 
        #                         d_model=256, n_heads=8, mix=True),
        #             AttentionLayer(FullAttention(False, factor=5, attention_dropout=0.1, output_attention=False), 
        #                         d_model=256, n_heads=8, mix=False),
        #             d_model=256,
        #             d_ff=256,
        #             dropout=0.1,
        #             activation='gelu',
        #         )
        #         for l in range(d_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(256)
        # )
        # self.projection = nn.Linear(256, noutput, bias=True) 
        self.tcs_layer = Times_Channel_Squeeze()
        
        nlayers = 1
        decoder_layers = TransformerDecoderLayer(
           in_dim, nhead, fdim, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.output_fc = nn.Linear(in_dim, noutput)
        # self.init_weights()

    # def init_weights(self) -> None:
    #     # self.encoder.weight.data.uniform_(-initrange, initrange)
    #     torch.nn.init.zeros_(self.input_fc.bias)
    #     torch.nn.init.xavier_normal_(self.input_fc.weight)
    #     torch.nn.init.zeros_(self.output_fc.bias)
    #     torch.nn.init.xavier_normal_(self.output_fc.weight)

    @abc.abstractmethod
    def forward(
            self,
            prediction_horizon,            
            tgt,#
            memory,#128, 9, 256]
            tgt_mask,#[1, 1]
            memory_mask,#[9, 9]
            memory_key_padding_mask,#[128, 9]
            ):#[128, 9]
        # input tgt => target last time state

        # tgt = torch.nan_to_num(tgt)
        # dec_emb = self.pos_embedding(self.input_fc(tgt)) #+ self.value_embedding(self.input_fc(tgt)) + self.input_fc(tgt)
        # memory = self.tcs_layer(memory)        
        # output = self.transformer_decoder(
        #    dec_emb,
        #    memory,
        #    tgt_mask=tgt_mask,
        #    memory_mask=memory_mask,
        #    memory_key_padding_mask=memory_key_padding_mask)
        # output = self.output_fc(output)
        
        # tgt = torch.nan_to_num(tgt)
        # tgt = self.input_fc(tgt)
        # dec_emb = self.pos_embedding(tgt) + self.value_embedding(tgt) + tgt
        # #memory = self.tcs_layer(memory)
        # dec_self_mask = tgt_mask.bool()
        # # print(dec_self_mask)
        # dec_enc_mask = memory_mask.bool()
        # # print(dec_enc_mask)
        # dec_out = self.decoder(dec_emb, memory, dec_self_mask, dec_enc_mask)
        # output = self.projection(dec_out)#[128, , 32]
        # print('output',output.size())
        return output[:,-prediction_horizon:,:]  

class Trajectory_Decoder(Decoder):
    def __init__(
            self,
            tgt_inp,
            in_dim,
            nhead,
            fdim,
            nlayers,
            noutput):
        super().__init__(nlayers, tgt_inp, in_dim, nhead, fdim, noutput)

    def forward(
            self,
            prediction_horizon,            
            tgt,#128, 12, 2
            memory,#128, 9, 256]
            tgt_mask,#[1, 1]
            memory_mask,#[9, 9]
            memory_key_padding_mask,#[128, 9]
            ):

        # tgt = torch.nan_to_num(tgt)
        # dec_emb = self.pos_embedding(self.input_fc(tgt)) #+ self.value_embedding(self.input_fc(tgt)) + self.input_fc(tgt)
        # memory = self.tcs_layer(memory)        
        # output = self.transformer_decoder(
        #    dec_emb,
        #    memory,
        #    tgt_mask=tgt_mask,
        #    memory_mask=None,
        #    memory_key_padding_mask=memory_key_padding_mask)
        # output = self.output_fc(output)

        tgt = torch.nan_to_num(tgt)
        memory = self.tcs_layer(memory) 
        output = self.transformer_decoder(
           self.pos_encoder(
               self.input_fc(tgt)),
           memory,
           tgt_mask=tgt_mask,
           memory_mask=None,
           memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_fc(output)
        
        # tgt = torch.nan_to_num(tgt)
        # tgt = self.input_fc(tgt)
        # dec_emb = self.pos_embedding(tgt) + self.value_embedding(tgt) + tgt
        # memory = self.tcs_layer(memory)
        # dec_self_mask = tgt_mask.bool()
        # # print(dec_self_mask)
        # dec_enc_mask = memory_mask.bool()
        # # print(dec_enc_mask)
        # dec_out = self.decoder(dec_emb, memory, dec_self_mask, dec_enc_mask)
        # output = self.projection(dec_out)#[128, , 32]
        # # print('output',output.size())_out)#[128, , 32]
        # # print('output',output)
        return output[:,-prediction_horizon:,:]        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (bs,timestep,feature)
        x = x + Variable(self.pe[:x.size()[-2]], requires_grad=False)
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x



################# informer encoder #####################

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        #print('new_x',new_x.size())
        x = x + self.dropout(new_x)
        #print('x1',x.size())
        y = x = self.norm1(x)
        #print('y11',y.size())
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        #print('y12',y.size())
        y = self.dropout(self.conv2(y).transpose(-1,1))
        #print('y13',y.size())

        return self.norm2(x+y), attn


class InformerEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(InformerEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                #print('x1',x.size())
                x = conv_layer(x)
                #print('x2',x.size())
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            #print('x3',x.size())
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
            #print('x4',x.size())
        return x, attns

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        #print('x21',x.size())
        #x = self.norm(x)
        #print('x22',x.size())
        x = self.activation(x)
        #print('x23',x.size())
        x = self.maxPool(x)
        #print('x24',x.size())
        x = x.transpose(1,2)
        #print('x25',x.size())
        return x


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn
    
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
    
class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            #print('x_s',x_s.size())
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        #print('x4',x_stack.size())
        
        return x_stack, attns

################# informer decoder #####################
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class InformerDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(InformerDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
class Times_Channel_Squeeze(nn.Module):
    def __init__(self, in_channels=256, layer_num=1, reduction=32):
        super(Times_Channel_Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.layer_seq = nn.Sequential()
        for i in range(layer_num):  
            self.layer_seq.add_module(f'lmlp_{i}', nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction),
                nn.LayerNorm(in_channels // reduction),
                nn.Dropout(p=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels),
                nn.LayerNorm(in_channels),
                nn.Sigmoid())
            )         

    def forward(self, x):
        y = x.permute(0,2,1)
        b, c, l = y.size() # (B,C,L)
        y = self.avg_pool(y) # (B,C,L) 通过avg=》 (B,C,1)
        #print('y1',y.size()) # [128, 96, 1]
        y = y.view(b, c)
        #print('y2',y.size()) # [128, 96]
        y = self.layer_seq(y)
        #print('y3',y.size()) # [128, 96]
        y = y.unsqueeze(-2)
        #print('y4',y.size()) # [128, 1, 96]
        x = x * y.expand_as(x)
        #print('x',x.size()) # [128, 1, 96]
        return x

