import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
Tensor = torch.Tensor
from other import * 
from other import _mha_shape_check, _canonical_mask, _none_or_dtype, _in_projection_packed
import math
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        
        
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        
        # 输出投影矩阵
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, args, query, key, value, query_sem, key_sem, is_scsa):
   
       
        attn_output, attn_output_weights = self.multi_head_attention_forward(
                args, query, key, value, query_sem, key_sem, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.out_proj.weight, self.out_proj.bias, is_scsa
                )
     
        return attn_output, attn_output_weights
    
    def multi_head_attention_forward(self,
        args,
        query,
        key,
        value,
        query_sem,
        key_sem,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        out_proj_weight,
        out_proj_bias,
        is_scsa,
    ):

        if is_scsa:
            map_64 = torch.load(args.sem_map_64,weights_only=True).to(query.device)
            map_32 = torch.load(args.sem_map_32,weights_only=True).to(query.device)
            x_SCA = self.SCA(query, key, value, query_sem, key_sem, map_32, map_64,
                                 num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)
            
            x_SSA = self.SSA(query, key, value, query_sem, key_sem, map_32, map_64,
                                 num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)
            t1 = args.t1
            t2 = args.t2
            return t1*x_SCA + t2* x_SSA, None
        
        else:
            tgt_len, bsz, embed_dim = query.shape
            src_len, _, _ = key.shape
            head_dim = embed_dim // num_heads
    
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
            q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            # print(q.size())  # [8, 4096, 64]

            src_len = k.size(1)
            B, Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

        

            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1)) # [8, 4096, 4096]
        
            attn_output_weights = softmax(attn_output_weights, dim=-1)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.mean(dim=1)

            return attn_output, attn_output_weights


    def SCA(self, content_q, style_k, style_v, content_sem_q, style_sem_k, map_32, map_64,
                                 num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
        tgt_len, bsz, embed_dim = content_sem_q.shape
        src_len, _, _ = style_sem_k.shape
        head_dim = embed_dim // num_heads
 
        q, k, v = _in_projection_packed(content_sem_q, style_sem_k, style_v, in_proj_weight, in_proj_bias)
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)
        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))
        
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        max_neg_value = -torch.finfo(attn_output_weights.dtype).max
        if q.size()[1]==1024:
            map = map_32
            map = map.repeat(B, 1, 1)
            attn_output_weights.masked_fill_(map<0.5, max_neg_value)
        if q.size()[1]==4096:
            map = map_64
            map = map.repeat(B, 1, 1)
            attn_output_weights.masked_fill_(map<0.5, max_neg_value)
        
        
        attn_output_weights = softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)
        
        
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        
        return attn_output
        


    
    def SSA(self, content_q, style_k, style_v, content_sem_q, style_sem_k, map_32, map_64,
                                num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
        
        tgt_len, bsz, embed_dim = content_q.shape
        src_len, _, _ = style_k.shape
        head_dim = embed_dim // num_heads
 
        q, k, v = _in_projection_packed(content_q, style_k, style_v, in_proj_weight, in_proj_bias)
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        
        src_len = k.size(1)
        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))
        
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        max_neg_value = -torch.finfo(attn_output_weights.dtype).max
        
        
        if q.size()[1]==1024:
            map = map_32
            map = map.repeat(B, 1, 1)
            attn_output_weights.masked_fill_(map<0.5, max_neg_value)
        if q.size()[1]==4096:
            map = map_64
            map = map.repeat(B, 1, 1)
            attn_output_weights.masked_fill_(map<0.5, max_neg_value)
            
            
        max_indices = torch.argmax(attn_output_weights, dim=2, keepdim=True)
        B = torch.full_like(attn_output_weights, max_neg_value)
        B.scatter_(2, max_indices, attn_output_weights.gather(2, max_indices))
        attn_output_weights = B    
        
        attn_output_weights = softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)
        
        
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        
        return attn_output     
