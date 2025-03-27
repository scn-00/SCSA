from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pickle
import os
from ldm.util import mean_variance_norm
from ldm.modules.diffusionmodules.util import checkpoint
from function import local_adain
import function
def exists(val):
    return val is not None
from function import local_adain

def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.attn = None
        self.q = None
        self.k = None
        self.v = None
    def SCA(self, x=None, context=None, mask=None, q_injected=None, k_injected=None, v_injected=None,x_injected=None,
                          injection_config=None, injection_maps = None, norm=None):
        
        a1 = injection_config['a1']
        attn_matrix_scale = injection_config['T']
     
        self.attn = None
        h = self.heads
        b = x_injected[0].shape[0]
        

      
        q_uncond = q_injected
        q_in = torch.cat([q_uncond]*b)
        q_ = self.to_q(norm(x_injected[0]))
        q_ = rearrange(q_, 'b n (h d) -> (b h) n d', h=h)
        q = q_in * a1 + q_ * (1. - a1)
 
        

        k_uncond = k_injected
        k_in = torch.cat([k_uncond]*b ,dim=0)
        k_ = self.to_k(norm(x_injected[1]))
        k_ = rearrange(k_, 'b n (h d) -> (b h) n d', h=h)
        k = k_in * a1 + k_ * (1. - a1)
         
    
        v_uncond = v_injected
        v = torch.cat([v_uncond]*b ,dim=0)     
        
        self.q = q
        self.k = k
        self.v = v
        sim = einsum('b i d, b j d -> b i j', q, k)
        
        sim *= attn_matrix_scale
        sim *= self.scale

        max_neg_value = -torch.finfo(sim.dtype).max
        if q_injected.size()[1]==1024:
            map = injection_maps[1]
            map = map.repeat(b, 1, 1)
            sim.masked_fill_(map<0.5, max_neg_value)
        if q_injected.size()[1]==4096:
            map = injection_maps[0]
            map = map.repeat(b, 1, 1)
            sim.masked_fill_(map<0.5, max_neg_value)
        
        attn = sim.softmax(dim=-1)
        self.attn = attn
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        return self.to_out(out)
    
    def SSA(self, x, context=None, mask=None, q_injected=None, k_injected=None, v_injected=None,x_injected=None,
                          injection_config=None, injection_maps = None, x_local = None, norm=None):
        
        a2 = injection_config['a2']
        attn_matrix_scale = injection_config['T']
        self.attn = None
        h = self.heads
        
        b = x_injected[0].shape[0]
        # ----获得q
        
        q_uncond = q_injected
        q_in = torch.cat([q_uncond]*b)
    
        q_ = self.to_q(mean_variance_norm(x_local))
        q_ = rearrange(q_, 'b n (h d) -> (b h) n d', h=h)
        q = q_in * a2 + q_ * (1. - a2)
      

        
   
        k_uncond = k_injected
        k = torch.cat([k_uncond]*b ,dim=0)    
         
     
        v_uncond = v_injected
        v = torch.cat([v_uncond]*b ,dim=0)     
        
        self.q = q
        self.k = k
        self.v = v
        sim = einsum('b i d, b j d -> b i j', q, k)
        
        sim *= attn_matrix_scale
        sim *= self.scale
        
        max_neg_value = -torch.finfo(sim.dtype).max
        if q_injected.size()[1]==1024:
            map = injection_maps[1]
            map = map.repeat(b, 1, 1)
            sim.masked_fill_(map<0.5, max_neg_value)
        if q_injected.size()[1]==4096:
            map = injection_maps[0]
            map = map.repeat(b, 1, 1)
            sim.masked_fill_(map<0.5, max_neg_value)
        
        max_indices = torch.argmax(sim, dim=2, keepdim=True)
        B = torch.full_like(sim, max_neg_value)
        B.scatter_(2, max_indices, sim.gather(2, max_indices))
        sim = B
        
        attn = sim.softmax(dim=-1)
        self.attn = attn
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        return self.to_out(out)
            
    def forward(self,
                x, context=None, mask=None, q_injected=None, k_injected=None, v_injected=None,x_injected=None,
                injection_config=None, injection_maps = None, inject_SCA=False, inject_SSA=False, x_local=None, norm=None):
        self.attn = None
        h = self.heads
        b = x.shape[0]
        attn_matrix_scale = 1.0
        q_mix = 0.
        if q_injected is not None:
            if inject_SCA:
                x_injected[0] = self.get_local_adain_q(x_injected[0], x_injected[1])
                return self.SCA(None, context, mask, q_injected[0], k_injected[0], v_injected, x_injected,
                            injection_config, injection_maps, norm = norm)
            if inject_SSA:
                x_injected[0] = self.get_local_adain_q(x_injected[0], x_injected[1])
                q = self.to_q(norm(x_injected[0]))
                q_injected[1] = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
                return self.SSA(x, context, mask, q_injected[1], k_injected[1], v_injected, x_injected,
                            injection_config, injection_maps, x_local, norm)
            
        else:

            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

            context = default(context, x)

            k = self.to_k(context)
            k = rearrange(k, 'b m (h d) -> (b h) m d', h=h)


            v = self.to_v(context)
            v = rearrange(v, 'b m (h d) -> (b h) m d', h=h)


            self.q = q
            self.k = k
            self.v = v

            sim = einsum('b i d, b j d -> b i j', q, k)
            
            sim *= self.scale

            attn = sim.softmax(dim=-1)
            self.attn = attn
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            return self.to_out(out)
        
    def get_local_adain_q(self,q,k,h=None):
     
        q = q.permute(0, 2, 1)
        b,c,hw = q.size()
        w = int(hw ** 0.5)
        q = q.view(b, c, w, w)
        

        k = k.permute(0, 2, 1)
        b,c,hw = k.size()
        w = int(hw ** 0.5)
        k = k.view(b, c, w, w)   
        
        q = local_adain(q, k, None, k.device)
        
        
        k = k.view(b, c, hw).permute(0, 2, 1)
        q = q.view(b, c, hw).permute(0, 2, 1)
        

        return q


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
        
    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                self_attn_x_injected=None,
                injection_config=None,
                injection_maps = None,
                ):
    
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_q_injected,
                                          self_attn_k_injected,
                                          self_attn_v_injected,
                                          self_attn_x_injected,
                                          injection_config,injection_maps), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_q_injected=None,
                 self_attn_k_injected=None,
                 self_attn_v_injected=None,
                 self_attn_x_injected=None,
                 injection_config=None,
                 injection_maps = None):
      
     
        x_SCA = self.attn1(self.norm1(x),
                       q_injected=self_attn_q_injected,
                       k_injected=self_attn_k_injected,
                       v_injected=self_attn_v_injected,
                       x_injected=self_attn_x_injected,
                       injection_config=injection_config,
                       injection_maps=injection_maps, inject_SCA=True, norm = self.norm1)
        
        
        
        x_SSA = self.attn1(self.norm1(x),
                q_injected=self_attn_q_injected,
                k_injected=self_attn_k_injected,
                v_injected=self_attn_v_injected,
                x_injected=self_attn_x_injected,
                injection_config=injection_config,
                injection_maps=injection_maps, inject_SSA=True, x_local=x, norm = self.norm1)
  
       
        if injection_config is not None:
            x = injection_config['t1'] * x_SCA +  injection_config['t2'] * x_SSA + x 

        else:
            x = 0.75 * x_SCA +  0.25 * x_SSA + x
      
        x = self.attn2(x=self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def get_local_adain_q(self,q,k,h=None):
   
        q = q.permute(0, 2, 1)
        b,c,hw = q.size()
        w = int(hw ** 0.5)
        q = q.view(b, c, w, w)
        
  
        k = k.permute(0, 2, 1)
        b,c,hw = k.size()
        w = int(hw ** 0.5)
        k = k.view(b, c, w, w)   
        
        q = local_adain(q, k, None, k.device)
        
        
        k = k.view(b, c, hw).permute(0, 2, 1)
        q = q.view(b, c, hw).permute(0, 2, 1)
        

        return q


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                self_attn_x_injected=None,
                injection_config=None,
                injection_maps = None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
       
        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_q_injected=self_attn_q_injected,
                      self_attn_k_injected=self_attn_k_injected,
                      self_attn_v_injected=self_attn_v_injected,
                      self_attn_x_injected=self_attn_x_injected,
                      injection_config=injection_config,
                      injection_maps = injection_maps)

            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in