import torch
from torch import _VF
from typing import Callable, List, Optional, Tuple, Union
Tensor = torch.Tensor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    DType = int
import warnings
from torch._C import _infer_size, _add_docstr
import importlib
def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):


    is_batched = True

    return is_batched


def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> Optional[Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask

def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


from torch.nn.functional import linear
from torch.nn.functional import scaled_dot_product_attention
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:

    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def pad(input: Tensor, pad: List[int], mode: str = "constant", value: Optional[float] = None) -> Tensor:

    if not torch.jit.is_scripting():
        if torch.are_deterministic_algorithms_enabled() and input.is_cuda:
            if mode == 'replicate':
                # Use slow decomp whose backward will be in terms of index_put.
                # importlib is required because the import cannot be top level
                # (cycle) and cannot be nested (TS doesn't support)
                return importlib.import_module('torch._decomp.decompositions')._replication_pad(
                    input, pad
                )
    return torch._C._nn.pad(input, pad, mode, value)


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[DType] = None) -> Tensor:

    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:

    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)

def local_adain(cnt_z_enc, sty_z_enc, opt, device, number):

    cnt_z_enc = change_size_1(cnt_z_enc)
    sty_z_enc = change_size_1(sty_z_enc)
    # 内容和风格图像的掩码名字
    path = '/home/chunnanshang/chunnanshang/styletransfer/datasets/sem/sem_precomputed_feats/'
    c_masks_name = opt.content.split('/')[-1].split('.')[0] + '_masks_' + str(number) + '.pt'
    s_masks_name = opt.style.split('/')[-1].split('.')[0] + '_masks_' + str(number) + '.pt'
    number1 = opt.content.split('/')[-1].split('.')[0].split('_')[0]
    number2 = opt.style.split('/')[-1].split('.')[0].split('_')[0]
    # 内容和风格图像的掩码加载
    c_masks = torch.load(path + number1 + '/' + c_masks_name, weights_only=True).to(device)
    s_masks = torch.load(path + number2 + '/' + s_masks_name, weights_only=True).to(device)
    n_color = c_masks.size()[0]
    result = torch.zeros(cnt_z_enc.size()).to(device)
    for i in range(n_color):
        result = result + calc_local_adain(c_masks[i], s_masks[i], cnt_z_enc, sty_z_enc)
    result = change_size_2(result)
    return result
    

def calc_local_mean_std(s_mask, sty_z_enc):
    b, c, h, w = sty_z_enc.size()
    s_number = s_mask.sum(dim=-1).sum(dim=-1)
    s_total = sty_z_enc.sum(dim=-1).sum(dim=-1)
    s_mean = (s_total / s_number).view(b, c, 1, 1)
    s_var = torch.pow((sty_z_enc - s_mean), 2) * s_mask
    s_std = (s_var.sum(dim=-1).sum(dim=-1) / s_number + 1e-5).sqrt().view(b, c, 1, 1)
    return s_mean, s_std
    


def calc_local_adain(c_mask, s_mask, cnt_z_enc, sty_z_enc):
    cnt_z_enc = cnt_z_enc * c_mask
    sty_z_enc = sty_z_enc * s_mask
    s_mean, s_std = calc_local_mean_std(s_mask, sty_z_enc)
    c_mean, c_std = calc_local_mean_std(c_mask, cnt_z_enc)
    c_norm = (cnt_z_enc - c_mean) / c_std
    cs = (c_norm * s_std + s_mean) * c_mask
    return cs

def change_size_1(cnt_z_enc):
    hw,b,c = cnt_z_enc.size()
    cnt_z_enc = cnt_z_enc.permute(1,2,0)
    h = int(hw**0.5)
    cnt_z_enc = cnt_z_enc.reshape(b,c,h,h)
    return cnt_z_enc

def change_size_2(cnt_z_enc):
    b,c,h,w = cnt_z_enc.size()
    cnt_z_enc = cnt_z_enc.permute(2,3,0,1)
    hw = int(h*w)
    cnt_z_enc = cnt_z_enc.reshape(hw,b,c)
    return cnt_z_enc


def calc_mean_std(cnt_feat,eps=1e-5):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    output = (cnt_feat-cnt_mean)/(cnt_std+eps)
    return output


def norm(cnt_z_enc):
    cnt_z_enc = calc_mean_std(change_size_1(cnt_z_enc))
    cnt_z_enc = change_size_2(cnt_z_enc)
    return cnt_z_enc
    