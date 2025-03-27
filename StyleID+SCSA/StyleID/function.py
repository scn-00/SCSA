import torch

global opt
opt = None

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output


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
    

    
    

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
    return output        
    
    
    
    
    
    