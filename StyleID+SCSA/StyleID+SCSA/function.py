import torch

global opt
opt = None


def local_adain(cnt_z_enc, sty_z_enc, args, device):
    global opt
    if args is not None:
        opt = args
    # 内容和风格图像的掩码名字
    path = '../../sem_precomputed_feats/'
    number = opt.cnt.split('/')[-2]
    b, c, h, w = cnt_z_enc.size()
    
    c_masks_name = opt.cnt.split('/')[-1].split('.')[0] + '_masks_'+ str(h) +'.pt'
    s_masks_name = opt.sty.split('/')[-1].split('.')[0] + '_masks_'+ str(h) +'.pt'
    # 内容和风格图像的掩码加载
    c_masks = torch.load(path + '/' + number + '/' + c_masks_name).to(device)
    s_masks = torch.load(path + '/' + number + '/' + s_masks_name).to(device)
    n_color = c_masks.size()[0]
    result = torch.zeros(cnt_z_enc.size()).to(device)
  
    for i in range(n_color):
        result = result + calc_local_adain(c_masks[i], s_masks[i], cnt_z_enc, sty_z_enc)
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
    

    
    

    

    
    
    
    
    