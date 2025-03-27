import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
from torchsummary import summary
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import sys
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle
from function import local_adain
feat_maps = []

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, cnt_feats_sem, sty_feats_sem, start_step, sem_map_64, sem_map_32):
    feat_maps = [{'config': {
                'T':opt.T,
                't1': opt.t1, # sca inject
                't2':opt.t2, # ssa inject 
                'a1':opt.a1, # sca 
                'a2':opt.a2, # ssa 
                'sem_maps':[sem_map_64, sem_map_32],
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        cnt_feat_sem = cnt_feats_sem[i]
        sty_feat_sem = sty_feats_sem[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = [cnt_feat_sem[ori_key], cnt_feat[ori_key]]
            if ori_key[-1] == 'k': 
                feat_maps[i][ori_key] = [sty_feat_sem[ori_key], sty_feat[ori_key]]
            if ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
            if ori_key[-1] == 'x':
                feat_maps[i][ori_key] = [cnt_feat[ori_key], sty_feat[ori_key]]   
    return feat_maps


def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    # image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu",weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)


    model.cuda()
    model.eval()
    return model


def get_feature(sty_name_, sty_name, device, feat_path_root, model, sampler, ddim_inversion_steps, time_idx_dict, save_feature_timesteps, ddim_sampler_callback, uc, start_step):
    
    init_sty = load_img(sty_name_).to(device)
    seed = -1
    number = sty_name_.split('/')[-2]
    sty_feat_name = os.path.join(feat_path_root+ '/' + number, os.path.basename(sty_name).split('.')[0] + '.pkl')
    sty_z_enc = None

    if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
        print("Precomputed feature loading: ", sty_feat_name)
        with open(sty_feat_name, 'rb') as h:
            sty_feat = pickle.load(h)
            sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
    else:
        print("There are no precomputed features. Please run StyleID/get_pre_features.py")
        sys.exit()

    return sty_z_enc, sty_feat, sty_feat_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = '../../sem_data/1/1.jpg')
    parser.add_argument('--sty', default = '../../sem_data/1/1_paint.jpg')
    parser.add_argument('--cnt_sem', default = '../../sem_data/1/1_sem.png')
    parser.add_argument('--sty_sem', default = '../../sem_data/1/1_paint_sem.png')
    parser.add_argument('--sem_map_64', default = '../../sem_precomputed_feats/1/1_1_paint_map_64.pt')
    parser.add_argument('--sem_map_32', default = '../../sem_precomputed_feats/1/1_1_paint_map_32.pt')
    parser.add_argument('--output_path', type=str, default='output_SCSA')
    parser.add_argument('--precomputed', type=str, default='../../sem_precomputed_feats', help='save path for precomputed feature')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.2, help='attention temperature scaling hyperparameter')
    parser.add_argument('--t1', type=float, default=0.8, help='sca的权重,全局和局部的权重')
    parser.add_argument('--t2', type=float, default=0.2, help='ssa的权重,全局和局部的权重')
    parser.add_argument('--a1', type=float, default=0.3, help='sca中inject的比例')
    parser.add_argument('--a2', type=float, default=0.5, help='ssa中inject的比例')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='../StyleID/models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='../models/sd-v1-4.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    opt = parser.parse_args()

    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed
    sem_map_64 = torch.load(opt.sem_map_64, weights_only=True).to(device)
    sem_map_32 = torch.load(opt.sem_map_32, weights_only=True).to(device)
    
    global feat_maps
    


    feat_maps = [{'config': {
                'sem_maps':[sem_map_64, sem_map_32],
                'T':opt.T,
                't1': opt.t1, 
                't2':opt.t2, 
                'a1':opt.a1, 
                'a2':opt.a2,
                }} for _ in range(50)]


    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    
    sty_z_enc, sty_feat, sty_feat_name = get_feature(opt.sty, opt.sty.split('/')[-1], device, feat_path_root, model, sampler, ddim_inversion_steps, time_idx_dict, save_feature_timesteps, None, uc, start_step)

    cnt_z_enc, cnt_feat, cnt_feat_name = get_feature(opt.cnt, opt.cnt.split('/')[-1], device, feat_path_root, model, sampler, ddim_inversion_steps, time_idx_dict, save_feature_timesteps, None, uc, start_step)

    cnt_z_enc_sem, cnt_feat_sem, cnt_feat_name_sem = get_feature(opt.cnt_sem, opt.cnt_sem.split('/')[-1], device, feat_path_root, model, sampler, ddim_inversion_steps, time_idx_dict, save_feature_timesteps, None, uc, start_step)       

    sty_z_enc_sem, sty_feat_sem, sty_feat_name_sem = get_feature(opt.sty_sem, opt.sty_sem.split('/')[-1], device, feat_path_root, model, sampler, ddim_inversion_steps, time_idx_dict, save_feature_timesteps, None, uc, start_step)       
    
    
    


    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():


                output_name = opt.cnt.split('/')[-1].split('.')[0] + '_' + opt.sty.split('/')[-1].split('.')[0] + "_StyleID_SCSA" + ".jpg"

        
        
                local_adain_z_enc = local_adain(cnt_z_enc, sty_z_enc, opt, device)
                feat_maps = feat_merge(opt, cnt_feat, sty_feat, cnt_feat_sem, sty_feat_sem, start_step, sem_map_64, sem_map_32)




   
                samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                batch_size=1,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=local_adain_z_enc,
                                                injected_features=feat_maps,
                                                start_step=start_step,
                                                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
             
                img = Image.fromarray(x_sample.astype(np.uint8))

                img.save(os.path.join(output_path, output_name))
            



if __name__ == "__main__":
    main()
