import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import local_adain
import numpy as np

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), # 256
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), # 128
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), # 64
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True), # 32
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class SANet(nn.Module):
    
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
    def SCA(self, content, style, content_sem, style_sem, map_32, map_64):
        F = self.f(mean_variance_norm(content_sem))
        G = self.g(mean_variance_norm(style_sem))
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        max_neg_value = -torch.finfo(S.dtype).max
        if F.size()[1]==1024:
            map = map_32
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        if F.size()[1]==4096:
            map = map_64
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        S = self.sm(S)
        H = self.h(style)
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        return O
    
    def SSA(self, content, style, content_sem, style_sem, map_32, map_64):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        max_neg_value = -torch.finfo(S.dtype).max
        if F.size()[1]==1024:
            map = map_32
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        if F.size()[1]==4096:
            map = map_64
            map = map.repeat(b, 1, 1)
            S.masked_fill_(map<0.5, max_neg_value)
        max_indices = torch.argmax(S, dim=2, keepdim=True)
        B = torch.full_like(S, max_neg_value)
        B.scatter_(2, max_indices, S.gather(2, max_indices))
        S = B        
        S = self.sm(S)
        H = self.h(style)
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        return O
    
    
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    

    def forward(self, content, style, content_sem, style_sem, map_32, map_64, args):
        x_SCA = self.SCA(content, style, content_sem, style_sem, map_32, map_64)
        x_SSA = self.SSA(content, style, content_sem, style_sem, map_32, map_64)
        x = args.t1*x_SCA+args.t2*x_SSA + content
        return x
        


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_sem, style4_1_sem, content5_1_sem, style5_1_sem, map_32, map_64, args):
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1, content4_1_sem, style4_1_sem, map_32, map_64, args) + \
            self.upsample5_1(self.sanet5_1(content5_1, style5_1, content5_1_sem, style5_1_sem, map_32, map_64, args))))

def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512, 512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = '../sem_data/29/29.jpg')
parser.add_argument('--style', type=str, default = '../sem_data/29/29_paint.jpg')
parser.add_argument('--content_sem', type=str, default = '../sem_data/29/29_sem.png')
parser.add_argument('--style_sem', type=str, default = '../sem_data/29/29_paint_sem.png')
parser.add_argument('--sem_map_64', default = '../sem_precomputed_feats/29/29_29_paint_map_64.pt')
parser.add_argument('--sem_map_32', default = '../sem_precomputed_feats/29/29_29_paint_map_32.pt') 

parser.add_argument('--t1', type=float, default = 0.7)
parser.add_argument('--t2', type=float, default = 0.3)

parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'models/decoder_iter_500000.pth')
parser.add_argument('--transform', type=str, default = 'models/transformer_iter_500000.pth')

# Additional options
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = 'output_SCSA',
                    help='Directory to save the output image(s)')


# Advanced options

args = parser.parse_args()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = decoder
transform = Transform(in_planes = 512)
vgg = vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()






content = content_tf(Image.open(args.content).convert('RGB'))
style = style_tf(Image.open(args.style).convert('RGB'))
content_sem = content_tf(Image.open(args.content_sem).convert('RGB'))
style_sem = style_tf(Image.open(args.style_sem).convert('RGB'))

style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)
style_sem = style_sem.to(device).unsqueeze(0)
content_sem = content_sem.to(device).unsqueeze(0)




with torch.no_grad():

    for x in range(args.steps):

        print('iteration ' + str(x))
        
        
        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Content5_1 = enc_5(Content4_1)
        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        Style5_1 = enc_5(Style4_1)
        Content4_1_sem = enc_4(enc_3(enc_2(enc_1(content_sem))))
        Content5_1_sem = enc_5(Content4_1_sem)
        Style4_1_sem = enc_4(enc_3(enc_2(enc_1(style_sem))))
        Style5_1_sem = enc_5(Style4_1_sem)
        map_64 = torch.load(args.sem_map_64).to(device)
        map_32 = torch.load(args.sem_map_32).to(device)
        

        
        # local_adain
        Content4_1_sadain = local_adain(Content4_1, Style4_1, args, device, number=64)
        Content5_1_sadain = local_adain(Content5_1, Style5_1, args, device, number=32)
        
        content = decoder(transform(Content4_1_sadain, Style4_1, Content5_1_sadain, Style5_1, Content4_1_sem, Style4_1_sem, Content5_1_sem, Style5_1_sem, map_32, map_64, args))
        
        
        

        content.clamp(0, 255)

    content = content.cpu()
    
    output_name = '{:s}/{:s}_{:s}_{:s}{:s}'.format(
                args.output, splitext(basename(args.content))[0],
                splitext(basename(args.style))[0], "SANet_SCSA",args.save_ext
            )
    save_image(content, output_name)







