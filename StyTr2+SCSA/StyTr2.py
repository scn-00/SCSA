import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer_StyTr2 as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512, 512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512, 512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512, 512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  

parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, default = '../sem_data/31/31_paint.jpg')
parser.add_argument('--style', type=str, default = '../sem_data/31/31.jpg')

parser.add_argument('--output', type=str, default='output_StyTr2',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')


parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()




# Advanced options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
preserve_color='store_true'





device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")



content_paths = [Path(args.content)]

style_paths = [Path(args.style)]    


if not os.path.exists(output_path):
    os.mkdir(output_path)


vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg, weights_only=True))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
Trans = transformer.Transformer()
embedding = StyTR.PatchEmbed()

decoder.eval()
Trans.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path,weights_only=True)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path,weights_only=True)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path,weights_only=True)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

network = StyTR.StyTrans_StyTr2(vgg,decoder,embedding,Trans,args)
network.eval()
network.to(device)



content_tf = test_transform()
style_tf = test_transform()

for content_path in content_paths:
    for style_path in style_paths:

        content = content_tf(Image.open(content_path).convert("RGB"))
        style = style_tf(Image.open(style_path).convert("RGB"))

    
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        
        with torch.no_grad():
            output= network(content,style)       
        output = output.cpu()
                
        output_name = '{:s}/{:s}_{:s}_StyTr2{:s}'.format(
            output_path, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0],save_ext
        )

        save_image(output, output_name)
        

   

