import argparse

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from ..AdaIN import net
from ..AdaIN.function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    # transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None, mask = None, switch_sig=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)

    ##mask###
    if mask is not None:
        resize = transforms.Resize([64, 128])
        mask = resize(mask)
        mask = np.array(mask)
        mask_res = []
        for i in range(5):
            mask_part = mask.copy()
            mask_part[(mask_part == i)] = 255
            mask_part[(mask_part == 255) == False] = 0
                # mask_part = np.resize(mask_part.copy(), (32, 64))
            # print(mask_part.shape, np.sum(mask_part))
            # mask_part_2 = np.concatenate([mask_part, mask_part], axis=0)
            # print(mask_part_2.shape)
            mask_part = np.round(mask_part / 255.0)
            # print("mask ", mask_part.shape, np.sum(mask_part))
            mask_part = torch.from_numpy(mask_part).float().cuda()
            mask_res.append(mask_part)
        ###

    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        if mask is not None:
            feature = []
            for part in range(1, 5):
                cache = []
                for batch in range(content_f.shape[0]):
                    if switch_sig is not None and switch_sig[batch, part - 1] == 1:
                        cache.append(adaptive_instance_normalization(style_f[batch,None] * mask_res[part], content_f[batch,None] * mask_res[part], mask_res[part]))
                    else:
                        cache.append(adaptive_instance_normalization(content_f[batch,None] * mask_res[part], style_f[batch,None] * mask_res[part], mask_res[part]))
                feature.append(torch.cat(cache,dim=0))

            feat = torch.sum(torch.stack(feature,dim=0),dim=0)


        else:
            feat = adaptive_instance_normalization(content_f, style_f)
    # print("feat ", feat)
    if mask is not None:
        feat = feat * alpha + content_f * (1 - alpha) + content_f * mask_res[0]
    else:
        feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='evo_trans/AdaIN/models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='evo_trans/AdaIN/models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--mask', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

def do_adain(style, content, mask=None, switch_sig=None):
    content = content_tf(content)#[3, 512, 1024]
    style = style_tf(style)
    if args.preserve_color:
        style = coral(style, content)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style,
                                args.alpha, mask=mask, switch_sig=switch_sig)
    # print(output.shape)  # [1, 3, 512, 1024]
    # output = output.cpu()
    return output

