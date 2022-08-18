from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from PIL import Image
from ..nnutils import cub_mesh as mesh_net
from ..nnutils import cub_deform2 as deform_net
from ..nnutils.nmr_pytorch import NeuralRenderer
from ..data import cub as cub_data
from ..utils import tf_visualizer
import os
import time
import numpy as np
import os.path as osp
from absl import app, flags
from tqdm import tqdm
import random
import torch
import torchvision
import torchvision.utils as vutils
import cv2
from ..AdaIN.test import do_adain

# Data:
flags.DEFINE_string('stemp_path', 'evo_trans/cachedir/snapshots/cub_net/', 'path to semantic template.')
flags.DEFINE_string('test_dir', 'test_data', 'test Data Directory')
flags.DEFINE_integer('img_size', 256, 'image size')

# Model:
flags.DEFINE_boolean('pred_cam', True, 'If true predicts camera')
flags.DEFINE_integer('axis', 1, 'symmetric axis')
flags.DEFINE_string('df_path', 'evo_trans/cachedir/snapshots/ab79/et_net_latest.pth', 'model path')

# Cub mesh:
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_boolean('multiple_cam_hypo', True, 'If true use multiple camera hypotheses')
flags.DEFINE_integer('nz_feat', 200, 'Encoded image feature size')
flags.DEFINE_integer('z_dim', 350, 'noise dimension of VAE')
flags.DEFINE_integer('gpu_num', 1, 'gpu number')
flags.DEFINE_integer('num_hypo_cams', 8, 'number of hypo cams')
flags.DEFINE_boolean('az_ele_quat', False, 'Predict camera as azi elev')
flags.DEFINE_float('scale_lr_decay', 0.05, 'Scale multiplicative factor')
flags.DEFINE_float('scale_bias', 1.0, 'Scale bias factor')

flags.DEFINE_boolean('use_texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

# Test
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('batch_size', 5, 'Size of minibatches')

## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Directory where networks are saved')
flags.DEFINE_string('vis_dir', osp.join(cache_path, 'visualization'),
                    'Root directory for visualizations')

opts = flags.FLAGS

class ShapenetTester():
    def __init__(self, opts):

        self.vis_dir = opts.vis_dir
        self.iteration_num=0
        self.opts = opts
        self.gpu_id = opts.gpu_id
        torch.cuda.set_device(opts.gpu_id)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

    def load(self):
        dic = torch.load(self.opts.df_path)
        saved_state_dict = dic["umr"]
        unwanted_keys = {"noise", "uv_sampler"}
        new_params = self.model_umr.state_dict().copy()
        for name, param in new_params.items():
            if name not in unwanted_keys:
                new_params[name].copy_(saved_state_dict[name])
        self.model_umr.load_state_dict(new_params)
        self.model.load_state_dict(dic["df"])
        print(tf_visualizer.green("Loaded checkpoint from {}.".format(self.opts.df_path)))

    def define_model(self):

        opts = self.opts
        self.avg_prob = torch.from_numpy(np.array(Image.open(osp.join(opts.stemp_path, "semantic_seg.png"))))
        # define model
        self.symmetric = opts.symmetric
        img_size = (opts.img_size, opts.img_size)
        self.model_umr = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat,
            axis=opts.axis,
            temp_path=opts.stemp_path)

        # load pretrained UMR model
        self.model_umr = self.model_umr.cuda(device=opts.gpu_id)

        ### build deformed model
        self.model = deform_net.Dense_Gated_Net(opts, self.model_umr.num_output).cuda()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        faces = self.model_umr.faces.view(1, -1, 3)
        mean = self.model_umr.get_mean_shape()
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.mean_shape = mean

        # define renderers
        self.vis_renderer = NeuralRenderer(opts.img_size)
        self.vis_renderer.ambient_light_only()
        self.vis_renderer.set_bgcolor([1, 1, 1])
        self.vis_renderer.set_light_dir([0, 1, -1], 0.4)
        self.iter_time = 0

        # load half mean shape
        self.mean_shape_half = torch.load(osp.join(opts.stemp_path, "mean_v.pth"),
                                     map_location=lambda storage, loc: storage.cuda(opts.gpu_id))
        self.vis_batch = None
        return

    def init_dataset(self):
        opts = self.opts
        self.data_module = cub_data
        self.dataloader = self.data_module.test_loader(opts, shuffle=False)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    
    def set_input(self, batch):
        opts = self.opts
        # =================================================================================== #
        #                               Load source images                                    #
        # =================================================================================== #
        self.input_imgs = batch['img'].type(torch.FloatTensor).cuda()
        self.input_imgs_t = batch['img_t'].type(torch.FloatTensor).cuda()
        self.imgs = self.input_imgs.clone()
        for b in range(self.input_imgs.size(0)):
            self.input_imgs[b] = self.resnet_transform(self.input_imgs[b])
        self.imgs_t = self.input_imgs_t.clone()
        for b in range(self.input_imgs_t.size(0)):
            self.input_imgs_t[b] = self.resnet_transform(self.input_imgs_t[b])
        self.index = batch['index']
        self.index_t = batch['index_t']
        self.name = batch['name']
        self.name_t = batch['name_t']
        self.switch_sig = batch['switch_sig']

    def get_current_visuals(self):
        self.curr_time = time.time()
        with torch.no_grad():
            outputs_t = self.model_umr.forward(self.input_imgs_t)
            outputs = self.model_umr.forward(self.input_imgs)
            img_feat = outputs['noise']
            img_feat_t = outputs_t['noise']
            output_df = self.model.forward(img_feat.unsqueeze(dim=2), img_feat.unsqueeze(dim=2),
                                           self.mean_shape_half)
            output_df_t = self.model.forward(img_feat_t.unsqueeze(dim=2), img_feat_t.unsqueeze(dim=2),
                                             self.mean_shape_half)
            output_df_d = self.model.forward(img_feat.unsqueeze(dim=2), img_feat_t.unsqueeze(dim=2),
                                             self.mean_shape_half)
            proj_cam = outputs['cam'].detach()
            pred_vs = output_df['deformed_shape']
            pred_vs_t = output_df_t['deformed_shape']
            pred_vs_d = output_df_d['deformed_shape']
            uv_flows = outputs['uvimage_pred'].permute(0, 2, 3, 1)
            uv_flows_t = outputs_t['uvimage_pred'].permute(0, 2, 3, 1)
            uv_images = torch.nn.functional.grid_sample(self.imgs, uv_flows, align_corners=True)
            uv_images_t = torch.nn.functional.grid_sample(self.imgs_t, uv_flows_t, align_corners=True)
            image_recon_s = self.mesh_render(pred_vs, proj_cam, uv_images)
            image_recon_t = self.mesh_render(pred_vs_t, proj_cam, uv_images_t)
            uv_images_evo_1 = self.generate(uv_images, uv_images_t, self.avg_prob[None], switch_sig=self.switch_sig)
            image_evo_1 = self.mesh_render(pred_vs_d, proj_cam, uv_images_evo_1)
            self.switch_sig_2 = self.switch_sig[torch.randperm(self.switch_sig.size(0))]
            uv_images_evo_2 = self.generate(uv_images, uv_images_t, self.avg_prob[None], switch_sig=self.switch_sig_2)
            image_evo_2 = self.mesh_render(pred_vs_d, proj_cam, uv_images_evo_2)
            self.switch_sig_3 = self.switch_sig[torch.randperm(self.switch_sig.size(0))]
            uv_images_evo_3 = self.generate(uv_images, uv_images_t, self.avg_prob[None], switch_sig=self.switch_sig_3)
            image_evo_3 = self.mesh_render(pred_vs_d, proj_cam, uv_images_evo_3)
            self.switch_sig_4 = self.switch_sig[torch.randperm(self.switch_sig.size(0))]
            uv_images_evo_4 = self.generate(uv_images, uv_images_t, self.avg_prob[None], switch_sig=self.switch_sig_4)
            image_evo_4 = self.mesh_render(pred_vs_d, proj_cam, uv_images_evo_4)
            vis_dict = {}
            vis_dict[f'vis_{self.curr_time}'] = torch.cat(
                [
                self.imgs,
                image_recon_s,
                image_evo_1,
                image_evo_2,
                image_evo_3,
                image_evo_4,
                image_recon_t,
                self.imgs_t,
                ], dim=3)
            return vis_dict

    def mesh_render(self, verts, cams, uv_images):
        tex = self.get_tex(uv_images)[..., None, :].repeat(1, 1, 1, 1, opts.tex_size, 1)
        image_pred = self.vis_renderer(verts.detach(), self.faces, cams.detach(), tex)
        return image_pred

    def get_tex(self,uv_images):
        uv_sampler = self.model_umr.uv_sampler
        tex = torch.nn.functional.grid_sample(uv_images, uv_sampler, align_corners=True)
        nb, nf, _, nc = tex.size()
        self.F = uv_sampler.size(1)
        tex = tex.view(tex.size(0), -1, self.F, 6, 6).permute(0, 2, 3, 4, 1)
        tex_left = tex[:, -self.model_umr.num_sym_faces:]
        tex_all = torch.cat([tex, tex_left], 1)
        return tex_all
    
    def generate(self, style, content, mask=None, switch_sig=None):
        result = do_adain(style, content, mask=mask, switch_sig=switch_sig)
        return result

    def putText(self, img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=1, color=(0, 0, 0), thickness=2):
        img = cv2.putText(img, text, position, font, font_size, color, thickness, cv2.LINE_AA)
        return img

    def add_text(self, img):
        h_int, w_int = img.shape[0]//5, img.shape[1]//8
        SW = [self.switch_sig, self.switch_sig_2, self.switch_sig_3, self.switch_sig_4]
        # write switch gate [head,neck,back,belly]
        for row in range(5):
            for column in range(2, 6):
                img = self.putText(img, f'SW={SW[column-2][row].tolist()}', (20 + w_int * column, 240 + h_int * row), font_size=0.8, color=(125,125,125))

        # write label
        img = cv2.copyMakeBorder(img, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        labels = ['Source', 'Recon_s', 'Ours_1', 'Ours_2','Ours_3', 'Ours_4', 'Recon_t', 'Target']
        for i, label in enumerate(labels, 0):
            img = self.putText(img, label, (45+w_int*i, 1360), font_size=1.5)
        return img

    def test(self):
        opts = self.opts
        self.model_umr.eval()
        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.dataloader)):
                self.set_input(batch)
                self.iteration_num += 1
                vis_dict = self.get_current_visuals()
                for k, v in vis_dict.items():
                    res = vutils.make_grid(v,nrow=1).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    img = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                    img = self.add_text(img)
                    img_small = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2))
                    cv2.imwrite(osp.join(self.vis_dir, f'{k}.jpg'), img)
                    cv2.imshow('image', img_small)
                    cv2.waitKey(0)

                del vis_dict
                print(tf_visualizer.green(f"({self.iteration_num}) Visualization saved at {self.vis_dir}."))

def main(_):
    torch.manual_seed(0)
    tester = ShapenetTester(opts)
    tester.define_model()
    tester.init_dataset()
    tester.load()
    print(tf_visualizer.blue('Start testing...'))
    tester.test()


if __name__ == '__main__':
    app.run(main)
