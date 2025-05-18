from __future__ import print_function, division
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import cv2

from tqdm import trange

from models import __models__
from utils import *
from datasets import middlebury_loader as mb

from PIL import Image
from torchvision import transforms
from datasets.data_io import pfm_imread
from utils.visualization import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='StereoSA')
parser.add_argument('--model', default='StereoSA', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/datasets/Middlebury/", help='data path')
parser.add_argument('--resolution', type=str, default='H')
parser.add_argument('--loadckpt', default='./checkpoint/StereoSA_sceneflow.ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--cv_scale', type=int, default=8, help='cost volume scale factor', choices=[8, 4])
args = parser.parse_args()

train_limg, train_rimg, train_gt, test_limg, test_rimg = mb.mb_loader(args.datapath, res=args.resolution)
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
model.eval()

state_dict = torch.load(args.loadckpt)
model_dict = model.state_dict()
pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)

os.makedirs('./demo/middlebury/', exist_ok=True)
os.makedirs('./demo/middlebury_gt/', exist_ok=True)
os.makedirs('./demo/error/', exist_ok=True)

def test_trainset():
    op = 0
    mae = 0

    for i in trange(len(train_limg)):

        limg_path = train_limg[i]
        rimg_path = train_rimg[i]

        limg = Image.open(limg_path).convert('RGB')
        rimg = Image.open(rimg_path).convert('RGB')

        w, h = limg.size
        wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32

        limg = limg.crop((w - wi, h - hi, w, h))
        rimg = rimg.crop((w - wi, h - hi, w, h))

        limg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
        rimg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            pred_disp  = model(limg_tensor, rimg_tensor, train_status=False)[-1]
            pred_disp = pred_disp[:, hi - h:, wi - w:]

        pred_np = pred_disp.squeeze().cpu().numpy()

        torch.cuda.empty_cache()

        disp_gt, _ = pfm_imread(train_gt[i])
        disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
        disp_gt[disp_gt == np.inf] = 0

        occ_mask = Image.open(train_gt[i].replace('disp0GT.pfm', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)

        mask = (disp_gt <= 0) | (occ_mask != 255) | (disp_gt >= args.maxdisp)

        error = np.abs(pred_np - disp_gt)
        error[mask] = 0

        print("Bad", limg_path, np.sum(error > 2.0) / (w * h - np.sum(mask)))

        op += np.sum(error > 2.0) / (w * h - np.sum(mask))
        mae += np.sum(error) / (w * h - np.sum(mask))

        filename = os.path.join('./demo/middlebury/', limg_path.split('/')[-2]+limg_path.split('/')[-1])
        filename_gt = os.path.join('./demo/middlebury_gt/', limg_path.split('/')[-2]+limg_path.split('/')[-1])
        pred_np_save = np.round(pred_np * 256).astype(np.uint16)
        cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        disp_gt_save = np.round(disp_gt * 256).astype(np.uint16)
        cv2.imwrite(filename_gt, cv2.applyColorMap(cv2.convertScaleAbs(disp_gt_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


        im_err = vis(pred_np[None, :, :], disp_gt[None, :, :], abs_thres=3., rel_thres=0.05, dilate_radius=1)[0, :, :, :].transpose(1, 2, 0)

        im_err = np.round(im_err * 256).astype(np.uint8)
        im_err = cv2.cvtColor(im_err, cv2.COLOR_BGR2RGB)
        fn = os.path.join('./demo/error/', limg_path.split('/')[-2]+limg_path.split('/')[-1])
        cv2.imwrite(fn, im_err)



    print("Bad 2.0", op / 15 * 100)
    print("EPE", mae / 15)


if __name__ == '__main__':
    test_trainset()
