import os

import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2

from torchvision import transforms
from tqdm import trange
from PIL import Image

from datasets import KITTI2015loader as kt2015
from datasets import KITTI2012loader as kt2012
from models import __models__
from utils.visualization import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='StereoSA')
parser.add_argument('--model', default='StereoSA', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default="/datasets/kitti_2015/training/", help='data path')
parser.add_argument('--kitti', type=str, default='2015')
parser.add_argument('--loadckpt', default='./checkpoint/StereoSA_sceneflow.ckpt', help='load the weights from a specific checkpoint')
args = parser.parse_args()

if args.kitti == '2015':
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(args.datapath)
else:
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(args.datapath)

test_limg = all_limg + test_limg
test_rimg = all_rimg + test_rimg
test_ldisp = all_ldisp + test_ldisp

model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
model.eval()

state_dict = torch.load(args.loadckpt)
model_dict = model.state_dict()
pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)

param_n = np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
print("param size = %fMB", param_n)

pred_mae = 0
pred_op = 0

save_dir = "./error"
save_dir1 = "./disp_gt"
save_dir2 = "./disp_colored"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir1, exist_ok=True)
os.makedirs(save_dir2, exist_ok=True)


for i in trange(len(test_limg)):
    limg = Image.open(test_limg[i]).convert('RGB')
    rimg = Image.open(test_rimg[i]).convert('RGB')

    os.makedirs(save_dir, exist_ok=True)

    fn = test_limg[i]
    fn = os.path.join(save_dir, fn.split('/')[-1])

    fn1 = test_limg[i]
    fn1 = os.path.join(save_dir1, fn.split('/')[-1])

    fn2 = test_limg[i]
    fn2 = os.path.join(save_dir2, fn.split('/')[-1])

    w, h = limg.size
    m = 32
    wi, hi = (w // m + 1) * m, (h // m + 1) * m
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    limg_tensor = transform(limg)
    rimg_tensor = transform(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt = Image.open(test_ldisp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        pred_disp  = model(limg_tensor, rimg_tensor, train_status=False)[-1]
        pred_disp = pred_disp[:, hi - h:, wi - w:]


    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 3
    mask = (disp_gt > 0) & (disp_gt < args.maxdisp)
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

    disp_est_uint = np.round(predict_np * 256).astype(np.uint16)
    cv2.imwrite(fn2, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    disp_gt_uint = np.round(disp_gt * 256).astype(np.uint16)
    cv2.imwrite(fn1, cv2.applyColorMap(cv2.convertScaleAbs(disp_gt_uint, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


    im_err = vis(predict_np[None, :, :], disp_gt[None, :, :], abs_thres=3., rel_thres=0.05, dilate_radius=1)[0, :, :, :].transpose(1, 2, 0)

    im_err = np.round(im_err * 256).astype(np.uint8)
    im_err = cv2.cvtColor(im_err, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fn, im_err)


print("#### EPE", pred_mae / len(test_limg))
print("#### >3.0", pred_op / len(test_limg))

