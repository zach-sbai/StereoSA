import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse

from torchvision import transforms
from tqdm import trange
from PIL import Image

from models import __models__
from datasets import ETH3D_loader as et
from datasets.data_io import pfm_imread
from utils.visualization import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='StereoSA')
parser.add_argument('--model', default='StereoSA', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default='/datasets/ETH3D/', help='data path')
parser.add_argument('--loadckpt', default='./checkpoint/StereoSA_sceneflow.ckpt',help='load the weights from a specific checkpoint')

args = parser.parse_args()

all_limg, all_rimg, all_disp, all_mask = et.et_loader(args.datapath)

model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
model.eval()

os.makedirs('./demo/ETH3D/', exist_ok=True)
os.makedirs('./demo/ETH3D_gt/', exist_ok=True)
os.makedirs('./demo/ETH3D_error/', exist_ok=True)

state_dict = torch.load(args.loadckpt)
model_dict = model.state_dict()
pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)

pred_mae = 0
pred_op = 0
for i in trange(len(all_limg)):
    limg = Image.open(all_limg[i]).convert('RGB')
    rimg = Image.open(all_rimg[i]).convert('RGB')

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

    disp_gt, _ = pfm_imread(all_disp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
    disp_gt[disp_gt == np.inf] = 0
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    occ_mask = np.ascontiguousarray(Image.open(all_mask[i]))

    with torch.no_grad():

        pred_disp  = model(limg_tensor, rimg_tensor, train_status=False)[-1]

        pred_disp = pred_disp[:, hi - h:, wi - w:]

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 1
    mask = (disp_gt > 0) & (occ_mask == 255)
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum(pred_error > op_thresh) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

    filename = os.path.join('./demo/ETH3D/', all_limg[i].split('/')[-2]+all_limg[i].split('/')[-1])
    filename_gt = os.path.join('./demo/ETH3D_gt/', all_limg[i].split('/')[-2]+all_limg[i].split('/')[-1])
    pred_np_save = np.round(predict_np*4 * 256).astype(np.uint16)
    cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    disp_gt_save = np.round(disp_gt*4 * 256).astype(np.uint16)
    cv2.imwrite(filename_gt, cv2.applyColorMap(cv2.convertScaleAbs(disp_gt_save, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    im_err = vis(predict_np[None, :, :], disp_gt[None, :, :], abs_thres=3., rel_thres=0.05, dilate_radius=1)[0, :, :, :].transpose(1, 2, 0)

    im_err = np.round(im_err * 256).astype(np.uint8)
    im_err = cv2.cvtColor(im_err, cv2.COLOR_BGR2RGB)
    fn = os.path.join('./demo/ETH3D_error/', all_limg[i].split('/')[-2]+all_limg[i].split('/')[-1])
    cv2.imwrite(fn, im_err)

print(pred_mae / len(all_limg))
print(pred_op / len(all_limg))
