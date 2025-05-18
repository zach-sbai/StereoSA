import torch.nn.functional as F
import torch

def model_loss_train(disp_ests, disp_gts, img_masks):

    weights = [1.00, 1.00/6]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts[0:2:3], weights, img_masks[0:2:3]):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask_img], disp_gt[mask_img], reduction="mean"))

    return sum(all_losses)

def model_loss_test(disp_ests, disp_gts,img_masks):
    weights = [1.00]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], reduction="mean"))
    return sum(all_losses)
