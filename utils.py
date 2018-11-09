import numpy as np
import cv2 as cv
import torch.nn.functional as F
import torch

img_rows = 320
img_cols = 320

epsilon = 1e-6
epsilon_sqr = epsilon ** 2

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def safe_crop(mat, x, y, crop_size=(img_rows, img_cols)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y : y + crop_height, x : x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    # Compress Image
    if crop_size != (img_rows, img_cols):
        ret = cv.resize(ret, dsize=(img_rows, img_cols), interpolation=cv.INTER_NEAREST)
    return ret

"""
Absolute difference between the ground truth alpha values and 
the predicted alpha values at each pixel
"""
def alpha_prediction_loss(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    diff = y_pred[:,:,:,0] - y_true[:,:,:,0]
    diff = diff*mask
    num_pixels = torch.sum(mask)
    return torch.sum(torch.sqrt(diff.pow(2) + epsilon_sqr)) / (num_pixels + epsilon)
    #return torch.sum(torch.sqrt(diff.pow(2))) / (num_pixels)

def compositional_loss(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    #mask = mask.view(-1, img_rows, img_cols, 1)
    image = y_true[:, :, :, 2:5]
    fg = y_true[:, :, :, 5:8]
    bg = y_true[:, :, :, 8:11]
    c_g = image
    c_p = y_pred*fg + (1.0 - y_pred)*bg
    diff = c_p-c_g
    diff = diff*mask.unsqueeze(3)
    num_pixels = torch.sum(mask)
    return torch.sum(torch.sqrt(diff.pow(2) + epsilon_sqr)) / (num_pixels + epsilon)
    #return torch.sum(torch.sqrt(diff.pow(2))) / (num_pixels)

def compute_mse_loss(pred, target, trimap):
    error_map = (pred-target)/255
    mask = np.equal(trimap, 128).astype(np.float32)
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    return loss

def compute_sad_loss(pred, target, trimap):
    error_map = np.abs(pred - target) / 255.
    mask = np.equal(trimap, 128).astype(np.float32)
    loss = np.sum(error_map * mask)

    # the loss is scaled by 1000 due to the large images used in our experiment.
    loss = loss / 1000
    # print('sad_loss: ' + str(loss))
    return loss

def overall_loss(y_true, y_pred):
    w_l = 1
    return w_l * alpha_prediction_loss(y_true, y_pred) + (1 - w_l) * compositional_loss(y_true, y_pred)

def get_final_output(out, trimap):
    mask = np.equal(trimap, 128).astype(np.float32)
    return (1 - mask) * trimap + mask * out

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
