import torch
import os
import cv2 as cv
from dataloader import *
from utils import *
import numpy as np
from segnet import *
from collections import OrderedDict

unknown_code = 128

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg

def compute_test(model):
    out_test_path = '/home/zy/deepmatting/matting/merged_test/'
    test_images = [f for f in os.listdir(out_test_path) if
                   os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]
    samples = random.sample(test_images, 10)

    bg_test = '/home/zy/deepmatting/matting/bg_test/'
    test_bgs = [f for f in os.listdir(bg_test) if
                os.path.isfile(os.path.join(bg_test, f)) and f.endswith('.jpg')]
    sample_bgs = random.sample(test_bgs, 10)

    for i in range(len(samples)):
        filename = samples[i]
        image_name = filename.split('.')[0]

        print('\nStart processing image: {}'.format(filename))

        bgr_img = cv.imread(os.path.join(out_test_path, filename))
        bg_h, bg_w = bgr_img.shape[:2]

        a = get_alpha_test(image_name)
        #print(image_name)
        a_h, a_w = a.shape[:2]

        alpha = np.zeros((bg_h, bg_w), np.float32)
        alpha[0:a_h, 0:a_w] = a
        trimap = generate_trimap(alpha)
        different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)
        x, y = random_choice(trimap, crop_size)

        bgr_img = safe_crop(bgr_img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)
        trimap = safe_crop(trimap, x, y, crop_size)
        cv.imwrite('test_image/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
        cv.imwrite('test_image/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
        cv.imwrite('test_image/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))

        x_test = np.empty((1, img_rows, img_cols, 4), dtype = np.float32)
        x_test[0, :, :, 0:3] = bgr_img / 255.
        x_test[0, :, :, 3] = trimap / 255.

        y_true = np.empty((1, img_rows, img_cols, (2+9)), dtype = np.float32)
        y_true[0, :, :, 0] = alpha / 255.
        y_true[0, :, :, 1] = trimap / 255.
        x_test = torch.from_numpy(x_test).cuda()
        x_test = x_test.transpose(1,2)
        x_test = x_test.transpose(1,3)
        y_pred = model(x_test)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.reshape(y_pred, (img_rows, img_cols))
        y_pred = y_pred * 255.0

        y_pred = get_final_output(y_pred, trimap)
        y_pred = y_pred.astype(np.uint8)

        sad_loss = compute_sad_loss(y_pred, alpha, trimap)
        mse_loss = compute_mse_loss(y_pred, alpha, trimap)
        str_msg = 'sad_loss: %.4f, mse_loss: %.4f, crop_size: %s' % (sad_loss, mse_loss, str(crop_size))

        print(str_msg)
        out = y_pred.copy()
        draw_str(out, (10, 20), str_msg)
        cv.imwrite('test_image/{}_out.png'.format(i), out)

        sample_bg = sample_bgs[i]
        bg = cv.imread(os.path.join(bg_test, sample_bg))
        bh, bw = bg.shape[:2]
        wratio = img_cols / bw
        hratio = img_rows / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
        im, bg = composite4(bgr_img, bg, y_pred, img_cols, img_rows)
        cv.imwrite('test_image/{}_compose.png'.format(i), im)
        cv.imwrite('test_image/{}_new_bg.png'.format(i), bg)


def load_model(model):
    pretrained_dict = torch.load('./logs/model_9_0.0328.pth')
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        # load params
    model.load_state_dict(new_state_dict)
    return model

if __name__ == '__main__':
    model = EncoderDecoder()
    model = model.cuda()

    model = load_model(model)

    model.train()
    compute_test(model)