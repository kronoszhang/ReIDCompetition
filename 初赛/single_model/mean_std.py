# -*- coding: utf-8 -*-

import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch


# calculate means and std
img_h, img_w = 256, 128
# imgs = np.zeros([img_w, img_h, 3, 1])
train_path = "/project/ywchong/ywchong/CODE/zc/AIC/data/MyDataset_V2/bounding_box_train_all"   # x 训练集路径
test_query_path = "/project/ywchong/ywchong/CODE/zc/AIC/data/chu_test_a/chu_test_a//query_a"   # 测试集A或B的query的路径
test_gallery_path = "/project/ywchong/ywchong/CODE/zc/AIC/data/chu_test_a/chu_test_a//gallery_a"  # 测试集A或B的gallery的路径

train_images = os.listdir(train_path)
# assert len(train_images) == 20429

query_images = os.listdir(test_query_path)
gallery_images = os.listdir(test_gallery_path)
test_images = query_images + gallery_images
# assert len(test_images) == 5366 + 1348

images = train_images
# images = test_images
# images = train_images + test_images


imgs = []
means, stdevs = [], []
for image in images:
    if image in train_images:
        img_path = os.path.join(train_path, image)
    else:
        img_path = os.path.join(test_images, image)
    img = Image.open(img_path)
    transform = transforms.ToTensor()
    img = transform(img)
    img = img.unsqueeze(dim=-1)
    img = img.permute(2, 1, 0, 3)
    imgs.append(img)
    # imgs = np.concatenate((imgs, img), axis=3)

print(imgs.shape)
imgs = torch.cat(imgs, dim=-1)
print(imgs.shape)
exit()
imgs = imgs.astype(np.float32)/255.
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))


# means.reverse()  # BGR --> RGB
# stdevs.reverse()
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
