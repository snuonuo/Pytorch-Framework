import os
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tools.my_dataset import RMBDataset
from tools.common_tools import transform_invert


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    # 自己实现__init__函数
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or isinstance(p, float)
        self.snr = snr
        self.p = p

    # 自己实现 __call__函数
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# ============================ step 1/5 数据 ============================
split_dir = os.path.join("..", "..", "..", "data", "rmb_split")
train_dir = os.path.join(split_dir, "zuoye")
valid_dir = os.path.join(split_dir, "zuoye")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    AddPepperNoise(0.9, p=0.5), # 自定义的类在这里引用, 不用实例化, compose会将所有操作作用到img上
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
# valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
# valid_loader = DataLoader(dataset=valid_data, batch_size=1)


# ============================ step 5/5 训练 ============================
for epoch in range(1):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W

        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()





