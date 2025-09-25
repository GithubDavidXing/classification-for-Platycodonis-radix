import numpy as np
from spectral import envi
import os
from torch.utils.data import Dataset
import random
import torch
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import cv2

class HSIloader():
    def __init__(self, dir_path, print_file=None):
        self.dir_path = dir_path
        self.dir_path += '/' if '/' not in self.dir_path and '\\' not in self.dir_path else ''
        self.os_path_file = list(os.walk(dir_path))
        self.pf = print_file
        _, self.class_name, _ = self.os_path_file[0]
        self.max_height, self.max_width = 0, 0

    def __load__(self, root_name, hdr_name, dat_name):
        if self.pf:
            print(f'Load {root_name}\\{hdr_name}\t and \t{root_name}\\{dat_name}')
        HSI = np.array(envi.open(root_name + '/' + hdr_name, root_name + '/' + dat_name).load())
        return HSI

    def load(self):
        self.dataset = []
        for main_idx in range(1, len(self.os_path_file)):
            dat_list = []
            hdr_list = []
            main_root, sub_root, data_name = self.os_path_file[main_idx]
            for file_name in data_name:
                if file_name.endswith('.dat'):
                    dat_list.append(file_name)
                elif file_name.endswith('.hdr'):
                    hdr_list.append(file_name)
            data_num = len(dat_list)

            for idx in range(data_num):
                dat_name = dat_list[idx]
                hdr_name = dat_name[:-3] + 'hdr'
                filename = dat_name
                try:
                    HSI = self.__load__(root_name=main_root, hdr_name=hdr_name, dat_name=dat_name)
                    HSI /= HSI.max()  # 归一化处理
                    height, width, _ = HSI.shape
                    self.max_height = max(self.max_height, height)
                    self.max_width = max(self.max_width, width)
                    self.dataset.append([HSI, main_idx - 1, filename])
                except:
                    raise IndexError(f'.dat and .hdr not match. Missing file:{main_root + hdr_name}')
        return self.dataset

# 数据增强：图像翻转
def arguement(img, vFlip, hFlip):
    for j in range(vFlip):
        img = img[:, ::-1].copy()  # 垂直翻转
    for j in range(hFlip):
        img = img[::-1, :].copy()  # 水平翻转
    return img


def attention(HSI, pf=False):
    h, w, c = HSI.shape
    f = HSI.reshape([-1, c])

    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=10, n_init=10).fit(f)
    l = kmeans.labels_.reshape(h, w)

    # 使用图像中心的较大区域来判断类别（而非单一的中心点）
    center_region = l[h//3:h*2//3, w//3:w*2//3]  # 选择图像中间部分作为标准
    foreground_class = np.argmax(np.bincount(center_region.flatten()))  # 出现频率最多的类别作为麦冬区域


    l = (l == foreground_class).astype(np.float32)


    kernel = np.ones((5, 5), np.uint8)
    l = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)

    num_labels, labels_im = cv2.connectedComponents(l.astype(np.uint8))
    largest_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])  # 找到面积最大的连通域
    l = (labels_im == largest_label).astype(np.float32)


    maskedimg = np.zeros_like(HSI)
    for i in range(c):
        maskedimg[:, :, i] = HSI[:, :, i] * l


    non_zero_mask = (maskedimg.sum(axis=2) != 0).astype(np.uint8)


    x, y, w, h = cv2.boundingRect(non_zero_mask)


    extracted_hsi = maskedimg[y:y + h, x:x + w, :]

    # 可视化掩膜和提取结果
    if pf:
        plt.subplot(1, 3, 1)
        plt.imshow(l, cmap='gray')
        plt.title('Mask')
        plt.subplot(1, 3, 2)
        plt.imshow(HSI.mean(-1), cmap='gray')
        plt.title('Original Image')
        plt.subplot(1, 3, 3)
        plt.imshow(extracted_hsi.mean(-1), cmap='gray')
        plt.title('Extracted HSI')
        plt.show()

    return extracted_hsi



def process(hsi, clip=True, pf=False, mode=1, BS=None):
    pro_hsi = hsi[:, :, 10: -10]  # 去掉前后各10个波段
    hi, wi, num_bands = hsi.shape

    # 归一化
    for h in range(hi):
        for w in range(wi):
            spectral_curve = hsi[h, w, :]
            max_value = spectral_curve.max()
            hsi[h, w, :] = spectral_curve / (max_value + 1.e-20)

    # 使用Savitzky-Golay滤波器平滑光谱
    for h in range(hi):
        for w in range(wi):
            hsi[h, w, :] = savgol_filter(hsi[h, w, :], 5, 2, mode='mirror')


    if clip:
        pro_hsi = attention(pro_hsi, pf)
    return pro_hsi

# 自定义数据集类
class AllDataset(Dataset):
    def __init__(self, data_root, print_file=False, arg=False, mode=1, BS=None):
        super(AllDataset, self).__init__()
        self.arg = arg
        self.pf = print_file
        self.imgloader = HSIloader(data_root, print_file=print_file)
        self.dataset = self.imgloader.load()
        self.img_num = len(self.dataset)
        self.pro_data = []

        # 提前计算 max_height 和 max_width
        self.max_height = max([hsi.shape[0] for hsi, _, _ in self.dataset])
        self.max_width = max([hsi.shape[1] for hsi, _, _ in self.dataset])

        first_processed = False

        for idx in range(self.img_num):
            hsi, label, filename = self.dataset[idx]
            pro_hsi = process(hsi, clip=True, pf=print_file, mode=mode, BS=BS)

            # 保存第一张原始和填充后的图像
            if not first_processed:
                save_path_original = './first_image/original_image.png'
                save_path_padded = './first_image/padded_image.png'
                os.makedirs(os.path.dirname(save_path_original), exist_ok=True)

                # 保存原始图像（通过波段平均获取2D图像）
                plt.imsave(save_path_original, hsi.mean(axis=-1), cmap='gray')

                # 填充图像并保存填充后的版本
                padded_pro_hsi = self.__pad__(pro_hsi, self.max_height, self.max_width)
                plt.imsave(save_path_padded, padded_pro_hsi.mean(axis=-1), cmap='gray')

                first_processed = True

            # # 数据增强：随机翻转
            if self.arg:
                vFlip = random.randint(0, 1)
                hFlip = random.randint(0, 1)
                pro_hsi = arguement(pro_hsi, vFlip, hFlip)
                vFlip = random.randint(0, 1)
                hFlip = random.randint(0, 1)
                pro_hsi2 = arguement(pro_hsi, vFlip, hFlip)
                self.pro_data.append([pro_hsi2, label, filename])

            self.pro_data.append([pro_hsi, label, filename])

        # 填充所有图像
        self.pro_data = [(self.__pad__(img, self.max_height, self.max_width), label, filename)
                         for img, label, filename in self.pro_data]


    def __pad__(self, img, max_height, max_width):
        h, w, c = img.shape
        padded_img = np.zeros((max_height, max_width, c))

        # 计算填充大小
        pad_top = (max_height - h) // 2
        pad_left = (max_width - w) // 2


        padded_img[pad_top:pad_top + h, pad_left:pad_left + w, :] = img
        return padded_img


    def get_labels(self):
        return [item[1] for item in self.pro_data]

    def __getitem__(self, idx):
        return torch.FloatTensor(np.ascontiguousarray(self.pro_data[idx][0]).T), self.pro_data[idx][1], \
               self.pro_data[idx][2]

        # return torch.FloatTensor(np.ascontiguousarray(self.pro_data[idx][0])), self.pro_data[idx][1], self.pro_data[idx][2]

    def __len__(self):
        return self.img_num if not self.arg else self.img_num * 2

# 获取数据集
def get_data(dir_path, printf=True, arg=False, mode=1, BS=None):
    all_data = AllDataset(dir_path, arg=arg, print_file=printf, mode=mode)
    return all_data

# 主函数
def main():
    # 设置数据集路径
    dataset_path = './data'  # 请确保此路径指向包含 .dat 和 .hdr 文件的目录

    # 创建数据集对象
    dataset = AllDataset(data_root=dataset_path, print_file=True, arg=False, mode=1)

    # 输出数据集基本信息
    print(f"数据集共有 {len(dataset)} 个样本。")

    # 打印每个类别的数量
    labels = dataset.get_labels()
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("每个类别的样本数量：")
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个样本")

    # 加载第一个样本并显示
    first_sample, label, filename = dataset[0]
    print(f"第一个样本的信息: 文件名: {filename}, 类别: {label}, 数据形状: {first_sample.shape}")

    # 展示第一张原始图像和填充后的图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(first_sample.mean(-1), cmap='gray')
    plt.title('Padded Image')
    plt.subplot(1, 2, 2)
    original_image_path = './first_image/original_image.png'
    if os.path.exists(original_image_path):
        original_image = plt.imread(original_image_path)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
    plt.show()
