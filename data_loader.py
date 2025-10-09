import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

# class MVTecDRAEMTestDataset(Dataset):

#     def __init__(self, root_dir, resize_shape=None):
#         self.root_dir = root_dir
#         self.images = sorted(glob.glob(root_dir + "/*/*.png"))
#         self.resize_shape = resize_shape

#     def __len__(self):
#         return len(self.images)

#     def transform_image(self, image_path, mask_path):
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         if mask_path is not None:
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         else:
#             mask = np.zeros((image.shape[0], image.shape[1]))
#         if self.resize_shape != None:
#             image = cv2.resize(image,
#                                dsize=(self.resize_shape[1],
#                                       self.resize_shape[0]))
#             mask = cv2.resize(mask,
#                               dsize=(self.resize_shape[1],
#                                      self.resize_shape[0]))

#         image = image / 255.0
#         mask = mask / 255.0

#         image = np.array(image).reshape(
#             (image.shape[0], image.shape[1], 3)).astype(np.float32)
#         mask = np.array(mask).reshape(
#             (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

#         image = np.transpose(image, (2, 0, 1))
#         mask = np.transpose(mask, (2, 0, 1))
#         return image, mask

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_path = self.images[idx]
#         dir_path, file_name = os.path.split(img_path)
#         base_dir = os.path.basename(dir_path)
#         if base_dir == 'good':
#             image, mask = self.transform_image(img_path, None)
#             has_anomaly = np.array([0], dtype=np.float32)
#         else:
#             mask_path = os.path.join(dir_path, '../../ground_truth/')
#             mask_path = os.path.join(mask_path, base_dir)
#             mask_file_name = file_name.split(".")[0] + "_mask.png"
#             mask_path = os.path.join(mask_path, mask_file_name)
#             image, mask = self.transform_image(img_path, mask_path)
#             has_anomaly = np.array([1], dtype=np.float32)

#         sample = {
#             'image': image,
#             'has_anomaly': has_anomaly,
#             'mask': mask,
#             'idx': idx
#         }

#         return sample


class MVTecDRAEM_Test_Visual_Dataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))[:2]
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image,
                               dsize=(self.resize_shape[1],
                                      self.resize_shape[0]))
            mask = cv2.resize(mask,
                              dsize=(self.resize_shape[1],
                                     self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {
            'image': image,
            'has_anomaly': has_anomaly,
            'mask': mask,
            'idx': idx
        }

        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "/*.png"))

        self.anomaly_source_paths = sorted(
            glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)),
                                   3,
                                   replace=False)
        aug = iaa.Sequential([
            self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]],
            self.augmenters[aug_ind[2]]
        ])
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img,
                                        dsize=(self.resize_shape[1],
                                               self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2**(torch.randint(min_perlin_scale, perlin_scale,
                                          (1, )).numpy()[0])
        perlin_scaley = 2**(torch.randint(min_perlin_scale, perlin_scale,
                                          (1, )).numpy()[0])

        perlin_noise = rand_perlin_2d_np(
            (self.resize_shape[0], self.resize_shape[1]),
            (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold,
                              np.ones_like(perlin_noise),
                              np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (
            1 - beta) * img_thr + beta * image * (perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(
                perlin_thr, dtype=np.float32), np.array([0.0],
                                                        dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly],
                                                  dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image,
                           dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], image.shape[2])).astype(
                np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(
            image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1, )).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths),
                                           (1, )).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(
            self.image_paths[idx],
            self.anomaly_source_paths[anomaly_source_idx])
        sample = {
            'image': image,
            "anomaly_mask": anomaly_mask,
            'augmented_image': augmented_image,
            'has_anomaly': has_anomaly,
            'idx': idx
        }

        return sample


class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, category_name, resize_shape=None):
        """
        Args:
            root_dir (string): MVTec dataset 的根目錄 (例如: "./mvtec")
            category_name (string): 物件類別名稱 (例如: "bottle")
            resize_shape (tuple): 圖像縮放的目標尺寸 (H, W)
        """
        self.root_dir = root_dir
        self.category_name = category_name
        self.resize_shape = resize_shape

        # 構建基礎路徑
        category_path = os.path.join(self.root_dir, self.category_name)
        test_path = os.path.join(category_path, 'test')
        ground_truth_path = os.path.join(category_path, 'ground_truth')

        self.image_paths = []
        self.anomaly_masks_paths = []  # 儲存所有異常掩碼的路徑
        self.is_anomaly_flags = []  # 記錄每個樣本是否為異常

        # 1. 處理 'good' 圖像
        good_image_paths = sorted(
            glob.glob(os.path.join(test_path, 'good', '*.png')))
        self.image_paths.extend(good_image_paths)
        self.anomaly_masks_paths.extend([None] *
                                        len(good_image_paths))  # 正常圖像沒有掩碼
        self.is_anomaly_flags.extend([0] * len(good_image_paths))

        # 2. 處理異常圖像和對應的 ground_truth
        defect_types = sorted(os.listdir(os.path.join(test_path)))
        defect_types = [
            d for d in defect_types
            if d != 'good' and os.path.isdir(os.path.join(test_path, d))
        ]

        for defect_type in defect_types:
            defect_image_paths = sorted(
                glob.glob(os.path.join(test_path, defect_type, '*.png')))
            self.image_paths.extend(defect_image_paths)
            self.is_anomaly_flags.extend([1] * len(defect_image_paths))

            # 對於每個異常圖像，找到其對應的 ground_truth 掩碼
            for img_path in defect_image_paths:
                img_filename = os.path.basename(img_path)
                # ****** 修改這裡：將 '_mask' 插入到檔案名中 ******
                # 假設檔案名格式為 'xxx.png'，我們希望變成 'xxx_mask.png'
                name_without_ext, ext = os.path.splitext(img_filename)
                gt_mask_filename = f"{name_without_ext}_mask{ext}"

                gt_mask_path = os.path.join(
                    ground_truth_path, defect_type,
                    gt_mask_filename)  # 使用新的帶 _mask 的檔案名
                self.anomaly_masks_paths.append(gt_mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # 載入原始圖像
        image = cv2.imread(image_path)
        image = cv2.resize(image,
                           dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], image.shape[2])).astype(
                np.float32) / 255.0

        # 載入真實異常掩碼
        if self.is_anomaly_flags[idx] == 1:
            gt_mask_path = self.anomaly_masks_paths[idx]
            if os.path.exists(gt_mask_path):
                anomaly_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                anomaly_mask = cv2.resize(anomaly_mask,
                                          dsize=(self.resize_shape[1],
                                                 self.resize_shape[0]))
                anomaly_mask = anomaly_mask / 255.0  # 二值化到 0-1 範圍
            else:
                print(
                    f"Warning: Ground truth mask not found for {image_path}. Using all zeros mask."
                )
                anomaly_mask = np.zeros(self.resize_shape, dtype=np.float32)
        else:
            anomaly_mask = np.zeros(self.resize_shape,
                                    dtype=np.float32)  # 正常圖像的掩碼是全零

        # 將圖像和掩碼轉換為 PyTorch tensor 的格式 (C, H, W)
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        anomaly_mask = np.expand_dims(anomaly_mask,
                                      axis=0)  # (H, W) -> (1, H, W)

        # 為了與訓練集的 `sample` 結構一致，`augmented_image` 設為原始圖像
        augmented_image = image.copy()

        sample = {
            'image': image,
            "anomaly_mask": anomaly_mask,
            'augmented_image': augmented_image,
            'has_anomaly': np.array([self.is_anomaly_flags[idx]],
                                    dtype=np.float32),
            'idx': idx
        }

        return sample
