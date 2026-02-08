import os  # 導入 os 模組，用於與操作系統進行交互，例如路徑操作。
import numpy as np  # 導入 numpy 並命名為 np，用於數值計算和矩陣操作。
from torch.utils.data import Dataset  # 從 PyTorch 導入 Dataset 類，這是自定義數據集的父類。
import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
import cv2  # 導入 OpenCV 庫，用於圖像處理。
import glob  # 導入 glob 模組，用於查找符合特定規則的文件路徑名。
import imgaug.augmenters as iaa  # 導入 imgaug 的增強器模組，用於圖像數據增強。
from perlin import rand_perlin_2d_np  # 從 perlin 模組導入 rand_perlin_2d_np 函數，用於生成 2D Perlin 噪聲。

class MVTecDRAEM_Test_Visual_Dataset(Dataset):  # 定義測試可視化數據集類，繼承自 Dataset。

    def __init__(self, root_dir, resize_shape=None):  # 初始化函數，接收根目錄和調整大小的形狀。
        self.root_dir = root_dir  # 保存數據集根目錄路徑。
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))[:2]  # 獲取根目錄下所有子目錄中的 png 圖像，排序並只取前 2 張用於可視化測試。
        self.resize_shape = resize_shape  # 保存圖像調整的大小。

    def __len__(self):  # 定義 __len__ 方法，返回數據集的大小。
        return len(self.images)  # 返回圖像列表的長度。

    def transform_image(self, image_path, mask_path):  # 定義圖像轉換方法，處理圖像和掩碼。
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 使用 OpenCV 讀取圖像，保持彩色模式。
        if mask_path is not None:  # 如果提供了掩碼路徑。
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 讀取掩碼圖像，使用灰度模式。
        else:  # 如果沒有提供掩碼路徑（正常樣本）。
            mask = np.zeros((image.shape[0], image.shape[1]))  # 創建全零掩碼，形狀與圖像一致。
        if self.resize_shape != None:  # 如果設置了調整大小的形狀。
            image = cv2.resize(image,  # 調整圖像大小。
                               dsize=(self.resize_shape[1],  # 目標寬度。
                                      self.resize_shape[0]))  # 目標高度。
            mask = cv2.resize(mask,  # 調整掩碼大小。
                              dsize=(self.resize_shape[1],  # 目標寬度。
                                     self.resize_shape[0]))  # 目標高度。

        image = image / 255.0  # 將圖像像素值歸一化到 [0, 1] 範圍。
        mask = mask / 255.0  # 將掩碼像素值歸一化到 [0, 1] 範圍。

        image = np.array(image).reshape(  # 將圖像轉換為 numpy 數組並調整形狀。
            (image.shape[0], image.shape[1], 3)).astype(np.float32)  # 確保是 3 通道且數據類型為 float32。
        mask = np.array(mask).reshape(  # 將掩碼轉換為 numpy 數組並調整形狀。
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)  # 確保是 1 通道且數據類型為 float32。

        image = np.transpose(image, (2, 0, 1))  # 轉置圖像維度，從 (H, W, C) 變為 (C, H, W)，符合 PyTorch 格式。
        mask = np.transpose(mask, (2, 0, 1))  # 轉置掩碼維度，從 (H, W, C) 變為 (C, H, W)。
        return image, mask  # 返回處理後的圖像和掩碼。

    def __getitem__(self, idx):  # 定義 __getitem__ 方法，用於獲取單個樣本。
        if torch.is_tensor(idx):  # 如果索引是 tensor。
            idx = idx.tolist()  # 將其轉換為列表。

        img_path = self.images[idx]  # 獲取圖像路徑。
        dir_path, file_name = os.path.split(img_path)  # 分離目錄路徑和文件名。
        base_dir = os.path.basename(dir_path)  # 獲取上一級目錄名稱（即類別或缺陷類型）。
        if base_dir == 'good':  # 如果是正常樣本。
            image, mask = self.transform_image(img_path, None)  # 調用轉換方法，掩碼為 None。
            has_anomaly = np.array([0], dtype=np.float32)  # 標記無異常。
        else:  # 如果是異常樣本。
            mask_path = os.path.join(dir_path, '../../ground_truth/')  # 構建 Ground Truth 目錄路徑。
            mask_path = os.path.join(mask_path, base_dir)  # 加入缺陷類型子目錄。
            mask_file_name = file_name.split(".")[0] + "_mask.png"  # 構建掩碼文件名。
            mask_path = os.path.join(mask_path, mask_file_name)  # 構建完整的掩碼路徑。
            image, mask = self.transform_image(img_path, mask_path)  # 調用轉換方法，傳入圖像和掩碼路徑。
            has_anomaly = np.array([1], dtype=np.float32)  # 標記有異常。

        sample = {  # 構建返回的樣本字典。
            'image': image,  # 圖像數據。
            'has_anomaly': has_anomaly,  # 異常標記。
            'mask': mask,  # 掩碼數據。
            'idx': idx  # 索引。
        }

        return sample  # 返回樣本。


class MVTecDRAEMTrainDataset(Dataset):  # 定義訓練數據集類，用於 DRAEM 模型訓練。

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):  # 初始化函數。
        """
        Args:
            root_dir (string): 包含所有圖像的目錄。
            transform (callable, optional): 可選的轉換操作。
        """
        self.root_dir = root_dir  # 保存根目錄。
        self.resize_shape = resize_shape  # 保存調整大小的形狀。

        self.image_paths = sorted(glob.glob(root_dir + "/*.png"))  # 獲取根目錄下所有 png 圖像路徑並排序。

        self.anomaly_source_paths = sorted(  # 獲取異常源圖像路徑（用於合成異常）。
            glob.glob(anomaly_source_path + "/*/*.jpg"))  # 假設異常源是 jpg 格式。

        self.augmenters = [  # 定義一系列圖像增強器，用於增加數據多樣性。
            iaa.GammaContrast((0.5, 2.0), per_channel=True),  # 調整 Gamma 對比度。
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),  # 調整亮度和加法。
            iaa.pillike.EnhanceSharpness(),  # 增強銳度。
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),  # 調整色調和飽和度。
            iaa.Solarize(0.5, threshold=(32, 128)),  # 曝光效果。
            iaa.Posterize(),  # 色調分離。
            iaa.Invert(),  # 反轉顏色。
            iaa.pillike.Autocontrast(),  # 自動對比度。
            iaa.pillike.Equalize(),  # 直方圖均衡化。
            iaa.Affine(rotate=(-45, 45))  # 仿射變換（旋轉）。
        ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])  # 定義旋轉增強器，用於旋轉 Perlin 噪聲。

    def __len__(self):  # 定義 __len__ 方法。
        return len(self.image_paths)  # 返回圖像數量。

    def randAugmenter(self):  # 定義隨機選擇增強器的方法。
        aug_ind = np.random.choice(np.arange(len(self.augmenters)),  # 從增強器列表中隨機選擇索引。
                                   3,  # 選擇 3 個。
                                   replace=False)  # 不重複選擇。
        aug = iaa.Sequential([  # 組合選中的增強器。
            self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]],  # 第一個、第二個增強器。
            self.augmenters[aug_ind[2]]  # 第三個增強器。
        ])
        return aug  # 返回組合後的增強器序列。

    def augment_image(self, image, anomaly_source_path):  # 定義圖像增強及異常合成方法。
        aug = self.randAugmenter()  # 獲取隨機增強器。
        perlin_scale = 6  #設置 Perlin 噪聲的最大尺度。
        min_perlin_scale = 0  # 設置 Perlin 噪聲的最小尺度。
        anomaly_source_img = cv2.imread(anomaly_source_path)  # 讀取異常源圖像。
        anomaly_source_img = cv2.resize(anomaly_source_img,  # 調整異常源圖像大小。
                                        dsize=(self.resize_shape[1],  # 目標寬度。
                                               self.resize_shape[0]))  # 目標高度。

        anomaly_img_augmented = aug(image=anomaly_source_img)  # 對異常源圖像應用增強。
        perlin_scalex = 2**(torch.randint(min_perlin_scale, perlin_scale,  # 隨機生成 X 方向的 Perlin 尺度。
                                          (1, )).numpy()[0])
        perlin_scaley = 2**(torch.randint(min_perlin_scale, perlin_scale,  # 隨機生成 Y 方向的 Perlin 尺度。
                                          (1, )).numpy()[0])

        perlin_noise = rand_perlin_2d_np(  # 生成 2D Perlin 噪聲。
            (self.resize_shape[0], self.resize_shape[1]),  # 噪聲形狀。
            (perlin_scalex, perlin_scaley))  # 噪聲尺度。
        perlin_noise = self.rot(image=perlin_noise)  # 對 Perlin 噪聲應用旋轉。
        threshold = 0.5  # 設置二值化閾值。
        perlin_thr = np.where(perlin_noise > threshold,  # 對 Perlin 噪聲進行二值化，生成掩碼。
                              np.ones_like(perlin_noise),  # 大於閾值設為 1。
                              np.zeros_like(perlin_noise))  # 小於閾值設為 0。
        perlin_thr = np.expand_dims(perlin_thr, axis=2)  # 擴展維度，適配圖像通道。

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0  # 使用掩碼提取增強後的異常源圖像部分。

        beta = torch.rand(1).numpy()[0] * 0.8  # 隨機生成混合係數 beta，最大為 0.8。

        augmented_image = image * (1 - perlin_thr) + (  # 將原圖與異常部分混合。
            1 - beta) * img_thr + beta * image * (perlin_thr)  # 使用 beta 進行透明度混合。

        no_anomaly = torch.rand(1).numpy()[0]  # 隨機決定是否生成無異常樣本。
        if no_anomaly > 0.5:  # 50% 機率不添加異常。
            image = image.astype(np.float32)  # 轉換圖像類型。
            return image, np.zeros_like(  # 返回原圖和全零掩碼。
                perlin_thr, dtype=np.float32), np.array([0.0],  # 異常標籤為 0。
                                                        dtype=np.float32)
        else:  # 50% 機率添加異常。
            augmented_image = augmented_image.astype(np.float32)  # 轉換增強後圖像類型。
            msk = (perlin_thr).astype(np.float32)  # 轉換掩碼類型。
            augmented_image = msk * augmented_image + (1 - msk) * image  # 確保異常區域覆蓋在原圖上。
            has_anomaly = 1.0  # 異常標籤為 1。
            if np.sum(msk) == 0:  # 如果掩碼和為 0（生成的異常區域為空）。
                has_anomaly = 0.0  # 標記為無異常。
            return augmented_image, msk, np.array([has_anomaly],  # 返回增強圖像、掩碼和標籤。
                                                  dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):  # 定義圖像轉換主流程。
        image = cv2.imread(image_path)  # 讀取圖像。
        image = cv2.resize(image,  # 調整圖像大小。
                           dsize=(self.resize_shape[1], self.resize_shape[0]))  # 目標尺寸。

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7  # 30% 機率對原圖進行旋轉增強。
        if do_aug_orig:  # 如果需要增強。
            image = self.rot(image=image)  # 旋轉圖像。

        image = np.array(image).reshape(  # 轉換並重塑圖像數組。
            (image.shape[0], image.shape[1], image.shape[2])).astype(
                np.float32) / 255.0  # 歸一化像素值。
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(  # 調用 augment_image 生成合成異常圖像。
            image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))  # 轉置增強圖像維度 (C, H, W)。
        image = np.transpose(image, (2, 0, 1))  # 轉置原圖像維度 (C, H, W)。
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))  # 轉置異常掩碼維度 (C, H, W)。
        return image, augmented_image, anomaly_mask, has_anomaly  # 返回處理後的數據。

    def __getitem__(self, idx):  # 定義 __getitem__ 方法。
        idx = torch.randint(0, len(self.image_paths), (1, )).item()  # 隨機選擇一張正常圖像的索引。
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths),  # 隨機選擇一張異常源圖像的索引。
                                           (1, )).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(  # 轉換圖像並生成異常。
            self.image_paths[idx],
            self.anomaly_source_paths[anomaly_source_idx])
        sample = {  # 構建返回樣本。
            'image': image,  # 原圖（可能經過旋轉）。
            "anomaly_mask": anomaly_mask,  # 異常掩碼。
            'augmented_image': augmented_image,  # 合成異常後的圖像。
            'has_anomaly': has_anomaly,  # 異常標籤。
            'idx': idx  # 索引。
        }

        return sample  # 返回樣本。


class MVTecDRAEMTestDataset(Dataset):  # 定義測試數據集類。

    def __init__(self, root_dir, category_name, resize_shape=None):  # 初始化函數。
        """
        Args:
            root_dir (string): MVTec dataset 的根目錄 (例如: "./mvtec")
            category_name (string): 物件類別名稱 (例如: "bottle")
            resize_shape (tuple): 圖像縮放的目標尺寸 (H, W)
        """
        self.root_dir = root_dir  # 保存根目錄。
        self.category_name = category_name  # 保存類別名稱。
        self.resize_shape = resize_shape  # 保存調整大小形狀。

        # 構建基礎路徑
        category_path = os.path.join(self.root_dir, self.category_name)  # 構建類別路徑。
        test_path = os.path.join(category_path, 'test')  # 構建測試集路徑。
        ground_truth_path = os.path.join(category_path, 'ground_truth')  # 構建 Ground Truth 路徑。

        self.image_paths = []  # 初始化圖像路徑列表。
        self.anomaly_masks_paths = []  # 儲存所有異常掩碼的路徑。
        self.is_anomaly_flags = []  # 記錄每個樣本是否為異常。

        # 1. 處理 'good' 圖像
        good_image_paths = sorted(  # 獲取 'good' 子目錄下的所有圖像。
            glob.glob(os.path.join(test_path, 'good', '*.png')))
        self.image_paths.extend(good_image_paths)  # 添加到圖像路徑列表。
        self.anomaly_masks_paths.extend([None] *  # 對應掩碼為 None。
                                        len(good_image_paths))  # 正常圖像沒有掩碼
        self.is_anomaly_flags.extend([0] * len(good_image_paths))  # 異常標記為 0。

        # 2. 處理異常圖像和對應的 ground_truth
        defect_types = sorted(os.listdir(os.path.join(test_path)))  # 獲取所有缺陷類型文件夾。
        defect_types = [  # 過濾掉 'good' 和非目錄項。
            d for d in defect_types
            if d != 'good' and os.path.isdir(os.path.join(test_path, d))
        ]

        for defect_type in defect_types:  # 遍歷每種缺陷類型。
            defect_image_paths = sorted(  # 獲取該缺陷類型下的所有圖像。
                glob.glob(os.path.join(test_path, defect_type, '*.png')))
            self.image_paths.extend(defect_image_paths)  # 添加到圖像路徑列表。
            self.is_anomaly_flags.extend([1] * len(defect_image_paths))  # 異常標記為 1。

            # 對於每個異常圖像，找到其對應的 ground_truth 掩碼
            for img_path in defect_image_paths:  # 遍歷每張缺陷圖像。
                img_filename = os.path.basename(img_path)  # 獲取文件名。
                # ****** 修改這裡：將 '_mask' 插入到檔案名中 ******
                # 假設檔案名格式為 'xxx.png'，我們希望變成 'xxx_mask.png'
                name_without_ext, ext = os.path.splitext(img_filename)  # 分離文件名和擴展名。
                gt_mask_filename = f"{name_without_ext}_mask{ext}"  # 構建對應的掩碼文件名。

                gt_mask_path = os.path.join(  # 構建完整的掩碼路徑。
                    ground_truth_path, defect_type,
                    gt_mask_filename)  # 使用新的帶 _mask 的檔案名。
                self.anomaly_masks_paths.append(gt_mask_path)  # 添加到掩碼路徑列表。

    def __len__(self):  # 定義 __len__ 方法。
        return len(self.image_paths)  # 返回圖像總數。

    def __getitem__(self, idx):  # 定義 __getitem__ 方法。
        image_path = self.image_paths[idx]  # 獲取圖像路徑。

        # 載入原始圖像
        image = cv2.imread(image_path)  # 讀取圖像。
        image = cv2.resize(image,  # 調整圖像大小。
                           dsize=(self.resize_shape[1], self.resize_shape[0]))  # 目標尺寸。
        image = np.array(image).reshape(  # 轉換並重塑圖像數組。
            (image.shape[0], image.shape[1], image.shape[2])).astype(
                np.float32) / 255.0  # 歸一化像素值。

        # 載入真實異常掩碼
        if self.is_anomaly_flags[idx] == 1:  # 如果是異常樣本。
            gt_mask_path = self.anomaly_masks_paths[idx]  # 獲取掩碼路徑。
            if os.path.exists(gt_mask_path):  # 如果掩碼文件存在。
                anomaly_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)  # 讀取掩碼。
                anomaly_mask = cv2.resize(anomaly_mask,  # 調整掩碼大小。
                                          dsize=(self.resize_shape[1],
                                                 self.resize_shape[0]))
                anomaly_mask = anomaly_mask / 255.0  # 二值化到 0-1 範圍。
            else:  # 如果掩碼文件不存在。
                print(  # 打印警告。
                    f"Warning: Ground truth mask not found for {image_path}. Using all zeros mask."
                )
                anomaly_mask = np.zeros(self.resize_shape, dtype=np.float32)  # 使用全零掩碼。
        else:  # 如果是正常樣本。
            anomaly_mask = np.zeros(self.resize_shape,
                                    dtype=np.float32)  # 正常圖像的掩碼是全零。

        # 將圖像和掩碼轉換為 PyTorch tensor 的格式 (C, H, W)
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        anomaly_mask = np.expand_dims(anomaly_mask,
                                      axis=0)  # (H, W) -> (1, H, W)

        # 為了與訓練集的 `sample` 結構一致，`augmented_image` 設為原始圖像
        augmented_image = image.copy()  # 複製圖像到 augmented_image。

        sample = {  # 構建返回樣本。
            'image': image,  # 原始圖像。
            "anomaly_mask": anomaly_mask,  # 異常掩碼。
            'augmented_image': augmented_image,  # 增強圖像（此處與原圖相同）。
            'has_anomaly': np.array([self.is_anomaly_flags[idx]],  # 異常標記。
                                    dtype=np.float32),
            'idx': idx  # 索引。
        }

        return sample  # 返回樣本。
