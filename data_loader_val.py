"""
MVTec AD 數據集加載模組

此模組定義了 `MVTecDataset` 類，用於加載 MVTec AD 自定義數據集。
它處理圖像讀取、預處理（調整大小、轉換為張量）以及標籤和掩碼的生成。
此模組還負責創建訓練和驗證數據加載器 (`train_loader`, `val_loader`)。

主要功能：
- MVTecDataset: 自定義 PyTorch Dataset 類。
- 數據預處理: 使用 torchvision.transforms 進行圖像和掩碼的調整及歸一化。
- 數據加載: 提供 DataLoader 實例以供模型訓練和評估使用。
"""

import os  # 導入 os 模組，用於與操作系統交互，例如路徑操作。

import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # 從 PyTorch 導入 DataLoader 和 Dataset，用於批次加載數據和定義數據集。
from torchvision import (
    transforms as T,
)  # 導入 torchvision.transforms 並重命名為 T，用於圖像預處理和增強。
from PIL import Image  # 從 PIL (Pillow) 導入 Image，用於圖像加載和處理。


class MVTecDataset(Dataset):
    """
    MVTec AD 數據集類，繼承自 torch.utils.data.Dataset。

    用於加載 MVTec AD (Anomaly Detection) 數據集，支持訓練集和測試集的讀取。
    對於測試集中的異常樣本，會同時加載對應的像素級標註掩碼 (Ground Truth Mask)。

    Args:
        root (str): 數據集根目錄路徑。
        category (str): 目標物體類別，例如 'bottle', 'cable' 等。預設為 'bottle'。
        split (str): 數據分割類型，'train' 用於訓練集，'test' 用於測試集。預設為 'train'。
        resize (int): 圖像調整大小的尺寸 (寬和高)。預設為 256。

    Attributes:
        img_dir (str): 圖像文件所在的目錄路徑。
        gt_dir (str): Ground Truth 掩碼所在的目錄路徑。
        data (list): 存儲所有圖像文件路徑的列表。
        labels (list): 存儲所有圖像對應標籤的列表 (0: 正常, 1: 異常)。
        masks (list): 存儲所有圖像對應掩碼路徑的列表 (正常樣本為 None)。
        transform (torchvision.transforms.Compose): 圖像預處理流程。
        mask_transform (torchvision.transforms.Compose): 掩碼預處理流程。
    """

    def __init__(
        self, root, category="bottle", split="train", resize=256
    ):  # 初始化函數，設置數據集路徑、類別、數據分割和圖像大小。
        self.root = root  # 保存數據集根目錄路徑。
        self.category = category  # 保存當前處理的類別（例如 "bottle"）。
        self.split = split  # 保存數據分割類型（"train" 或 "test"）。
        self.img_dir = os.path.join(root, category, split)  # 構建圖像目錄的路徑。
        self.gt_dir = os.path.join(
            root, category, "ground_truth"
        )  # 構建 Ground Truth（標註）目錄的路徑。

        self.data = []  # 初始化列表，用於存儲圖像路徑。
        self.labels = []  # 初始化列表，用於存儲標籤（0 表示正常，1 表示異常）。
        self.masks = []  # 初始化列表，用於存儲掩碼路徑。

        for defect_type in sorted(
            os.listdir(self.img_dir)
        ):  # 遍歷圖像目錄下的所有缺陷類型文件夾。
            img_folder = os.path.join(
                self.img_dir, defect_type
            )  # 構建特定缺陷類型的文件夾路徑。
            if not os.path.isdir(img_folder):  # 如果不是文件夾，則跳過。
                continue  # 跳過非文件夾項。
            img_files = sorted(
                os.listdir(img_folder)
            )  # 獲取該文件夾下的所有圖像文件並排序。
            for f in img_files:  # 遍歷所有圖像文件。
                img_path = os.path.join(img_folder, f)  # 構建圖像的完整路徑。
                if defect_type == "normal":  # 如果缺陷類型是 "normal"（正常）。
                    self.data.append(img_path)  # 將圖像路徑添加到數據列表。
                    self.labels.append(0)  # 添加標籤 0，表示正常樣本。
                    self.masks.append(None)  # 正常樣本沒有掩碼，添加 None。
                else:  # 如果是異常樣本。
                    mask_path = os.path.join(
                        self.gt_dir, defect_type, f.replace(".png", "_mask.png")
                    )  # 構建對應掩碼的路徑。
                    self.data.append(img_path)  # 將圖像路徑添加到數據列表。
                    self.labels.append(1)  # 添加標籤 1，表示異常樣本。
                    self.masks.append(mask_path)  # 將掩碼路徑添加到掩碼列表。

        self.transform = T.Compose(
            [  # 定義圖像預處理轉換序列。
                T.Resize((resize, resize)),  # 將圖像大小調整為指定的 resize 大小。
                T.ToTensor(),  # 將圖像轉換為 PyTorch 張量，並歸一化到 [0, 1]。
            ]
        )  # 結束圖像轉換定義。
        self.mask_transform = T.Compose(
            [  # 定義掩碼預處理轉換序列。
                T.Resize(
                    (resize, resize), interpolation=Image.NEAREST
                ),  # 將掩碼大小調整為指定的 resize 大小，使用最近鄰插值以保持二值性質。
                T.ToTensor(),  # 將掩碼轉換為 PyTorch 張量。
            ]
        )  # 結束掩碼轉換定義。

    def __len__(self):  # 定義 __len__ 方法，返回數據集的大小。
        return len(self.data)  # 返回數據列表的長度。

    def __getitem__(self, idx):  # 定義 __getitem__ 方法，用於根據索引獲取數據樣本。
        img_path = self.data[idx]  # 根據索引獲取圖像路徑。
        label = self.labels[idx]  # 根據索引獲取標籤。
        mask_path = self.masks[idx]  # 根據索引獲取掩碼路徑。

        img = Image.open(img_path).convert(
            "RGB"
        )  # 打開圖像並轉換為 RGB 模式，確保通道數一致。
        img = self.transform(img)  # 對圖像應用預定義的轉換。

        if mask_path is None:  # 如果掩碼路徑為 None（正常樣本）。
            mask = torch.zeros(
                (1, img.shape[1], img.shape[2])
            )  # 創建全零掩碼，形狀與圖像一致，表示無異常區域。
        else:  # 如果有掩碼路徑（異常樣本）。
            mask = Image.open(mask_path).convert("L")  # 打開掩碼圖像並轉換為灰度模式。
            mask = self.mask_transform(mask)  # 對掩碼應用預定義的轉換。
            mask = (
                mask > 0.5
            ).float()  # 將掩碼二值化（大於 0.5 為 1，否則為 0），並轉換為浮點型。

        return img, (
            torch.tensor(label, dtype=torch.long),
            mask,
        )  # 返回圖像及其對應的標籤與掩碼（作為元組）。


# === train_loader & val_loader ===
train_dataset = MVTecDataset(
    root="./mvtec_ad", category="bottle", split="train", resize=256
)  # 實例化訓練數據集，指定路徑、類別、分割和大小。
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4
)  # 創建訓練數據加載器，設置批量大小、打亂數據和工作線程數。

val_dataset = MVTecDataset(
    root="./mvtec_ad", category="bottle", split="test", resize=256
)  # 實例化驗證數據集（測試集），指定路徑、類別、分割和大小。
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, num_workers=4
)  # 創建驗證數據加載器，設置批量大小、不打亂數據和工作線程數。

print("Train samples:", len(train_dataset))  # 打印訓練集樣本數量，用於確認加載正確。
print("Val samples:", len(val_dataset))  # 打印驗證集樣本數量，用於確認加載正確。
