"""
Anomaly Detection Training Script
異常檢測訓練腳本

此腳本實現了基於學生-教師網絡架構（Student-Teacher Architecture）的異常檢測模型訓練流程。
主要功能包括：
1. 數據加載：加載訓練和測試數據集（MVTec AD）。
2. 模型初始化：建立教師模型（重建與分割）和學生模型。
3. 訓練循環：執行模型訓練，計算重建損失、分割損失和蒸餾損失。
4. 驗證與評估：在測試集上評估模型性能，計算 AUROC、F1 Score 等指標。
5. 可視化：保存預測結果和診斷圖像。

Usage:
    python main.py --gpu_id <gpu_id> --obj_id <obj_id> --lr <learning_rate> \
        --bs <batch_size> --epochs <epochs>
"""

import os  # 導入 os 模組，用於與操作系統交互。
import random  # 導入 random 模組，用於生成隨機數。
import argparse  # 導入 argparse，用於解析命令行參數。

import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
from torch import optim  # 導入 PyTorch 的優化器模組。
from torch.utils.data import DataLoader  # 導入 DataLoader，用於批次加載數據。
import torch.nn.functional as F  # 導入 PyTorch 的函數式接口。
import numpy as np  # 導入 numpy，用於數值計算。
import matplotlib.pyplot as plt  # 導入 matplotlib.pyplot 用於繪圖。
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
)  # 導入更多評估指標。

from loss import FocalLoss, SSIM  # 從 loss 模組導入 FocalLoss 和 SSIM 損失函數。
from model_unet import (
    ReconstructiveSubNetwork,
    DiscriminativeSubNetwork,
)  # 從 model_unet 導入子網絡模型。
from data_loader import (
    MVTecDRAEMTrainDataset,
    MVTecDRAEMTestDataset,
)  # 從 data_loader 導入訓練和測試數據集類。


def setup_seed(seed):
    """
    設定全域隨機種子，確保實驗結果的可重複性 (Reproducibility)。

    此函數會統一設定 PyTorch (CPU & GPU)、NumPy 以及 Python 內建 random 的亂數種子。
    同時強制 cuDNN 使用確定性算法 (deterministic)，避免因並行計算優化導致結果有些微差異。

    Args:
        seed (int): 隨機種子數值 (例如: 42, 111)。
    """
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)  # 設置 CPU 的隨機種子。
    torch.cuda.manual_seed_all(seed)  # 設置所有 GPU 的隨機種子。
    np.random.seed(seed)  # 設置 numpy 的隨機種子。
    random.seed(seed)  # 設置 random 模組的隨機種子。
    torch.backends.cudnn.deterministic = (
        True  # 設置 cuDNN 為確定性模式，保證結果可重現。
    )
    torch.backends.cudnn.benchmark = (
        False  # 關閉 cuDNN 的自動最佳化搜尋，避免不同硬件上結果不一致。
    )


# =======================
# Utilities
# =======================
def get_available_gpu():  # 定義獲取可用 GPU 的函數。
    """自動選擇記憶體使用率最低的GPU"""
    if not torch.cuda.is_available():  # 如果沒有 GPU 可用。
        return -1  # 返回 -1，表示使用 CPU。

    gpu_count = torch.cuda.device_count()  # 獲取 GPU 數量。
    if gpu_count == 0:  # 如果數量為 0。
        return -1  # 返回 -1。

    # 檢查每個GPU的記憶體使用情況
    gpu_memory = []  # 初始化列表，存儲 GPU 內存信息。
    for i in range(gpu_count):  # 遍歷每個 GPU。
        torch.cuda.set_device(i)  # 設置當前設備。
        memory_allocated = torch.cuda.memory_allocated(i)  # 獲取已分配內存。
        memory_reserved = torch.cuda.memory_reserved(i)  # 獲取保留內存。
        gpu_memory.append((i, memory_allocated, memory_reserved))  # 添加到列表。

    # 選擇記憶體使用最少的GPU
    available_gpu = min(gpu_memory, key=lambda x: x[1])[
        0
    ]  # 根據已分配內存排序，選擇最小的那個 GPU ID。
    return available_gpu  # 返回 GPU ID。


def weights_init(m):  # 定義權重初始化函數。
    """卷積層權重 → 小亂數讓網路容易學習、梯度穩定
    BatchNorm權重 → 初始縮放 1 保持數值穩定，偏置 0 不改變均值"""
    # 取得模組的類別名稱，例如 'Conv2d', 'BatchNorm2d' 等
    classname = m.__class__.__name__  # 獲取類名。

    # 如果是卷積層 (Conv)
    if classname.find("Conv") != -1:  # 如果類名包含 'Conv'。
        # 權重初始化為均值 0、標準差 0.02 的高斯分布
        # 原因：這樣可以讓卷積層一開始輸出的特徵值分布均衡，避免梯度消失或梯度爆炸
        m.weight.data.normal_(0.0, 0.02)  # 使用正態分佈初始化權重。

    # 如果是批次正規化層 (BatchNorm)
    elif classname.find("BatchNorm") != -1:  # 如果類名包含 'BatchNorm'。
        # 權重初始化為均值 1、標準差 0.02 的高斯分布
        # 原因：BatchNorm 的權重 (gamma) 控制輸出縮放，初始化為 1 可以保持初始特徵分布不變
        m.weight.data.normal_(1.0, 0.02)  # 初始化權重為 1。
        # 偏置初始化為 0
        # 原因：偏置 (beta) 控制輸出偏移量，初始化為 0 可以保持輸出均值不偏移
        m.bias.data.fill_(0)  # 初始化偏置為 0。


def visualize_predictions(
    teacher_model,
    teacher_seg_model,
    student_model,  # 定義可視化預測結果的函數。
    student_seg_model,
    batch,
    device,
    save_path,
):
    """
    視覺化教師模型和學生模型的預測結果對比
    """
    teacher_model.eval()  # 將教師模型設置為評估模式。
    student_model.eval()  # 將學生模型設置為評估模式。
    with torch.no_grad():  # 禁用梯度計算。
        input_image = batch["image"].to(device)  # 獲取輸入圖像並移動到設備。
        aug_image = batch["augmented_image"].to(device)  # 獲取增強圖像並移動到設備。
        gt_mask = batch["anomaly_mask"].to(device)  # 獲取異常掩碼並移動到設備。

        # 教師預測
        teacher_recon = teacher_model(aug_image)  # 教師模型重建圖像。
        teacher_joined_in = torch.cat(
            (teacher_recon, aug_image), dim=1
        )  # 將重建圖像與增強圖像拼接。
        teacher_seg_out_mask = teacher_seg_model(
            teacher_joined_in
        )  # 教師分割模型輸出。
        teacher_seg_map = torch.softmax(
            teacher_seg_out_mask, dim=1
        )  # 計算 softmax 概率。

        # 學生預測
        student_recon = student_model(aug_image)  # 學生模型重建圖像。
        student_joined_in = torch.cat(
            (student_recon, aug_image), dim=1
        )  # 將重建圖像與增強圖像拼接。
        student_seg_out_mask = student_seg_model(
            student_joined_in
        )  # 學生分割模型輸出。
        student_seg_map = torch.softmax(
            student_seg_out_mask, dim=1
        )  # 計算 softmax 概率。

        # 轉換為 numpy 用於繪圖
        input_np = (
            input_image.cpu().numpy()[0].transpose(1, 2, 0)
        )  # 轉換輸入圖像為 numpy 格式 (H, W, C)。
        aug_np = (
            aug_image.cpu().numpy()[0].transpose(1, 2, 0)
        )  # 轉換增強圖像為 numpy 格式。
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]  # 取第一個通道的 Ground Truth 掩碼。

        # 處理分割結果
        teacher_seg_np = (
            torch.softmax(teacher_seg_map, dim=1)[0, 1].cpu().numpy()
        )  # 獲取教師模型預測的異常類別概率。
        student_seg_np = (
            torch.softmax(student_seg_map, dim=1)[0, 1].cpu().numpy()
        )  # 獲取學生模型預測的異常類別概率。

        # 創建圖像
        _, axes = plt.subplots(
            2, 4, figsize=(20, 10)
        )  # 創建 2x4 的子圖佈局，使用 _ 忽略未使用的 figure 物件。

        # 第一行：原始輸入與標註
        axes[0, 0].imshow(input_np)  # 顯示原始輸入影像。
        axes[0, 0].set_title("Original Image")  # 設置標題。
        axes[0, 0].axis("off")  # 關閉坐標軸。
        axes[0, 1].imshow(aug_np)  # 顯示增強後影像。
        axes[0, 1].set_title("Augmented Image")  # 設置標題。
        axes[0, 1].axis("off")  # 關閉坐標軸。
        axes[0, 2].imshow(gt_mask_np, cmap="jet")  # 顯示 Ground Truth 異常遮罩。
        axes[0, 2].set_title("Ground Truth Mask")  # 設置標題。
        axes[0, 2].axis("off")  # 關閉坐標軸。
        axes[0, 3].axis("off")  # 第四個位置留白。

        # 第二行：模型預測與比較
        im1 = axes[1, 0].imshow(
            teacher_seg_np, cmap="jet", vmin=0, vmax=1
        )  # 顯示教師模型的預測熱力圖。
        axes[1, 0].set_title("Teacher Segmentation")  # 設置標題。
        axes[1, 0].axis("off")  # 關閉坐標軸。
        plt.colorbar(im1, ax=axes[1, 0])  # 添加顏色條。

        im2 = axes[1, 1].imshow(
            student_seg_np, cmap="jet", vmin=0, vmax=1
        )  # 顯示學生模型的預測熱力圖。
        axes[1, 1].set_title("Student Segmentation")  # 設置標題。
        axes[1, 1].axis("off")  # 關閉坐標軸。
        plt.colorbar(im2, ax=axes[1, 1])  # 添加顏色條。
        # 差異圖
        diff = np.abs(teacher_seg_np - student_seg_np)  # 計算教師與學生預測的絕對差異。
        im3 = axes[1, 2].imshow(diff, cmap="hot", vmin=0, vmax=1)  # 顯示差異圖。
        axes[1, 2].set_title("Teacher-Student Difference")  # 設置標題。
        axes[1, 2].axis("off")  # 關閉坐標軸。
        plt.colorbar(im3, ax=axes[1, 2])  # 添加顏色條。
        # 二值化對比
        student_binary = (student_seg_np > 0.5).astype(
            np.float32
        )  # 對學生預測進行二值化。
        axes[1, 3].imshow(student_binary, cmap="gray")  # 顯示二值化結果。
        axes[1, 3].set_title("Student Binary (>0.5)")  # 設置標題。
        axes[1, 3].axis("off")  # 關閉坐標軸。

        plt.tight_layout()  # 自動調整佈局。
        plt.savefig(save_path + ".png", dpi=150, bbox_inches="tight")  # 保存圖像。
        plt.close()  # 關閉圖像以釋放內存。
        student_model.train()  # 將學生模型切換回訓練模式。
        print(f"✅ Visualization saved: {save_path}.png")  # 打印保存路徑。


def detailed_diagnostic_visualization(
    teacher_model,
    teacher_seg_model,  # 定義詳細診斷可視化的函數。
    student_model,
    student_seg_model,
    loss_focal,
    batch,
    device,
    save_path,
    epoch,
    i_batch,
):
    """
    更詳細的診斷視覺化，包含損失值和指標
    """
    teacher_model.eval()  # 教師模型設為評估模式。
    student_model.eval()  # 學生模型設為評估模式。
    with torch.no_grad():  # 禁用梯度。
        input_image = batch["image"].to(device)  # 獲取輸入數據。
        aug_image = batch["augmented_image"].to(device)  # 獲取增強數據。
        gt_mask = batch["anomaly_mask"].to(device)  # 獲取掩碼。

        # 獲取預測
        # 教師預測
        teacher_recon = teacher_model(aug_image)  # 教師重建。
        teacher_joined_in = torch.cat((teacher_recon, aug_image), dim=1)  # 教師拼接。
        teacher_seg_out_mask = teacher_seg_model(teacher_joined_in)  # 教師分割。
        teacher_seg_map = torch.softmax(teacher_seg_out_mask, dim=1)  # 教師概率。

        # 學生預測
        student_recon = student_model(aug_image)  # 學生重建。
        student_joined_in = torch.cat((student_recon, aug_image), dim=1)  # 學生拼接。
        student_seg_out_mask = student_seg_model(student_joined_in)  # 學生分割。
        student_seg_map = torch.softmax(student_seg_out_mask, dim=1)  # 學生概率。

        # 計算當前損失（僅用於顯示）
        seg_distill_loss = F.mse_loss(
            student_seg_map, teacher_seg_map
        ).item()  # 計算分割蒸餾損失。
        student_seg_softmax = torch.softmax(
            student_seg_map, dim=1
        )  # 再次計算 softmax (重複操作，可優化)。
        orig_seg_loss = loss_focal(
            student_seg_softmax, gt_mask
        ).item()  # 計算原始分割損失。

        # 轉換為 numpy
        input_np = input_image.cpu().numpy()[0].transpose(1, 2, 0)  # 轉為 numpy。
        aug_np = aug_image.cpu().numpy()[0].transpose(1, 2, 0)  # 轉為 numpy。
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]  # 轉為 numpy。
        teacher_seg_np = (
            torch.softmax(teacher_seg_map, dim=1)[0, 1].cpu().numpy()
        )  # 教師預測轉為 numpy。
        student_seg_np = (
            torch.softmax(student_seg_map, dim=1)[0, 1].cpu().numpy()
        )  # 學生預測轉為 numpy。

        # 創建診斷圖
        _, axes = plt.subplots(
            2, 5, figsize=(25, 10)
        )  # 創建 2x5 子圖，使用 _ 忽略未使用的 figure 物件。

        # 第一行：輸入與預測
        axes[0, 0].imshow(input_np)  # 顯示原始影像。
        axes[0, 0].set_title("Original Image")  # 設置標題。
        axes[0, 0].axis("off")
        axes[0, 1].imshow(aug_np)  # 顯示增強影像。
        axes[0, 1].set_title("Augmented Image")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(gt_mask_np, cmap="jet")  # 顯示 Ground Truth 異常遮罩。
        axes[0, 2].set_title("GT Mask")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(
            teacher_seg_np, cmap="jet", vmin=0, vmax=1
        )  # 顯示教師模型預測。
        axes[0, 3].set_title(
            f"Teacher Seg\nMax: {teacher_seg_np.max():.3f}"
        )  # 顯示最大值。
        axes[0, 3].axis("off")
        axes[0, 4].imshow(
            student_seg_np, cmap="jet", vmin=0, vmax=1
        )  # 顯示學生模型預測。
        axes[0, 4].set_title(
            f"Student Seg\nMax: {student_seg_np.max():.3f}"
        )  # 顯示最大值。
        axes[0, 4].axis("off")

        # 第二行：分析和差異
        diff = np.abs(teacher_seg_np - student_seg_np)  # 計算差異。
        axes[1, 0].imshow(diff, cmap="hot")  # 顯示差異圖。
        axes[1, 0].set_title(f"Difference\nAvg: {diff.mean():.3f}")  # 顯示平均差異。
        axes[1, 0].axis("off")
        # 學生二值化
        student_binary = (student_seg_np > 0.5).astype(np.float32)  # 學生預測二值化。
        axes[1, 1].imshow(student_binary, cmap="gray")  # 顯示學生二值化結果。
        axes[1, 1].set_title("Student Binary\n(>0.5)")
        axes[1, 1].axis("off")
        # 教師二值化
        teacher_binary = (teacher_seg_np > 0.5).astype(np.float32)  # 教師預測二值化。
        axes[1, 2].imshow(teacher_binary, cmap="gray")  # 顯示教師二值化結果。
        axes[1, 2].set_title("Teacher Binary\n(>0.5)")
        axes[1, 2].axis("off")

        # 損失信息
        # 顯示目前訓練週期與批次編號，以及兩種損失值：
        # - seg_distill_loss：學生模仿教師的損失
        # - orig_seg_loss：學生對 Ground Truth 的預測損失
        axes[1, 3].text(
            0.1,
            0.7,
            f"Epoch: {epoch}\nBatch: {i_batch}\n\n"
            f"Seg Distill Loss: {seg_distill_loss:.4f}\n"
            f"Orig Seg Loss: {orig_seg_loss:.4f}",
            fontsize=12,
        )  # 在圖上顯示文本信息。
        axes[1, 3].axis("off")

        # 統計信息
        # 顯示教師與學生模型的統計資訊（平均值與標準差），
        # 用於分析模型預測的穩定性與分佈特性：
        # - Mean：代表整體異常機率的平均強度
        # - Std：代表預測分佈的離散程度，越高表示模型預測越不穩定
        axes[1, 4].text(
            0.1,
            0.7,
            f"Teacher Stats:\n"
            f"Mean: {teacher_seg_np.mean():.3f}\n"
            f"Std: {teacher_seg_np.std():.3f}\n\n"
            f"Student Stats:\n"
            f"Mean: {student_seg_np.mean():.3f}\n"
            f"Std: {student_seg_np.std():.3f}",
            fontsize=12,
        )  # 顯示統計信息。
        axes[1, 4].axis("off")

        plt.tight_layout()  # 調整佈局。
        plt.savefig(
            save_path + "_diagnostic.png", dpi=150, bbox_inches="tight"
        )  # 保存圖片。
        plt.close()  # 關閉圖片。
        student_model.train()  # 恢復訓練模式。
        print(
            f"✅ Diagnostic visualization saved: {save_path}_diagnostic.png"
        )  # 打印保存信息。


# =======================
# Main Pipeline
# =======================
def load_teacher_models(obj_name, device):
    """
    載入教師模型 (Teacher Model) 的重建與分割網路。

    Args:
        obj_name (str): 目標類別名稱。
        device (str): 運算裝置。

    Returns:
        tuple: (teacher_model, teacher_seg_model)
    """
    # 構建教師重建模型路徑
    # 使用 f-string 直接嵌入變數，修正無插值變數的警告
    recon_path = (
        f"./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_{obj_name}_"
    )
    checkpoint_path = recon_path + ".pckl"
    teacher_recon_ckpt = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    img_channels = 3
    # 建立教師models的結構
    teacher_model = ReconstructiveSubNetwork(
        in_channels=img_channels,
        out_channels=img_channels,
        base_width=128,
    ).to(device)

    # 載入權重並設為評估模式
    teacher_model.load_state_dict(teacher_recon_ckpt, strict=True)
    teacher_model.eval()

    # 將教師模型的所有參數設為不可訓練
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 構建教師分割模型路徑
    # 使用 f-string 直接嵌入變數，修正無插值變數的警告
    seg_path = (
        f"./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_{obj_name}__seg"
    )
    checkpoint_seg_path = seg_path + ".pckl"
    teacher_seg_ckpt = torch.load(
        checkpoint_seg_path, map_location=device, weights_only=True
    )

    img_seg_channels = 6
    img_seg_channels_out = 2
    teacher_seg_model = DiscriminativeSubNetwork(
        in_channels=img_seg_channels,
        out_channels=img_seg_channels_out,
        base_channels=64,
        out_features=False,
    ).to(device)

    teacher_seg_model.load_state_dict(teacher_seg_ckpt, strict=True)
    teacher_seg_model.eval()

    return teacher_model, teacher_seg_model


def create_student_models(device):
    """
    初始化學生模型 (Student Model) 的重建與分割網路。

    Args:
        device (str): 運算裝置。

    Returns:
        tuple: (student_model, student_seg_model)
    """
    img_channels = 3
    # 學生重建網路較窄
    student_model = ReconstructiveSubNetwork(
        in_channels=img_channels,
        out_channels=img_channels,
        base_width=64,
    ).to(device)
    student_model.apply(weights_init)

    img_seg_channels = 6
    img_seg_channels_out = 2
    # 學生分割網路較窄
    student_seg_model = DiscriminativeSubNetwork(
        in_channels=img_seg_channels,
        out_channels=img_seg_channels_out,
        base_channels=32,
        out_features=False,
    ).to(device)
    student_seg_model.apply(weights_init)

    return student_model, student_seg_model


def prepare_dataloaders(obj_name, args):
    """
    準備訓練與驗證資料加載器。

    Args:
        obj_name (str): 目標類別名稱。
        args (Namespace): 參數物件。

    Returns:
        tuple: (train_loader, val_loader)
    """
    path = "./mvtec"
    path_dtd = "./dtd/images/"

    # 載入訓練資料集
    train_dataset = MVTecDRAEMTrainDataset(
        root_dir=path + f"/{obj_name}/train/good/",
        anomaly_source_path=path_dtd,
        resize_shape=[256, 256],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, num_workers=4
    )

    # 載入驗證資料集
    val_dataset = MVTecDRAEMTestDataset(
        root_dir=path,
        category_name=obj_name,
        resize_shape=[256, 256],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def train_epoch(models, loaders, optimizer, losses, epoch, args, device):
    """
    執行單個訓練 Epoch。

    Args:
        models (dict): 包含教師與學生模型。
        loaders (dict): 包含訓練加載器。
        optimizer (Optimizer): 優化器。
        losses (dict): 損失函數集合。
        epoch (int): 當前 Epoch。
        args (Namespace): 參數物件。
        device (str): 運算裝置。

    Returns:
        tuple: (avg_total_loss, avg_orig_seg_loss)
    """
    teacher_model = models["teacher"]
    teacher_seg_model = models["teacher_seg"]
    student_model = models["student"]
    student_seg_model = models["student_seg"]
    train_loader = loaders["train"]
    loss_focal = losses["focal"]
    loss_ssim = losses["ssim"]

    student_model.train()
    student_seg_model.train()

    # 超參數定義
    lambda_l2 = 1.0
    lambda_ssim = 0.5
    lambda_segment = 2.0
    lambda_recon_distill = 0.3
    lambda_seg_distill = 0.7

    # 動態蒸餾權重
    if epoch < 10:
        lambda_distill = 0.0
    else:
        lambda_distill = 0.5 * (epoch / args.epochs)

    epoch_loss = 0.0
    epoch_orig_seg_loss = 0.0
    num_batches = 0

    for sample_batched in train_loader:
        input_image = sample_batched["image"].to(device)
        ground_truth_mask = sample_batched["anomaly_mask"].to(device).float()
        aug_gray_batch = sample_batched["augmented_image"].to(device)
        is_normal = sample_batched.get(
            "is_normal", torch.ones_like(ground_truth_mask)
        ).bool()

        # 教師網路前向
        with torch.no_grad():
            teacher_recon = teacher_model(aug_gray_batch)
            teacher_joined_in = torch.cat((teacher_recon, aug_gray_batch), dim=1)
            teacher_out_mask = teacher_seg_model(teacher_joined_in)
            teacher_seg_map = torch.softmax(teacher_out_mask, dim=1)

        # 學生網路前向
        student_recon = student_model(aug_gray_batch)
        student_joined_in = torch.cat((student_recon, aug_gray_batch), dim=1)
        student_out_mask = student_seg_model(student_joined_in)
        student_seg_map = torch.softmax(student_out_mask, dim=1)

        # 計算重建損失
        l2_map = (student_recon - input_image) ** 2
        weighted_l2 = (l2_map * (1 - ground_truth_mask)).mean()

        ssim_map = (
            loss_ssim(student_recon, input_image, return_map=True)
            if hasattr(loss_ssim, "return_map")
            else loss_ssim(student_recon, input_image)
        )
        if isinstance(ssim_map, torch.Tensor) and ssim_map.ndim > 0:
            weighted_ssim = (ssim_map * (1 - ground_truth_mask)).mean()
        else:
            weighted_ssim = ssim_map

        loss_recon = lambda_l2 * weighted_l2 + lambda_ssim * weighted_ssim

        # 計算分割損失
        segment_loss = loss_focal(student_seg_map, ground_truth_mask)
        loss_seg = lambda_segment * segment_loss
        epoch_orig_seg_loss += loss_seg.item()

        # 計算蒸餾損失
        recon_distill_loss = F.mse_loss(student_recon, teacher_recon)
        seg_distill_loss = F.mse_loss(student_seg_map, teacher_seg_map)
        distill_loss = (
            lambda_recon_distill * recon_distill_loss
            + lambda_seg_distill * seg_distill_loss
        )

        distill_mask = is_normal.float().mean()
        loss_distill = lambda_distill * distill_mask * distill_loss

        # 總損失
        if epoch < 10:
            total_loss = loss_recon + loss_seg
        else:
            total_loss = loss_recon + loss_seg + loss_distill

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        num_batches += 1

    avg_total_loss = epoch_loss / num_batches if num_batches > 0 else 0
    avg_orig_seg_loss = epoch_orig_seg_loss / num_batches if num_batches > 0 else 0

    return avg_total_loss, avg_orig_seg_loss


def validate_epoch(models, val_loader, device):
    """
    執行驗證 Epoch，計算各項指標。

    Args:
        models (dict): 包含學生模型。
        val_loader (DataLoader): 驗證加載器。
        device (str): 運算裝置。

    Returns:
        dict: 包含各項指標的字典。
    """
    student_model = models["student"]
    student_seg_model = models["student_seg"]

    student_model.eval()
    student_seg_model.eval()

    all_pred_masks = []
    all_gt_masks = []

    with torch.no_grad():
        for sample_batched_val in val_loader:
            input_image_val = sample_batched_val["image"].to(device)
            ground_truth_mask_val = (
                sample_batched_val["anomaly_mask"].to(device).float()
            )

            student_recon = student_model(input_image_val)
            student_joined_in = torch.cat((student_recon, input_image_val), dim=1)
            student_seg_out_mask = student_seg_model(student_joined_in)
            student_seg_map = torch.softmax(student_seg_out_mask, dim=1)
            student_seg_map_val = student_seg_map[:, 1:2, :, :]

            all_pred_masks.append(student_seg_map_val.cpu().numpy())
            all_gt_masks.append(ground_truth_mask_val.cpu().numpy())

    all_pred_masks = np.concatenate(all_pred_masks, axis=0)
    all_gt_masks = np.concatenate(all_gt_masks, axis=0)
    all_pred_masks_flat = all_pred_masks.flatten()
    all_gt_masks_flat = all_gt_masks.flatten().astype(int)

    metrics = {}
    try:
        fpr, tpr, _ = roc_curve(all_gt_masks_flat, all_pred_masks_flat)
        metrics["pixel_auroc"] = auc(fpr, tpr)
    except ValueError:
        metrics["pixel_auroc"] = float("nan")

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(
            all_gt_masks_flat, all_pred_masks_flat
        )
        metrics["pixel_pr_auc"] = auc(recall_curve, precision_curve)
    except ValueError:
        metrics["pixel_pr_auc"] = float("nan")

    binary_pred = (all_pred_masks_flat > 0.5).astype(int)
    metrics["precision"] = precision_score(
        all_gt_masks_flat, binary_pred, zero_division=0
    )
    metrics["recall"] = recall_score(all_gt_masks_flat, binary_pred, zero_division=0)
    metrics["f1"] = f1_score(all_gt_masks_flat, binary_pred, zero_division=0)
    metrics["iou"] = jaccard_score(all_gt_masks_flat, binary_pred, zero_division=0)

    return metrics


def main(obj_names, args):
    """
    執行異常檢測模型的主要訓練流程。

    Args:
        obj_names (list): 要訓練的目標類別名稱列表。
        args (argparse.Namespace): 包含訓練參數的命名空間對象。
    """
    setup_seed(111)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for obj_name in obj_names:
        # 1. 載入教師模型
        teacher_model, teacher_seg_model = load_teacher_models(obj_name, device)

        # 2. 初始化學生模型
        student_model, student_seg_model = create_student_models(device)

        # 3. 準備優化器與調度器
        optimizer = torch.optim.Adam(
            [
                {"params": student_model.parameters(), "lr": args.lr},
                {"params": student_seg_model.parameters(), "lr": args.lr},
            ]
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            [args.epochs * 0.8, args.epochs * 0.9],
            gamma=0.2,
            last_epoch=-1,
        )

        # 4. 準備資料
        train_loader, val_loader = prepare_dataloaders(obj_name, args)

        # 5. 準備損失函數
        loss_focal = FocalLoss()
        loss_ssim = SSIM()
        losses = {"focal": loss_focal, "ssim": loss_ssim}

        # 6. 準備儲存路徑
        save_root = "./save_files"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        checkpoint_dir = os.path.join(save_root, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_pixel_auroc = 0.0

        # 模型字典，方便傳遞
        models = {
            "teacher": teacher_model,
            "teacher_seg": teacher_seg_model,
            "student": student_model,
            "student_seg": student_seg_model,
        }
        loaders = {"train": train_loader, "val": val_loader}

        for epoch in range(args.epochs):
            print("Epoch:", epoch)

            # 7. 訓練一個 Epoch
            avg_total_loss, avg_orig_seg_loss = train_epoch(
                models, loaders, optimizer, losses, epoch, args, device
            )

            scheduler.step()

            print("-" * 50)
            print(f"Epoch {epoch} Summary:")
            print(f"  - Average Total Loss    : {avg_total_loss:.6f}")
            print(f"  - Average Seg Loss      : {avg_orig_seg_loss:.6f}")
            print("-" * 50)

            # 8. 驗證與評估
            if val_loader:
                metrics = validate_epoch(models, val_loader, device)
                pixel_auroc = metrics["pixel_auroc"]

                print("-" * 50)
                print(f"Epoch {epoch} Anomaly Detection Metrics:")
                print(f"  - Pixel-level AUROC   : {metrics['pixel_auroc']:.4f}")
                print(f"  - Pixel-level PR-AUC  : {metrics['pixel_pr_auc']:.4f}")
                print(f"  - Pixel-level Precision: {metrics['precision']:.4f}")
                print(f"  - Pixel-level Recall  : {metrics['recall']:.4f}")
                print(f"  - Pixel-level F1 Score: {metrics['f1']:.4f}")
                print(f"  - Pixel-level IoU     : {metrics['iou']:.4f}")
                print("-" * 50)

                # 恢復訓練模式
                student_model.train()
                student_seg_model.train()

                # 9. 儲存最佳模型
                if not np.isnan(pixel_auroc) and pixel_auroc > best_pixel_auroc:
                    best_pixel_auroc = pixel_auroc
                    save_path = os.path.join(
                        checkpoint_dir, f"{obj_name}_best_recon.pckl"
                    )
                    save_seg_path = os.path.join(
                        checkpoint_dir, f"{obj_name}_best_seg.pckl"
                    )
                    torch.save(student_model.state_dict(), save_path)
                    torch.save(student_seg_model.state_dict(), save_seg_path)
                    print(
                        f"✅ New best model saved at epoch {epoch} (Pixel AUROC={best_pixel_auroc:.4f})"
                    )

        torch.cuda.empty_cache()


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    # --gpu_id -2：自動選擇最佳GPU
    # --gpu_id -1：強制使用CPU
    # --gpu_id  0：使用GPU 0（原有行為）

    parser = argparse.ArgumentParser()  # 創建參數解析器。
    parser.add_argument(
        "--obj_id", action="store", type=int, required=True
    )  # 添加 obj_id 參數。
    parser.add_argument("--epochs", default=25, type=int)  # 添加 epochs 參數。
    parser.add_argument(
        "--bs", action="store", type=int, required=True
    )  # 添加 batch size 參數。
    parser.add_argument(
        "--lr", action="store", type=float, required=True
    )  # 添加 learning rate 參數。
    parser.add_argument(
        "--gpu_id",
        action="store",
        type=int,
        default=-2,
        required=False,
        help="GPU ID (-2: auto-select, -1: CPU)",
    )  # 添加 gpu_id 參數。
    # 解析參數並命名為 parsed_args 以避免與函數參數 args 衝突
    parsed_args = parser.parse_args()

    # 自動選擇GPU
    if parsed_args.gpu_id == -2:  # 自動選擇模式
        parsed_args.gpu_id = get_available_gpu()  # 獲取可用 GPU。
        print(f"自動選擇 GPU: {parsed_args.gpu_id}")  # 打印選擇結果。

    obj_batch = [
        ["capsule"],
        ["bottle"],
        ["carpet"],
        ["leather"],
        ["pill"],  # 定義目標類別列表。
        ["transistor"],
        ["tile"],
        ["cable"],
        ["zipper"],
        ["toothbrush"],
        ["metal_nut"],
        ["hazelnut"],
        ["screw"],
        ["grid"],
        ["wood"],
    ]

    if int(parsed_args.obj_id) == -1:  # 如果 obj_id 為 -1，選擇所有類別。
        obj_list = [
            "capsule",
            "bottle",
            "carpet",
            "leather",
            "pill",
            "transistor",
            "tile",
            "cable",
            "zipper",
            "toothbrush",
            "metal_nut",
            "hazelnut",
            "screw",
            "grid",
            "wood",
        ]
        picked_classes = obj_list
    else:  # 否則選擇指定類別。
        picked_classes = obj_batch[int(parsed_args.obj_id)]

    # 根據選擇的GPU執行
    if parsed_args.gpu_id == -1:  # 如果使用 CPU。
        # 使用CPU
        main(picked_classes, parsed_args)  # 執行主函數。
    else:
        # 使用GPU
        with torch.cuda.device(parsed_args.gpu_id):  # 設置 CUDA 設備上下文。
            main(picked_classes, parsed_args)  # 執行主函數。
