import os  # 導入 os 模組，用於與操作系統交互。
import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
import torch.nn as nn  # 導入 PyTorch 的神經網絡模組。
import torch.optim as optim  # 導入 PyTorch 的優化器模組。
from torch.utils.data import DataLoader  # 導入 DataLoader，用於批次加載數據。
from torchvision.utils import make_grid, save_image  # 導入 torchvision 工具，用於生成網格圖像和保存圖像。
from torchvision import transforms as T  # 導入 torchvision 的圖像轉換工具。
from PIL import Image  # 導入 PIL 庫，用於圖像處理。
import numpy as np  # 導入 numpy，用於數值計算。
from sklearn.metrics import roc_auc_score  # 導入 ROC AUC 計算函數。
from torch.utils.tensorboard import SummaryWriter  # 導入 TensorBoard 寫入器，用於可視化訓練過程。
import random  # 導入 random 模組，用於生成隨機數。
import argparse  # 導入 argparse，用於解析命令行參數。
import torch.nn.functional as F  # 導入 PyTorch 的函數式接口。
from sklearn.metrics import roc_auc_score  # 重複導入，保留原樣。
from loss import FocalLoss, SSIM  # 從 loss 模組導入 FocalLoss 和 SSIM 損失函數。
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork  # 從 model_unet 導入子網絡模型。
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset  # 從 data_loader 導入訓練和測試數據集類。
import matplotlib.pyplot as plt  # 導入 matplotlib.pyplot 用於繪圖。
import torchvision.transforms as transforms  # 重複導入 transforms。
from datetime import datetime  # 導入 datetime 用於獲取時間。
from loss import FocalLoss, SSIM  # 重複導入。
import numpy as np  # 重複導入。
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, jaccard_score  # 導入更多評估指標。


def setup_seed(seed):  # 定義設置隨機種子的函數，用於保證實驗的可重複性。
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)  # 設置 CPU 的隨機種子。
    torch.cuda.manual_seed_all(seed)  # 設置所有 GPU 的隨機種子。
    np.random.seed(seed)  # 設置 numpy 的隨機種子。
    random.seed(seed)  # 設置 random 模組的隨機種子。
    torch.backends.cudnn.deterministic = True  # 設置 cuDNN 為確定性模式，保證結果可重現。
    torch.backends.cudnn.benchmark = False  # 關閉 cuDNN 的自動最佳化搜尋，避免不同硬件上結果不一致。


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
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]  # 根據已分配內存排序，選擇最小的那個 GPU ID。
    return available_gpu  # 返回 GPU ID。


def weights_init(m):  # 定義權重初始化函數。
    """ 卷積層權重 → 小亂數讓網路容易學習、梯度穩定
        BatchNorm權重 → 初始縮放 1 保持數值穩定，偏置 0 不改變均值"""
    # 取得模組的類別名稱，例如 'Conv2d', 'BatchNorm2d' 等
    classname = m.__class__.__name__  # 獲取類名。

    # 如果是卷積層 (Conv)
    if classname.find('Conv') != -1:  # 如果類名包含 'Conv'。
        # 權重初始化為均值 0、標準差 0.02 的高斯分布
        # 原因：這樣可以讓卷積層一開始輸出的特徵值分布均衡，避免梯度消失或梯度爆炸
        m.weight.data.normal_(0.0, 0.02)  # 使用正態分佈初始化權重。

    # 如果是批次正規化層 (BatchNorm)
    elif classname.find('BatchNorm') != -1:  # 如果類名包含 'BatchNorm'。
        # 權重初始化為均值 1、標準差 0.02 的高斯分布
        # 原因：BatchNorm 的權重 (gamma) 控制輸出縮放，初始化為 1 可以保持初始特徵分布不變
        m.weight.data.normal_(1.0, 0.02)  # 初始化權重為 1。
        # 偏置初始化為 0
        # 原因：偏置 (beta) 控制輸出偏移量，初始化為 0 可以保持輸出均值不偏移
        m.bias.data.fill_(0)  # 初始化偏置為 0。


def visualize_predictions(teacher_model, teacher_seg_model, student_model,  # 定義可視化預測結果的函數。
                          student_seg_model, batch, device, save_path):
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
        teacher_joined_in = torch.cat((teacher_recon, aug_image), dim=1)  # 將重建圖像與增強圖像拼接。
        teacher_seg_out_mask = teacher_seg_model(teacher_joined_in)  # 教師分割模型輸出。
        teacher_seg_map = torch.softmax(teacher_seg_out_mask, dim=1)  # 計算 softmax 概率。

        # 學生預測
        student_recon = student_model(aug_image)  # 學生模型重建圖像。
        student_joined_in = torch.cat((student_recon, aug_image), dim=1)  # 將重建圖像與增強圖像拼接。
        student_seg_out_mask = student_seg_model(student_joined_in)  # 學生分割模型輸出。
        student_seg_map = torch.softmax(student_seg_out_mask, dim=1)  # 計算 softmax 概率。

        # 轉換為 numpy 用於繪圖
        input_np = input_image.cpu().numpy()[0].transpose(1, 2, 0)  # 轉換輸入圖像為 numpy 格式 (H, W, C)。
        aug_np = aug_image.cpu().numpy()[0].transpose(1, 2, 0)  # 轉換增強圖像為 numpy 格式。
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]  # 取第一個通道的 Ground Truth 掩碼。

        # 處理分割結果
        teacher_seg_np = torch.softmax(teacher_seg_map,
                                       dim=1)[0, 1].cpu().numpy()  # 獲取教師模型預測的異常類別概率。
        student_seg_np = torch.softmax(student_seg_map,
                                       dim=1)[0, 1].cpu().numpy()  # 獲取學生模型預測的異常類別概率。

        # 創建圖像
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 創建 2x4 的子圖佈局。

        # 第一行：原始輸入與標註
        axes[0, 0].imshow(input_np)  # 顯示原始輸入影像。
        axes[0, 0].set_title('Original Image')  # 設置標題。
        axes[0, 0].axis('off')  # 關閉坐標軸。
        axes[0, 1].imshow(aug_np)  # 顯示增強後影像。
        axes[0, 1].set_title('Augmented Image')  # 設置標題。
        axes[0, 1].axis('off')  # 關閉坐標軸。
        axes[0, 2].imshow(gt_mask_np, cmap='jet')  # 顯示 Ground Truth 異常遮罩。
        axes[0, 2].set_title('Ground Truth Mask')  # 設置標題。
        axes[0, 2].axis('off')  # 關閉坐標軸。
        axes[0, 3].axis('off')  # 第四個位置留白。

        # 第二行：模型預測與比較
        im1 = axes[1, 0].imshow(teacher_seg_np, cmap='jet', vmin=0,
                                vmax=1)  # 顯示教師模型的預測熱力圖。
        axes[1, 0].set_title('Teacher Segmentation')  # 設置標題。
        axes[1, 0].axis('off')  # 關閉坐標軸。
        plt.colorbar(im1, ax=axes[1, 0])  # 添加顏色條。

        im2 = axes[1, 1].imshow(student_seg_np, cmap='jet', vmin=0,
                                vmax=1)  # 顯示學生模型的預測熱力圖。
        axes[1, 1].set_title('Student Segmentation')  # 設置標題。
        axes[1, 1].axis('off')  # 關閉坐標軸。
        plt.colorbar(im2, ax=axes[1, 1])  # 添加顏色條。
        # 差異圖
        diff = np.abs(teacher_seg_np - student_seg_np)  # 計算教師與學生預測的絕對差異。
        im3 = axes[1, 2].imshow(diff, cmap='hot', vmin=0,
                                vmax=1)  # 顯示差異圖。
        axes[1, 2].set_title('Teacher-Student Difference')  # 設置標題。
        axes[1, 2].axis('off')  # 關閉坐標軸。
        plt.colorbar(im3, ax=axes[1, 2])  # 添加顏色條。
        # 二值化對比
        student_binary = (student_seg_np > 0.5).astype(np.float32)  # 對學生預測進行二值化。
        im4 = axes[1, 3].imshow(student_binary,
                                cmap='gray')  # 顯示二值化結果。
        axes[1, 3].set_title('Student Binary (>0.5)')  # 設置標題。
        axes[1, 3].axis('off')  # 關閉坐標軸。

        plt.tight_layout()  # 自動調整佈局。
        plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')  # 保存圖像。
        plt.close()  # 關閉圖像以釋放內存。
        student_model.train()  # 將學生模型切換回訓練模式。
        print(f"✅ Visualization saved: {save_path}.png")  # 打印保存路徑。


def detailed_diagnostic_visualization(teacher_model, teacher_seg_model,  # 定義詳細診斷可視化的函數。
                                      student_model, student_seg_model,
                                      loss_focal, batch, device, save_path,
                                      epoch, i_batch):
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
        seg_distill_loss = F.mse_loss(student_seg_map, teacher_seg_map).item()  # 計算分割蒸餾損失。
        student_seg_softmax = torch.softmax(student_seg_map, dim=1)  # 再次計算 softmax (重複操作，可優化)。
        orig_seg_loss = loss_focal(student_seg_softmax, gt_mask).item()  # 計算原始分割損失。

        # 轉換為 numpy
        input_np = input_image.cpu().numpy()[0].transpose(1, 2, 0)  # 轉為 numpy。
        aug_np = aug_image.cpu().numpy()[0].transpose(1, 2, 0)  # 轉為 numpy。
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]  # 轉為 numpy。
        teacher_seg_np = torch.softmax(teacher_seg_map,
                                       dim=1)[0, 1].cpu().numpy()  # 教師預測轉為 numpy。
        student_seg_np = torch.softmax(student_seg_map,
                                       dim=1)[0, 1].cpu().numpy()  # 學生預測轉為 numpy。

        # 創建診斷圖
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))  # 創建 2x5 子圖。

        # 第一行：輸入與預測
        axes[0, 0].imshow(input_np)  # 顯示原始影像。
        axes[0, 0].set_title('Original Image')  # 設置標題。
        axes[0, 0].axis('off')
        axes[0, 1].imshow(aug_np)  # 顯示增強影像。
        axes[0, 1].set_title('Augmented Image')
        axes[0, 1].axis('off')
        axes[0, 2].imshow(gt_mask_np, cmap='jet')  # 顯示 Ground Truth 異常遮罩。
        axes[0, 2].set_title('GT Mask')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(teacher_seg_np, cmap='jet', vmin=0,
                          vmax=1)  # 顯示教師模型預測。
        axes[0, 3].set_title(f'Teacher Seg\nMax: {teacher_seg_np.max():.3f}')  # 顯示最大值。
        axes[0, 3].axis('off')
        axes[0, 4].imshow(student_seg_np, cmap='jet', vmin=0, vmax=1)  # 顯示學生模型預測。
        axes[0, 4].set_title(f'Student Seg\nMax: {student_seg_np.max():.3f}')  # 顯示最大值。
        axes[0, 4].axis('off')

        # 第二行：分析和差異
        diff = np.abs(teacher_seg_np - student_seg_np)  # 計算差異。
        axes[1, 0].imshow(diff, cmap='hot')  # 顯示差異圖。
        axes[1, 0].set_title(f'Difference\nAvg: {diff.mean():.3f}')  # 顯示平均差異。
        axes[1, 0].axis('off')
        # 學生二值化
        student_binary = (student_seg_np > 0.5).astype(np.float32)  # 學生預測二值化。
        axes[1, 1].imshow(student_binary,
                          cmap='gray')  # 顯示學生二值化結果。
        axes[1, 1].set_title('Student Binary\n(>0.5)')
        axes[1, 1].axis('off')
        # 教師二值化
        teacher_binary = (teacher_seg_np > 0.5).astype(np.float32)  # 教師預測二值化。
        axes[1, 2].imshow(teacher_binary,
                          cmap='gray')  # 顯示教師二值化結果。
        axes[1, 2].set_title('Teacher Binary\n(>0.5)')
        axes[1, 2].axis('off')

        # 損失信息
        # 顯示目前訓練週期與批次編號，以及兩種損失值：
        # - seg_distill_loss：學生模仿教師的損失
        # - orig_seg_loss：學生對 Ground Truth 的預測損失
        axes[1, 3].text(0.1,
                        0.7, f'Epoch: {epoch}\nBatch: {i_batch}\n\n'
                        f'Seg Distill Loss: {seg_distill_loss:.4f}\n'
                        f'Orig Seg Loss: {orig_seg_loss:.4f}',
                        fontsize=12)  # 在圖上顯示文本信息。
        axes[1, 3].axis('off')

        # 統計信息
        # 顯示教師與學生模型的統計資訊（平均值與標準差），
        # 用於分析模型預測的穩定性與分佈特性：
        # - Mean：代表整體異常機率的平均強度
        # - Std：代表預測分佈的離散程度，越高表示模型預測越不穩定
        axes[1, 4].text(0.1,
                        0.7, f'Teacher Stats:\n'
                        f'Mean: {teacher_seg_np.mean():.3f}\n'
                        f'Std: {teacher_seg_np.std():.3f}\n\n'
                        f'Student Stats:\n'
                        f'Mean: {student_seg_np.mean():.3f}\n'
                        f'Std: {student_seg_np.std():.3f}',
                        fontsize=12)  # 顯示統計信息。
        axes[1, 4].axis('off')

        plt.tight_layout()  # 調整佈局。
        plt.savefig(save_path + '_diagnostic.png',
                    dpi=150,
                    bbox_inches='tight')  # 保存圖片。
        plt.close()  # 關閉圖片。
        student_model.train()  # 恢復訓練模式。
        print(f"✅ Diagnostic visualization saved: {save_path}_diagnostic.png")  # 打印保存信息。


# =======================
# Main Pipeline
# =======================
def main(obj_names, args):  # 定義主函數。
    setup_seed(111)  # 固定隨機種子。
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 設置設備。
    for obj_name in obj_names:  # 遍歷每個目標類別。
        # Load teacher
        recon_path = f'./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_' + obj_name + '_'  # 構建教師重建模型路徑。
        checkpoint_path = recon_path + ".pckl"  # 完整路徑。
        teacher_recon_ckpt = torch.load(checkpoint_path,
                                        map_location=device,
                                        weights_only=True)  # 加載教師重建權重。
        print("teacher_recon_ckpt keys:", teacher_recon_ckpt.keys())  # 打印鍵值。

        # 假設是處理3通道的RGB圖像
        IMG_CHANNELS = 3
        # 建立教師模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
        teacher_model = ReconstructiveSubNetwork(
            in_channels=IMG_CHANNELS,
            out_channels=IMG_CHANNELS,
            base_width=128,  # 教師重建網路較寬。
        ).to(device)

        # 現在使用修正後的 state_dict 載入，並使用 strict=True 來確保所有權重都正確載入
        teacher_model.load_state_dict(teacher_recon_ckpt, strict=True)
        # 將教師模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
        teacher_model.eval()

        # 將教師模型的所有參數設為不可訓練，避免在後續訓練中被更新
        for p in teacher_model.parameters():
            p.requires_grad = False

        seg_path = f'./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_' + obj_name + '__seg'  # 構建教師分割模型路徑。
        checkpoint_seg_path = seg_path + ".pckl"  # 完整路徑。
        teacher_seg_ckpt = torch.load(checkpoint_seg_path,
                                      map_location=device,
                                      weights_only=True)  # 加載教師分割權重。
        print("teacher_seg_ckpt keys:", teacher_seg_ckpt.keys())  # 打印鍵值。

        IMG_SEG_CHANNELS = 6  # 分割輸入通道數（圖像+重建圖）。
        IMG_SEG_CHANNELS_OUT = 2  # 輸出通道數（異常/正常）。
        teacher_seg_model = DiscriminativeSubNetwork(
            in_channels=IMG_SEG_CHANNELS,
            out_channels=IMG_SEG_CHANNELS_OUT,
            base_channels=64,  # 教師分割網路較寬。
            out_features=False,
        ).to(device)
        teacher_seg_model.load_state_dict(teacher_seg_ckpt, strict=True)  # 加載權重。
        # 將教師模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
        teacher_seg_model.eval()

        # Student model
        #dropout 防止過擬合，幫助學生模型泛化，避免過擬合教師模型提取的特徵。在蒸餾訓練時，讓學生模型學到更穩健的特徵，而不是完全模仿教師模型的單一路徑
        student_model = ReconstructiveSubNetwork(
            in_channels=IMG_CHANNELS,
            out_channels=IMG_CHANNELS,
            base_width=64,  # 學生重建網路較窄。
        ).to(device)

        #初始化 卷積層和 BatchNorm 層的初始權重分布合理，幫助模型更快收斂
        student_model.apply(weights_init)

        student_seg_model = DiscriminativeSubNetwork(
            in_channels=IMG_SEG_CHANNELS,
            out_channels=IMG_SEG_CHANNELS_OUT,
            base_channels=32,  # 學生分割網路較窄。
            out_features=False,
        ).to(device)

        #初始化 卷積層和 BatchNorm 層的初始權重分布合理，幫助模型更快收斂
        student_seg_model.apply(weights_init)

        # 定義優化器，只優化學生模型和特徵對齊層的參數
        # optimizer = torch.optim.Adam(list(student_model.parameters()) +
        #                              list(feature_aligns.parameters()),
        #                              lr=args.lr)
        optimizer = torch.optim.Adam([{
            "params": student_model.parameters(),
            "lr": args.lr
        }, {
            "params": student_seg_model.parameters(),
            "lr": args.lr
        }])
        # 設定學習率調整策略，使用 MultiStepLR(一開始大步走，後面小步走)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,  # 需要調整的優化器
            [args.epochs * 0.8, args.epochs * 0.9],  # 在訓練 80% 和 90% 時調整學習率
            gamma=0.2,  # 每次調整時學習率乘上 0.2
            last_epoch=-1)  # 從頭開始計算學習率

        # 定義損失函數
        loss_focal = FocalLoss()  #解決類別不平衡、強化模型對難分類樣本的學習。
        # loss_l2 = torch.nn.modules.loss.MSELoss()  # L2 損失函數（均方誤差）。
        loss_ssim = SSIM()  # SSIM 損失函數。

        path = f'./mvtec'  # 訓練資料路徑
        path_dtd = f'./dtd/images/'  # DTD 紋理數據路徑。
        # Load datasets
        # 載入訓練資料集，指定根目錄、類別、資料切分方式為 "train"，並將影像尺寸調整為 256x256
        #python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/
        train_dataset = MVTecDRAEMTrainDataset(root_dir=path +
                                               f'/{obj_name}/train/good/',
                                               anomaly_source_path=path_dtd,
                                               resize_shape=[256, 256])
        # 建立訓練資料的 DataLoader，設定每批次大小為 16，打亂資料順序，使用 4 個執行緒加速載入
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.bs,
                                  shuffle=True,
                                  num_workers=4)

        # --- 驗證資料加載 (新增部分) ---
        val_dataset = MVTecDRAEMTestDataset(
            root_dir=path,  # 傳遞 mvtec 的根目錄
            category_name=obj_name,  # 傳遞類別名稱
            resize_shape=[256, 256])  # 設置調整大小形狀。
        val_loader = DataLoader(val_dataset,
                                batch_size=args.bs,
                                shuffle=False,
                                num_workers=4)

        # 主儲存資料夾路徑
        save_root = "./save_files"

        # 若主資料夾不存在，則建立
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        # 指定模型檢查點（checkpoint）儲存的資料夾路徑
        # 模型檢查點儲存路徑
        checkpoint_dir = os.path.join(save_root, "checkpoints")
        # 如果檢查點資料夾不存在，則建立該資料夾（exist_ok=True 表示若已存在則不報錯）
        os.makedirs(checkpoint_dir, exist_ok=True)

        # n_iter = 0

        # --- 超參數定義 ---
        lambda_l2 = 1.0  # L2 損失權重。
        lambda_ssim = 0.5  # (D) 降低 SSIM 比例。
        lambda_segment = 2.0  # (A) 提高分割損失權重。
        lambda_recon_distill = 0.3  # (E) 分離蒸餾權重。
        lambda_seg_distill = 0.7  # 分割蒸餾權重。
        best_pixel_auroc = 0.0  # 初始化最佳 Pixel AUROC。

        for epoch in range(args.epochs):  # 訓練循環。
            print("Epoch:", epoch)  # 打印 Epoch。

            # (B) 動態蒸餾權重：前 10 epoch 不啟用，之後線性增加
            if epoch < 10:
                lambda_distill = 0.0  # 前 10 個 epoch 不使用蒸餾損失。
            else:
                lambda_distill = 0.5 * (epoch / args.epochs)  # 線性增加蒸餾權重。

            epoch_loss = 0.0  # 初始化 epoch 損失。
            epoch_seg_distill_loss = 0.0  # 初始化分割蒸餾損失。
            epoch_orig_seg_loss = 0.0  # 初始化原始分割損失。
            num_batches = 0  # 批次計數器。

            for i_batch, sample_batched in enumerate(train_loader):  # 批次循環。
                # ==================== 數據加載 ====================
                input_image = sample_batched["image"].to(device)  # 獲取輸入圖像。
                ground_truth_mask = sample_batched["anomaly_mask"].to(
                    device).float()  # 獲取異常掩碼。
                aug_gray_batch = sample_batched["augmented_image"].to(device)  # 獲取增強灰度圖（實際上是增強後的彩色圖）。
                is_normal = sample_batched.get(
                    "is_normal", torch.ones_like(ground_truth_mask)).bool()  # 獲取是否正常樣本的標記。

                # ==================== 教師網路前向 ====================
                with torch.no_grad():
                    teacher_recon = teacher_model(aug_gray_batch)  # 教師模型重建。
                    teacher_joined_in = torch.cat(
                        (teacher_recon, aug_gray_batch), dim=1)  # 拼接輸入。
                    teacher_out_mask = teacher_seg_model(teacher_joined_in)  # 教師分割輸出。
                    teacher_seg_map = torch.softmax(teacher_out_mask, dim=1)  # 教師分割概率。

                # ==================== 學生網路前向 ====================
                student_recon = student_model(aug_gray_batch)  # 學生模型重建。
                student_joined_in = torch.cat((student_recon, aug_gray_batch),
                                              dim=1)  # 拼接輸入。
                student_out_mask = student_seg_model(student_joined_in)  # 學生分割輸出。
                student_seg_map = torch.softmax(student_out_mask, dim=1)  # 學生分割概率。

                # ==================== (C) 加權重建損失 ====================
                # 將 L2、SSIM 對正常區域 (1 - mask) 加權
                l2_map = (student_recon - input_image)**2  # 計算 L2 誤差圖。
                weighted_l2 = (l2_map * (1 - ground_truth_mask)).mean()  # 僅對正常區域計算平均 L2 損失。

                ssim_map = loss_ssim(student_recon, input_image, return_map=True) \
                    if hasattr(loss_ssim, "return_map") else loss_ssim(student_recon, input_image) # 計算 SSIM 損失圖。
                if isinstance(ssim_map, torch.Tensor) and ssim_map.ndim > 0:  # 如果返回的是張量且維度大於 0。
                    weighted_ssim = (ssim_map * (1 - ground_truth_mask)).mean()  # 僅對正常區域計算平均 SSIM 損失。
                else:
                    weighted_ssim = ssim_map  # 否則直接使用 SSIM 損失。

                loss_recon = lambda_l2 * weighted_l2 + lambda_ssim * weighted_ssim  # 組合重建損失。

                # ==================== (A) 分割損失 ====================
                segment_loss = loss_focal(student_seg_map, ground_truth_mask)  # 計算 Focal Loss。
                loss_seg = lambda_segment * segment_loss  # 加權分割損失。

                # ==================== (E) 分離蒸餾損失 ====================
                recon_distill_loss = F.mse_loss(student_recon, teacher_recon)  # 計算重建蒸餾損失 (MSE)。
                seg_distill_loss = F.mse_loss(student_seg_map, teacher_seg_map)  # 計算分割蒸餾損失 (MSE)。
                distill_loss = (lambda_recon_distill * recon_distill_loss +
                                lambda_seg_distill * seg_distill_loss)  # 組合蒸餾損失。

                # 蒸餾掩碼：僅對正常樣本啟用
                distill_mask = is_normal.float().mean()  # 計算蒸餾掩碼權重 (只有正常樣本貢獻)。
                loss_distill = lambda_distill * distill_mask * distill_loss  # 加權蒸餾損失。

                # ==================== (B) Warmup 蒸餾控制 ====================
                if epoch < 10:  # 如果在前 10 個 epoch。
                    total_loss = loss_recon + loss_seg  # 總損失不包含蒸餾損失。
                else:  # 否則。
                    total_loss = loss_recon + loss_seg + loss_distill  # 總損失包含蒸餾損失。

                # -------------------- 反向傳播與統計 --------------------
                optimizer.zero_grad()  # 清空梯度。
                total_loss.backward()  # 反向傳播。
                optimizer.step()  # 更新參數。

                epoch_loss += total_loss.item()  # 累加總損失。
                num_batches += 1  # 增加批次計數。

            # --- Epoch 結束處理 ---
            scheduler.step()  # 更新學習率。

            # 平均損失輸出
            avg_total_loss = epoch_loss / num_batches  # 計算平均總損失。
            avg_orig_seg_loss = epoch_orig_seg_loss / num_batches  # 計算平均原始分割損失。
            print("-" * 50)  # 打印分隔線。
            print(f"Epoch {epoch} Summary:")  # 打印 Epoch 總結。
            print(f"  - Average Total Loss    : {avg_total_loss:.6f}")  # 打印平均總損失。
            print(f"  - Average Seg Loss      : {avg_orig_seg_loss:.6f}")  # 打印平均分割損失。
            print("-" * 50)  # 打印分隔線。

            # --- 驗證階段 (保持原程式) ---
            if val_loader:  # 如果有驗證集。
                student_model.eval()  # 學生模型設為評估模式。
                student_seg_model.eval()  # 學生分割模型設為評估模式。
                all_pred_masks = []  # 初始化預測掩碼列表。
                all_gt_masks = []  # 初始化真實掩碼列表。

                with torch.no_grad():  # 禁用梯度。
                    for i_batch_val, sample_batched_val in enumerate(
                            val_loader):
                        input_image_val = sample_batched_val["image"].to(
                            device)  # 獲取驗證圖像。
                        ground_truth_mask_val = sample_batched_val[
                            "anomaly_mask"].to(device).float()  # 獲取驗證掩碼。

                        student_recon = student_model(input_image_val)  # 學生重建。
                        student_joined_in = torch.cat(
                            (student_recon, input_image_val), dim=1)  # 拼接。
                        student_seg_out_mask = student_seg_model(
                            student_joined_in)  # 學生分割。
                        student_seg_map = torch.softmax(student_seg_out_mask,
                                                        dim=1)  # 學生概率。
                        student_seg_map_val = student_seg_map[:, 1:
                                                              2, :, :]  # B,1,H,W  # 取異常通道。

                        all_pred_masks.append(
                            student_seg_map_val.cpu().numpy())  # 添加預測結果。
                        all_gt_masks.append(
                            ground_truth_mask_val.cpu().numpy())  # 添加真實結果。

                # === Flatten & compute AUROC / F1 / IoU ===
                all_pred_masks = np.concatenate(all_pred_masks, axis=0)  # 連接所有批次結果。
                all_gt_masks = np.concatenate(all_gt_masks, axis=0)  # 連接所有批次真實值。
                all_pred_masks_flat = all_pred_masks.flatten()  # 展平。
                all_gt_masks_flat = all_gt_masks.flatten().astype(int)  # 展平並轉為整數。

                try:
                    fpr, tpr, _ = roc_curve(all_gt_masks_flat,
                                            all_pred_masks_flat)  # 計算 ROC 曲線。
                    pixel_auroc = auc(fpr, tpr)  # 計算 AUROC。
                except ValueError:
                    pixel_auroc = float('nan')  # 處理異常。

                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(
                        all_gt_masks_flat, all_pred_masks_flat)  # 計算 PR 曲線。
                    pixel_pr_auc = auc(recall_curve, precision_curve)  # 計算 PR-AUC。
                except ValueError:
                    pixel_pr_auc = float('nan')  # 處理異常。

                threshold = 0.5  # 設置閾值。
                binary_pred_masks_flat = (all_pred_masks_flat
                                          > threshold).astype(int)  # 二值化預測。

                pixel_precision = precision_score(all_gt_masks_flat,
                                                  binary_pred_masks_flat,
                                                  zero_division=0)  # 計算精確率。
                pixel_recall = recall_score(all_gt_masks_flat,
                                            binary_pred_masks_flat,
                                            zero_division=0)  # 計算召回率。
                pixel_f1 = f1_score(all_gt_masks_flat,
                                    binary_pred_masks_flat,
                                    zero_division=0)  # 計算 F1 分數。
                pixel_iou = jaccard_score(all_gt_masks_flat,
                                          binary_pred_masks_flat,
                                          zero_division=0)  # 計算 IoU。

                print("-" * 50)  # 打印分隔線。
                print(f"Epoch {epoch} Anomaly Detection Metrics:")  # 打印 Epoch 指標。
                print(f"  - Pixel-level AUROC   : {pixel_auroc:.4f}")  # 打印 AUROC。
                print(f"  - Pixel-level PR-AUC  : {pixel_pr_auc:.4f}")  # 打印 PR-AUC。
                print(f"  - Pixel-level Precision: {pixel_precision:.4f}")  # 打印 Precision。
                print(f"  - Pixel-level Recall  : {pixel_recall:.4f}")  # 打印 Recall。
                print(f"  - Pixel-level F1 Score: {pixel_f1:.4f}")  # 打印 F1 Score。
                print(f"  - Pixel-level IoU     : {pixel_iou:.4f}")  # 打印 IoU。
                print("-" * 50)  # 打印分隔線。

                student_model.train()  # 學生模型恢復訓練模式。
                student_seg_model.train()  # 學生分割模型恢復訓練模式。

                # --- 儲存最佳模型 ---
                if not np.isnan(
                        pixel_auroc) and pixel_auroc > best_pixel_auroc:  # 如果當前 AUROC 是最好的。
                    best_pixel_auroc = pixel_auroc  # 更新最佳 AUROC。
                    save_path = os.path.join(checkpoint_dir,
                                             f"{obj_name}_best_recon.pckl")  # 構建保存路徑。
                    save_seg_path = os.path.join(checkpoint_dir,
                                                 f"{obj_name}_best_seg.pckl")  # 構建保存路徑。
                    torch.save(student_model.state_dict(), save_path)  # 保存學生模型權重。
                    torch.save(student_seg_model.state_dict(), save_seg_path)  # 保存學生分割模型權重。
                    print(
                        f"✅ New best model saved at epoch {epoch} (Pixel AUROC={best_pixel_auroc:.4f})"
                    )  # 打印保存信息。

        torch.cuda.empty_cache()  # 清空 CUDA 緩存。


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    """
    --gpu_id -2：自動選擇最佳GPU
    --gpu_id -1：強制使用CPU
    --gpu_id  0：使用GPU 0（原有行為）
    """

    parser = argparse.ArgumentParser()  # 創建參數解析器。
    parser.add_argument('--obj_id', action='store', type=int, required=True)  # 添加 obj_id 參數。
    parser.add_argument('--epochs', default=25, type=int)  # 添加 epochs 參數。
    parser.add_argument('--bs', action='store', type=int, required=True)  # 添加 batch size 參數。
    parser.add_argument('--lr', action='store', type=float, required=True)  # 添加 learning rate 參數。
    parser.add_argument('--gpu_id',
                        action='store',
                        type=int,
                        default=-2,
                        required=False,
                        help='GPU ID (-2: auto-select, -1: CPU)')  # 添加 gpu_id 參數。
    args = parser.parse_args()  # 解析參數。

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()  # 獲取可用 GPU。
        print(f"自動選擇 GPU: {args.gpu_id}")  # 打印選擇結果。

    obj_batch = [['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],  # 定義目標類別列表。
                 ['transistor'], ['tile'], ['cable'], ['zipper'],
                 ['toothbrush'], ['metal_nut'], ['hazelnut'], ['screw'],
                 ['grid'], ['wood']]

    if int(args.obj_id) == -1:  # 如果 obj_id 為 -1，選擇所有類別。
        obj_list = [
            'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
            'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
            'screw', 'grid', 'wood'
        ]
        picked_classes = obj_list
    else:  # 否則選擇指定類別。
        picked_classes = obj_batch[int(args.obj_id)]

    # 根據選擇的GPU執行
    if args.gpu_id == -1:  # 如果使用 CPU。
        # 使用CPU
        main(picked_classes, args)  # 執行主函數。
    else:
        # 使用GPU
        with torch.cuda.device(args.gpu_id):  # 設置 CUDA 設備上下文。
            main(picked_classes, args)  # 執行主函數。
