import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import random  # 亂數控制
import argparse  # 命令列參數處理
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, SSIM
from model_unet import ReconstructiveSubNetwork,DiscriminativeSubNetwork
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datetime import datetime
from loss import FocalLoss, SSIM
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score, jaccard_score


def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋


# =======================
# Utilities
# =======================
def get_available_gpu():
    """自動選擇記憶體使用率最低的GPU"""
    if not torch.cuda.is_available():
        return -1  # 沒有GPU可用

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1

    # 檢查每個GPU的記憶體使用情況
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_reserved = torch.cuda.memory_reserved(i)
        gpu_memory.append((i, memory_allocated, memory_reserved))

    # 選擇記憶體使用最少的GPU
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu


def weights_init(m):
    """ 卷積層權重 → 小亂數讓網路容易學習、梯度穩定
        BatchNorm權重 → 初始縮放 1 保持數值穩定，偏置 0 不改變均值"""
    # 取得模組的類別名稱，例如 'Conv2d', 'BatchNorm2d' 等
    classname = m.__class__.__name__

    # 如果是卷積層 (Conv)
    if classname.find('Conv') != -1:
        # 權重初始化為均值 0、標準差 0.02 的高斯分布
        # 原因：這樣可以讓卷積層一開始輸出的特徵值分布均衡，避免梯度消失或梯度爆炸
        m.weight.data.normal_(0.0, 0.02)

    # 如果是批次正規化層 (BatchNorm)
    elif classname.find('BatchNorm') != -1:
        # 權重初始化為均值 1、標準差 0.02 的高斯分布
        # 原因：BatchNorm 的權重 (gamma) 控制輸出縮放，初始化為 1 可以保持初始特徵分布不變
        m.weight.data.normal_(1.0, 0.02)
        # 偏置初始化為 0
        # 原因：偏置 (beta) 控制輸出偏移量，初始化為 0 可以保持輸出均值不偏移
        m.bias.data.fill_(0)


def visualize_predictions(teacher_model, student_model, batch, device,
                          save_path):
    """
    視覺化教師模型和學生模型的預測結果對比
    """
    teacher_model.eval()
    student_model.eval()
    with torch.no_grad():
        input_image = batch["image"].to(device)
        aug_image = batch["augmented_image"].to(device)
        gt_mask = batch["anomaly_mask"].to(device)

        # 教師預測
        teacher_recon, teacher_seg = teacher_model(aug_image)
        # 學生預測
        student_recon, student_seg = student_model(aug_image)

        # 轉換為 numpy 用於繪圖
        input_np = input_image.cpu().numpy()[0].transpose(1, 2, 0)
        aug_np = aug_image.cpu().numpy()[0].transpose(1, 2, 0)
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]  # 取第一個通道

        # 處理分割結果
        teacher_seg_np = torch.softmax(teacher_seg,
                                       dim=1)[0, 1].cpu().numpy()  # 異常類別概率
        student_seg_np = torch.softmax(student_seg, dim=1)[0, 1].cpu().numpy()

        # 創建圖像
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # 第一行：原始輸入與標註
        axes[0, 0].imshow(input_np)  # 原始輸入影像，用於參考真實場景
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(aug_np)  # 增強後影像，實際輸入模型的資料
        axes[0, 1].set_title('Augmented Image')
        axes[0, 1].axis('off')
        axes[0, 2].imshow(gt_mask_np, cmap='jet')  # Ground Truth 異常遮罩，作為標準答案
        axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')  # 空白

        # 第二行：模型預測與比較
        im1 = axes[1, 0].imshow(teacher_seg_np, cmap='jet', vmin=0,
                                vmax=1)  # 教師模型的異常機率分佈，作為學生模型的學習目標
        axes[1, 0].set_title('Teacher Segmentation')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])

        im2 = axes[1, 1].imshow(student_seg_np, cmap='jet', vmin=0,
                                vmax=1)  # 學生模型的異常機率分佈，用於與教師模型做對比
        axes[1, 1].set_title('Student Segmentation')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        # 差異圖
        diff = np.abs(teacher_seg_np - student_seg_np)
        im3 = axes[1, 2].imshow(diff, cmap='hot', vmin=0,
                                vmax=1)  # 教師與學生模型預測差異圖，顯示兩者在空間上的預測偏差
        axes[1, 2].set_title('Teacher-Student Difference')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2])
        # 二值化對比
        student_binary = (student_seg_np > 0.5).astype(np.float32)
        im4 = axes[1, 3].imshow(student_binary,
                                cmap='gray')  # 學生模型的二值化結果（閾值0.5），用於觀察最終異常判斷區域
        axes[1, 3].set_title('Student Binary (>0.5)')
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
        plt.close()
        student_model.train()
        print(f"✅ Visualization saved: {save_path}.png")


def detailed_diagnostic_visualization(teacher_model, student_model, loss_focal,
                                      batch, device, save_path, epoch,
                                      i_batch):
    """
    更詳細的診斷視覺化，包含損失值和指標
    """
    teacher_model.eval()
    student_model.eval()
    with torch.no_grad():
        input_image = batch["image"].to(device)
        aug_image = batch["augmented_image"].to(device)
        gt_mask = batch["anomaly_mask"].to(device)

        # 獲取預測
        teacher_recon, teacher_seg = teacher_model(aug_image)
        student_recon, student_seg = student_model(aug_image)

        # 計算當前損失（僅用於顯示）
        seg_distill_loss = F.mse_loss(student_seg, teacher_seg).item()
        student_seg_softmax = torch.softmax(student_seg, dim=1)
        orig_seg_loss = loss_focal(student_seg_softmax, gt_mask).item()

        # 轉換為 numpy
        input_np = input_image.cpu().numpy()[0].transpose(1, 2, 0)
        aug_np = aug_image.cpu().numpy()[0].transpose(1, 2, 0)
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]
        teacher_seg_np = torch.softmax(teacher_seg, dim=1)[0, 1].cpu().numpy()
        student_seg_np = torch.softmax(student_seg, dim=1)[0, 1].cpu().numpy()

        # 創建診斷圖
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))

        # 第一行：輸入與預測
        axes[0, 0].imshow(input_np)  # 原始影像
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(aug_np)  # 增強影像
        axes[0, 1].set_title('Augmented Image')
        axes[0, 1].axis('off')
        axes[0, 2].imshow(gt_mask_np, cmap='jet')  # Ground Truth 異常遮罩
        axes[0, 2].set_title('GT Mask')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(teacher_seg_np, cmap='jet', vmin=0,
                          vmax=1)  # 教師模型預測，並顯示最大異常機率值，評估其敏感度
        axes[0, 3].set_title(f'Teacher Seg\nMax: {teacher_seg_np.max():.3f}')
        axes[0, 3].axis('off')
        axes[0, 4].imshow(student_seg_np, cmap='jet', vmin=0, vmax=1)
        axes[0, 4].set_title(f'Student Seg\nMax: {student_seg_np.max():.3f}')
        axes[0, 4].axis('off')  # 學生模型預測，並顯示最大異常機率值，評估其偵測能力

        # 第二行：分析和差異
        diff = np.abs(teacher_seg_np - student_seg_np)
        axes[1, 0].imshow(diff, cmap='hot')  # 教師與學生的差異圖，並顯示平均差異值，用於衡量知識蒸餾效果
        axes[1, 0].set_title(f'Difference\nAvg: {diff.mean():.3f}')
        axes[1, 0].axis('off')
        # 學生二值化
        student_binary = (student_seg_np > 0.5).astype(np.float32)
        axes[1, 1].imshow(student_binary,
                          cmap='gray')  # 學生模型的二值化結果，觀察其最終異常判斷區域
        axes[1, 1].set_title('Student Binary\n(>0.5)')
        axes[1, 1].axis('off')
        # 教師二值化
        teacher_binary = (teacher_seg_np > 0.5).astype(np.float32)
        axes[1, 2].imshow(teacher_binary,
                          cmap='gray')  # 教師模型的二值化結果，作為學生模型的參考標準
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
                        fontsize=12)
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
                        fontsize=12)
        axes[1, 4].axis('off')

        plt.tight_layout()
        plt.savefig(save_path + '_diagnostic.png',
                    dpi=150,
                    bbox_inches='tight')
        plt.close()
        student_model.train()
        print(f"✅ Diagnostic visualization saved: {save_path}_diagnostic.png")


# =======================
# Main Pipeline
# =======================
def main(obj_names, args):
    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for obj_name in obj_names:
        # Load teacher
        recon_path = f'./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_' + obj_name + '_'
        checkpoint_path = recon_path + ".pckl"
        teacher_recon_ckpt = torch.load(checkpoint_path,
                                        map_location=device,
                                        weights_only=True)
        print("teacher_recon_ckpt keys:", teacher_recon_ckpt.keys())

        # 假設是處理3通道的RGB圖像
        IMG_CHANNELS = 3
        # 建立教師模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
        teacher_model = ReconstructiveSubNetwork(
            in_channels=IMG_CHANNELS,
            out_channels=IMG_CHANNELS,
            base_width=128,  # 教師重建網路較寬
        ).to(device)

        # 現在使用修正後的 state_dict 載入，並使用 strict=True 來確保所有權重都正確載入
        teacher_model.load_state_dict(teacher_recon_ckpt, strict=True)
        # 將教師模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
        teacher_model.eval()

        # 將教師模型的所有參數設為不可訓練，避免在後續訓練中被更新
        for p in teacher_model.parameters():
            p.requires_grad = False

        seg_path = f'./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_' + obj_name + '__seg'
        checkpoint_seg_path = seg_path + ".pckl"
        teacher_seg_ckpt = torch.load(checkpoint_seg_path,
                                      map_location=device,
                                      weights_only=True)
        print("teacher_seg_ckpt keys:", teacher_seg_ckpt.keys())

        IMG_SEG_CHANNELS = 6
        IMG_SEG_CHANNELS_OUT = 2
        teacher_seg_model = DiscriminativeSubNetwork(
            in_channels=IMG_SEG_CHANNELS,
            out_channels=IMG_SEG_CHANNELS_OUT,
            base_channels=64,  # 教師重建網路較寬
            out_features=False,
        ).to(device)
        teacher_seg_model.load_state_dict(teacher_seg_ckpt, strict=True)
        # 將教師模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
        teacher_seg_model.eval()

        # Student model
        #dropout 防止過擬合，幫助學生模型泛化，避免過擬合教師模型提取的特徵。在蒸餾訓練時，讓學生模型學到更穩健的特徵，而不是完全模仿教師模型的單一路徑
        student_model = ReconstructiveSubNetwork(
            in_channels=IMG_CHANNELS,
            out_channels=IMG_CHANNELS,
            base_width=64,  # 學生重建網路較窄
        ).to(device)

        #初始化 卷積層和 BatchNorm 層的初始權重分布合理，幫助模型更快收斂
        student_model.apply(weights_init)

        student_seg_model = DiscriminativeSubNetwork(
            in_channels=IMG_SEG_CHANNELS,
            out_channels=IMG_SEG_CHANNELS_OUT,
            base_channels=32,  # 學生重建網路較窄
            out_features=False,
        ).to(device)

        #初始化 卷積層和 BatchNorm 層的初始權重分布合理，幫助模型更快收斂
        student_seg_model.apply(weights_init)

        # 定義優化器，只優化學生模型和特徵對齊層的參數
        # optimizer = torch.optim.Adam(list(student_model.parameters()) +
        #                              list(feature_aligns.parameters()),
        #                              lr=args.lr)
        optimizer = torch.optim.Adam([
                                      {"params": student_model.parameters(), "lr": args.lr},
                                      {"params": student_seg_model.parameters(), "lr": args.lr}])
        # 設定學習率調整策略，使用 MultiStepLR(一開始大步走，後面小步走)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,  # 需要調整的優化器
            [args.epochs * 0.8, args.epochs * 0.9],  # 在訓練 80% 和 90% 時調整學習率
            gamma=0.2,  # 每次調整時學習率乘上 0.2
            last_epoch=-1)  # 從頭開始計算學習率

        # 定義損失函數
        loss_focal = FocalLoss()  #解決類別不平衡、強化模型對難分類樣本的學習。
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()

        path = f'./mvtec'  # 訓練資料路徑
        path_dtd = f'./dtd/images/'
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
            resize_shape=[256, 256])
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

        # 開始進行多輪訓練迴圈
        n_iter = 0

        # --- 超參數定義 ---
        lambda_l2 = 1.0
        lambda_ssim = 1.0
        lambda_segment = 2.0  # 分割損失權重 1.0 -> 1.5 ->2.0
        lambda_distill = 0.2  # 蒸餾損失權重，作為輔助項 0.5
        best_pixel_auroc = 0.0  # 初始化最佳 Pixel AUROC
        for epoch in range(args.epochs):
            print("Epoch: " + str(epoch))

            epoch_loss = 0.0
            epoch_seg_distill_loss = 0.0
            epoch_orig_seg_loss = 0.0
            num_batches = 0

            for i_batch, sample_batched in enumerate(train_loader):
                # 數據加載
                input_image = sample_batched["image"].to(device)
                ground_truth_mask = sample_batched["anomaly_mask"].to(
                    device).float()
                aug_gray_batch = sample_batched["augmented_image"].to(
                    device)  #增強後的彩色圖像

                # --- 教師網路前向傳播 ---
                with torch.no_grad():
                    teacher_recon = teacher_model(
                        aug_gray_batch)
                    teacher_joined_in = torch.cat((teacher_recon, aug_gray_batch), dim=1)
                    teacher_out_mask = teacher_seg_model(teacher_joined_in)
                    teacher_seg_map = torch.softmax(teacher_out_mask, dim=1)
                # --- 學生網路前向傳播 ---
                student_recon= student_model(
                    aug_gray_batch)
                student_joined_in = torch.cat((student_recon, aug_gray_batch), dim=1)
                student_out_mask = student_seg_model(student_joined_in)
                student_seg_map = torch.softmax(student_out_mask, dim=1)

                # --- 計算損失函數 ---
                # 1. 重建損失
                l2_loss = loss_l2(student_recon, input_image)
                ssim_loss = loss_ssim(student_recon, input_image)

                # 2. 分割損失
                segment_loss = loss_focal(student_seg_map, ground_truth_mask)

                # 3. 知識蒸餾損失
                recon_distill_loss = F.mse_loss(student_recon, teacher_recon)
                seg_distill_loss = F.mse_loss(student_seg_map, teacher_seg_map)
                distill_loss = recon_distill_loss + seg_distill_loss

                # --- 總損失（使用權重參數）---
                total_loss = (lambda_l2 * l2_loss + lambda_ssim * ssim_loss +
                              lambda_segment * segment_loss +
                              lambda_distill * distill_loss)

                # ==================== 診斷輸出 ====================
                if i_batch % 50 == 0:
                    print(f"\n[Epoch {epoch}, Batch {i_batch}] Loss values:")
                    print(
                        f"  - L2 Loss           : {l2_loss.item():.4f} (Weighted: {lambda_l2 * l2_loss.item():.4f})"
                    )
                    print(
                        f"  - SSIM Loss         : {ssim_loss.item():.4f} (Weighted: {lambda_ssim * ssim_loss.item():.4f})"
                    )
                    print(
                        f"  - Segment Loss      : {segment_loss.item():.4f} (Weighted: {lambda_segment * segment_loss.item():.4f})"
                    )
                    print(
                        f"  - Recon Distill Loss: {recon_distill_loss.item():.4f}"
                    )
                    print(
                        f"  - Seg Distill Loss  : {seg_distill_loss.item():.4f}"
                    )
                    print(
                        f"  - Total Distill Loss: {distill_loss.item():.4f} (Weighted: {lambda_distill * distill_loss.item():.4f})"
                    )
                    print(f"  - Total Loss        : {total_loss.item():.4f}")

                # --- 反向傳播與優化 ---
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # # --- 視覺化 ---
                # if i_batch % 100 == 0:
                #     visualize_predictions(
                #         teacher_model, student_model, sample_batched, device,
                #         os.path.join(save_root,
                #                      f"vis_epoch_{epoch}_batch_{i_batch}"))

                # if i_batch % 500 == 0:
                #     detailed_diagnostic_visualization(
                #         teacher_model, student_model, loss_focal,
                #         sample_batched, device,
                #         os.path.join(save_root,
                #                      f"diag_epoch_{epoch}_batch_{i_batch}"),
                #         epoch, i_batch)

                # --- 累加損失統計 ---
                epoch_loss += total_loss.item()
                epoch_seg_distill_loss += seg_distill_loss.item()
                epoch_orig_seg_loss += segment_loss.item(
                )  # 修正：使用segment_loss而不是orig_seg_loss
                num_batches += 1
                n_iter += 1

            # --- Epoch結束處理 ---
            scheduler.step()

            # 計算平均損失
            avg_total_loss = epoch_loss / num_batches
            avg_orig_seg_loss = epoch_orig_seg_loss / num_batches
            print("-" * 50)
            print(f"Epoch {epoch} Summary:")
            print(f"  - Average Total Loss    : {avg_total_loss:.6f}")
            print(f"  - Average Seg Loss      : {avg_orig_seg_loss:.6f}")
            print("-" * 50)

            # --- 缺陷檢測指標計算 (新增部分) ---
            # 確保你有一個驗證 DataLoader (val_loader)
            if val_loader:
                student_model.eval()  # 設定為評估模式
                student_seg_model.eval()
                all_pred_masks = []
                all_gt_masks = []
                with torch.no_grad():
                    for i_batch_val, sample_batched_val in enumerate(
                            val_loader):
                        input_image_val = sample_batched_val["image"].to(
                            device)

                        ground_truth_mask_val = sample_batched_val[
                            "anomaly_mask"].to(device).float()
                        # aug_gray_batch_val = sample_batched_val[
                        #     "augmented_image"].to(device)

                        student_recon= student_model(
                            input_image_val)
                        student_joined_in = torch.cat((input_image_val.detach(), input_image_val), dim=1)

                        student_seg_out_mask = student_seg_model(student_joined_in)
                        student_seg_map = torch.softmax(student_seg_out_mask, dim=1)

                        student_seg_map_val = student_seg_map[:,1, :, :]
                        student_seg_map_val = student_seg_map_val.unsqueeze(1)  # 確保形狀是 (B, 1, H, W)

                        # 將預測結果和真實標籤收集起來
                        all_pred_masks.append(
                            student_seg_map_val.cpu().numpy())
                        all_gt_masks.append(
                            ground_truth_mask_val.cpu().numpy())

                # 將所有批次的結果串接成一個大陣列
                all_pred_masks = np.concatenate(all_pred_masks, axis=0)
                all_gt_masks = np.concatenate(all_gt_masks, axis=0)

                # 將多維度圖像展平為一維陣列，以便計算指標
                all_pred_masks_flat = all_pred_masks.flatten()
                all_gt_masks_flat = all_gt_masks.flatten()
                #將 ground truth 轉換為整數類型
                all_gt_masks_flat = all_gt_masks_flat.astype(int)

                # 計算 P-AUROC
                # 注意: roc_curve 需要 positive class 為 1
                try:
                    # roc_curve 和 precision_recall_curve 通常可以處理 0.0/1.0 的浮點數
                    # 但為了保險起見和保持一致性，建議也傳入 int 類型
                    fpr, tpr, _ = roc_curve(
                        all_gt_masks_flat, all_pred_masks_flat
                    )  # 這裡 all_pred_masks_flat 還是連續值，是正確的
                    pixel_auroc = auc(fpr, tpr)
                except ValueError:
                    pixel_auroc = float('nan')  # 如果只有一個類別，roc_curve 會報錯

                # 計算 PR-AUC
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(
                        all_gt_masks_flat, all_pred_masks_flat
                    )  # 這裡 all_pred_masks_flat 還是連續值，是正確的
                    pixel_pr_auc = auc(recall_curve, precision_curve)
                except ValueError:
                    pixel_pr_auc = float('nan')

                # 設定閾值計算其他指標 (例如 0.5)
                threshold = 0.5
                binary_pred_masks_flat = (all_pred_masks_flat
                                          > threshold).astype(int)

                pixel_precision = precision_score(
                    all_gt_masks_flat,  # y_true 現在是 int
                    binary_pred_masks_flat,  # y_pred 也是 int
                    zero_division=0)
                pixel_recall = recall_score(
                    all_gt_masks_flat,  # y_true 現在是 int
                    binary_pred_masks_flat,  # y_pred 也是 int
                    zero_division=0)
                pixel_f1 = f1_score(
                    all_gt_masks_flat,  # y_true 現在是 int
                    binary_pred_masks_flat,  # y_pred 也是 int
                    zero_division=0)
                pixel_iou = jaccard_score(
                    all_gt_masks_flat,  # y_true 現在是 int
                    binary_pred_masks_flat,  # y_pred 也是 int
                    zero_division=0)

                print("-" * 50)
                print(f"Epoch {epoch} Anomaly Detection Metrics:")
                print(f"  - Pixel-level AUROC   : {pixel_auroc:.4f}")
                print(f"  - Pixel-level PR-AUC  : {pixel_pr_auc:.4f}")
                print(
                    f"  - Pixel-level Precision: {pixel_precision:.4f} (at threshold {threshold})"
                )
                print(
                    f"  - Pixel-level Recall  : {pixel_recall:.4f} (at threshold {threshold})"
                )
                print(
                    f"  - Pixel-level F1 Score: {pixel_f1:.4f} (at threshold {threshold})"
                )
                print(
                    f"  - Pixel-level IoU     : {pixel_iou:.4f} (at threshold {threshold})"
                )
                print("-" * 50)

                student_model.train()  # 切回訓練模式

                # --- 保存最佳模型 (根據 Pixel-level AUROC) ---
                # 僅在 pixel_auroc 不是 NaN 時進行比較
                if not np.isnan(
                        pixel_auroc) and pixel_auroc > best_pixel_auroc:
                    best_pixel_auroc = pixel_auroc
                    # best_auroc_epoch = epoch # 可以保存 epoch 號碼
                    save_path = os.path.join(
                        checkpoint_dir,
                        f"{obj_name}_best_auroc.pckl")  # 建議更名以區分
                    torch.save(student_model.state_dict(), save_path)
                    print(
                        f"✅ New best model saved at epoch {epoch} based on Pixel-level AUROC!"
                    )
                    print(f"   Best Pixel-level AUROC: {best_pixel_auroc:.4f}")

        torch.cuda.empty_cache()


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    """
    --gpu_id -2：自動選擇最佳GPU
    --gpu_id -1：強制使用CPU
    --gpu_id  0：使用GPU 0（原有行為）
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--gpu_id',
                        action='store',
                        type=int,
                        default=-2,
                        required=False,
                        help='GPU ID (-2: auto-select, -1: CPU)')
    args = parser.parse_args()

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
                 ['transistor'], ['tile'], ['cable'], ['zipper'],
                 ['toothbrush'], ['metal_nut'], ['hazelnut'], ['screw'],
                 ['grid'], ['wood']]

    if int(args.obj_id) == -1:
        obj_list = [
            'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
            'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
            'screw', 'grid', 'wood'
        ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    # 根據選擇的GPU執行
    if args.gpu_id == -1:
        # 使用CPU
        main(picked_classes, args)
    else:
        # 使用GPU
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)
