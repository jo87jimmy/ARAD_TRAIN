import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
import torchvision.transforms as transforms  # 導入 torchvision.transforms，用於圖像預處理和增強。
from PIL import Image  # 從 PIL (Pillow) 導入 Image，用於圖像加載和處理。
import numpy as np  # 導入 numpy 並命名為 np，用於數值計算和數組操作。
import matplotlib.pyplot as plt  # 導入 matplotlib.pyplot 用於數據可視化。
from model_unet import AnomalyDetectionModel  # 從 model_unet 模組導入 AnomalyDetectionModel 類，這是用於異常檢測的模型架構。

# --- 載入模型與權重 ---

# 設定參數 (必須與訓練時學生模型的參數一致)
IMG_CHANNELS = 3  # 定義圖像通道數，RGB 圖像為 3。
SEG_CLASSES = 2  # 定義分割類別數，這裡通常是 2（正常和異常）。
STUDENT_RECON_BASE = 64  # 定義學生模型重建網絡的基礎通道數。
STUDENT_DISC_BASE = 64  # 定義學生模型判別網絡的基礎通道數。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 檢查是否有可用的 CUDA 設備，否則使用 CPU。

# 實例化學生模型架構
student_model = AnomalyDetectionModel(  # 創建 AnomalyDetectionModel 的實例。
    recon_in=IMG_CHANNELS,  # 重建網絡輸入通道數。
    recon_out=IMG_CHANNELS,  # 重建網絡輸出通道數。
    recon_base=STUDENT_RECON_BASE,  # 重建網絡基礎通道數。
    disc_in=IMG_CHANNELS * 2,  # 判別網絡輸入通道數（通常是原圖 + 重建圖的拼接，所以是 2 倍）。
    disc_out=SEG_CLASSES,  # 判別網絡輸出類別數。
    disc_base=STUDENT_DISC_BASE  # 判別網絡基礎通道數。
).to(DEVICE)  # 將模型移動到指定的設備（GPU 或 CPU）。

# 載入訓練好的學生模型權重
model_weights_path = './student_model_checkpoints/student_model.pckl'  # 定義學生模型權重文件的路徑。
student_model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))  # 加載模型權重，並映射到正確的設備。

# --- 2. 設定為評估模式 ---
student_model.eval()  # 將模型設置為評估模式，這會影響 Dropout 和 BatchNorm 等層的行為。

# --- 3. 定義一個完整的推論函數 ---

def predict_anomaly(model, image_path, device):  # 定義預測異常的函數，接收模型、圖像路徑和設備作為參數。
    """
    對單張圖片進行異常檢測推論

    Args:
        model (nn.Module): 訓練好的學生模型
        image_path (str): 輸入圖片的路徑
        device (str): 'cuda' or 'cpu'

    Returns:
        tuple: (原始圖像, 重建圖像, 異常遮罩) 均為 numpy array
    """
    # 定義圖像預處理流程 (應與訓練時的驗證集/測試集流程一致)
    preprocess = transforms.Compose([  # 使用 transforms.Compose 組合多個預處理步驟。
        transforms.Resize((224, 224)),  # 將圖像大小調整為 224x224。
        transforms.ToTensor(),  # 將圖像轉換為 PyTorch 張量，並歸一化到 [0, 1]。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用 ImageNet 的均值和標準差進行歸一化。
    ])

    # 載入並預處理圖像
    image = Image.open(image_path).convert("RGB")  # 打開圖像並確保轉換為 RGB 模式。
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # 應用預處理增加 batch 維度，並移動到設備。

    # --- 4. 執行前向傳播 (在 no_grad 上下文中以節省資源) ---
    with torch.no_grad():  # 禁用梯度計算，節省內存並加速推論。
        # 推論時，我們只需要分割圖，但模型會同時返回重建圖
        # 我們不需要特徵圖，所以 return_feats=False
        recon_image_tensor, seg_map_logits = model(image_tensor, return_feats=False)  # 執行模型前向傳播，獲取重建圖像和分割 logits。

    # --- 5. 後處理輸出 ---

    # a. 處理分割圖
    # seg_map_logits 的形狀是 [1, 2, H, W]，其中 2 是類別數 (0:正常, 1:異常)
    # 使用 softmax 將 logits 轉換為機率
    seg_map_probs = torch.softmax(seg_map_logits, dim=1)  # 對分割 logits 應用 Softmax，計算每個類別的概率。
    # 使用 argmax 找出每個像素點機率最高的類別，得到 [1, H, W] 的預測遮罩
    anomaly_mask_tensor = torch.argmax(seg_map_probs, dim=1)  # 取概率最大的類別索引作為預測結果。

    # b. 將 Tensor 轉換為可用於顯示的 NumPy Array
    original_image_np = np.array(image.resize((224, 224)))  # 將原始圖像調整大小並轉換為 numpy 數組，用於顯示。

    # 反正規化重建圖像以便顯示
    recon_image_np = recon_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)  # 將重建張量轉換為 numpy 數組，並調整維度順序為 (H, W, C)。
    mean = np.array([0.485, 0.456, 0.406])  # 定義歸一化使用的均值。
    std = np.array([0.229, 0.224, 0.225])  # 定義歸一化使用的標準差。
    recon_image_np = std * recon_image_np + mean  # 進行反歸一化操作。
    recon_image_np = np.clip(recon_image_np, 0, 1)  # 將像素值限制在 [0, 1] 範圍內。

    anomaly_mask_np = anomaly_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)  # 將異常掩碼轉換為 numpy 數組，並轉為 uint8 類型。

    return original_image_np, recon_image_np, anomaly_mask_np  # 返回處理後的原始圖像、重建圖像和異常掩碼。


# --- 使用範例 ---
image_path_to_test = 'path/to/your/test_image.png'  # 定義要測試的圖像路徑（請根據實際情況修改）。
original, reconstruction, anomaly_mask = predict_anomaly(student_model, image_path_to_test, DEVICE)  # 調用預測函數進行推論。

# --- 可視化結果 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 創建一個包含 3 個子圖的圖像，設置總大小。
axes[0].imshow(original)  # 在第一個子圖顯示原始圖像。
axes[0].set_title('Original Image')  # 設置第一個子圖的標題。
axes[0].axis('off')  # 關閉第一個子圖的坐標軸。

axes[1].imshow(reconstruction)  # 在第二個子圖顯示重建圖像。
axes[1].set_title('Reconstructed Image')  # 設置第二個子圖的標題。
axes[1].axis('off')  # 關閉第二個子圖的坐標軸。

# 將異常遮罩（0和1）與原始圖像疊加顯示
axes[2].imshow(original)  # 在第三個子圖先顯示原始圖像。
axes[2].imshow(anomaly_mask, cmap='jet', alpha=0.4)  # 將異常掩碼以半透明方式疊加在原圖上，使用 'jet' 顏色映射。
axes[2].set_title('Anomaly Mask')  # 設置第三個子圖的標題。
axes[2].axis('off')  # 關閉第三個子圖的坐標軸。

plt.show()  # 顯示圖像。