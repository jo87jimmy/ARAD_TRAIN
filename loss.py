import numpy as np  # 導入 numpy 並命名為 np，用於數值計算。
import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
import torch.nn as nn  # 導入神經網絡模組並命名為 nn。
import torch.nn.functional as F  # 導入函數式接口並命名為 F，包含各種激活函數和損失函數等。
from math import exp  # 從 math 模組導入 exp，用於數學指數運算。

class FocalLoss(nn.Module):  # 定義 FocalLoss 類，繼承自 nn.Module。
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):  # 初始化函數。
        super(FocalLoss, self).__init__()  # 調用父類構造函數。
        self.apply_nonlin = apply_nonlin  # 保存可選的非線性變換函數（如 softmax）。
        self.alpha = alpha  # 保存 alpha 參數，用於類別加權。
        self.gamma = gamma  # 保存 gamma 參數，用於調節對難分類樣本的關注度。
        self.balance_index = balance_index  # 保存平衡類別的索引。
        self.smooth = smooth  # 保存標籤平滑的參數。
        self.size_average = size_average  # 保存是否對損失進行平均的標誌。

        if self.smooth is not None:  # 如果設置了平滑參數。
            if self.smooth < 0 or self.smooth > 1.0:  # 檢查平滑參數是否在合法範圍 [0, 1] 內。
                raise ValueError('smooth value should be in [0,1]')  # 如果不在範圍內，拋出錯誤。

    def forward(self, logit, target):  # 定義前向傳播函數。
        if self.apply_nonlin is not None:  # 如果定義了非線性變換。
            logit = self.apply_nonlin(logit)  # 對輸入 logit 應用非線性變換。
        num_class = logit.shape[1]  # 獲取類別數量。

        if logit.dim() > 2:  # 如果輸入 logit 的維度大於 2 (例如 [N, C, H, W])。
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)  # 將空間維度展平。
            logit = logit.permute(0, 2, 1).contiguous()  # 交換維度，使其變為 [N, m, C]。
            logit = logit.view(-1, logit.size(-1))  # 展平為 [N*m, C]。
        target = torch.squeeze(target, 1)  # 移除 target 的通道維度（如果是 1）。
        target = target.view(-1, 1)  # 將 target 展平為 [N*m, 1]。
        alpha = self.alpha  # 獲取 alpha 參數。

        if alpha is None:  # 如果未指定 alpha。
            alpha = torch.ones(num_class, 1)  # 創建全 1 的 alpha 張量。
        elif isinstance(alpha, (list, np.ndarray)):  # 如果 alpha 是列表或數組。
            assert len(alpha) == num_class  # 斷言 alpha 長度等於類別數。
            alpha = torch.FloatTensor(alpha).view(num_class, 1)  # 轉換為張量並調整形狀。
            alpha = alpha / alpha.sum()  # 歸一化 alpha。
        elif isinstance(alpha, float):  # 如果 alpha 是浮點數。
            alpha = torch.ones(num_class, 1)  # 創建全 1 的 alpha 張量。
            alpha = alpha * (1 - self.alpha)  # 使用 (1-alpha) 填充。
            alpha[self.balance_index] = self.alpha  # 設置平衡類別的 alpha 值。

        else:  # 如果 alpha 類型不支持。
            raise TypeError('Not support alpha type')  # 拋出類型錯誤。

        if alpha.device != logit.device:  # 確保 alpha 和 logit 在同一個設備上。
            alpha = alpha.to(logit.device)  # 移動 alpha 到目標設備。

        idx = target.cpu().long()  # 獲取目標索引，並轉換為 long 類型。

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()  # 創建全零的 one-hot 張量。
        one_hot_key = one_hot_key.scatter_(1, idx, 1)  # 根據索引填充 one-hot 張量。
        if one_hot_key.device != logit.device:  # 確保 one-hot 張量在正確的設備上。
            one_hot_key = one_hot_key.to(logit.device)  # 移動 one-hot 張量。

        if self.smooth:  # 如果啟用了標籤平滑。
            one_hot_key = torch.clamp(  # 對 one-hot 標籤進行平滑處理。
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth  # 計算預測概率 pt。
        logpt = pt.log()  # 計算 log(pt)。

        gamma = self.gamma  # 獲取 gamma 參數。

        alpha = alpha[idx]  # 獲取對應樣本的 alpha 值。
        alpha = torch.squeeze(alpha)  # 壓縮維度。
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt  # 計算 Focal Loss 公式：-alpha * (1-pt)^gamma * log(pt)。

        if self.size_average:  # 如果需要平均損失。
            loss = loss.mean()  # 計算平均值。
        return loss  # 返回損失值。

def gaussian(window_size, sigma):  # 定義高斯核生成函數。
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])  # 計算一維高斯分布。
    return gauss/gauss.sum()  # 歸一化高斯核。

def create_window(window_size, channel=1):  # 定義創建二維高斯窗口的函數。
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 創建一維窗口並增加維度。
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # 通過矩陣乘法創建二維窗口，並增加 batch 和 channel 維度。
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # 擴展窗口以匹配通道數。
    return window  # 返回高斯窗口。

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):  # 定義 SSIM 計算函數。
    if val_range is None:  # 如果未指定值範圍。
        if torch.max(img1) > 128:  # 判斷圖像是否為 0-255 範圍。
            max_val = 255
        else:
            max_val = 1  # 否則是 0-1 範圍。

        if torch.min(img1) < -0.5:  # 判斷是否歸一化到 -1 到 1。
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val  # 計算動態範圍 L。
    else:  # 如果指定了值範圍。
        l = val_range  # 使用指定的值範圍。

    padd = window_size//2  # 計算卷積的填充大小。
    (_, channel, height, width) = img1.size()  # 獲取圖像尺寸。
    if window is None:  # 如果未提供窗口。
        real_size = min(window_size, height, width)  # 計算實際窗口大小。
        window = create_window(real_size, channel=channel).to(img1.device)  # 創建高斯窗口並移動到設備。

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)  # 計算 img1 的局部均值（高斯模糊）。
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)  # 計算 img2 的局部均值。

    mu1_sq = mu1.pow(2)  # 計算均值的平方。
    mu2_sq = mu2.pow(2)  # 計算均值的平方。
    mu1_mu2 = mu1 * mu2  # 計算均值的乘積。

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq  # 計算 img1 的局部方差。
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq  # 計算 img2 的局部方差。
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2  # 計算協方差。

    c1 = (0.01 * l) ** 2  # 計算穩定常數 C1。
    c2 = (0.03 * l) ** 2  # 計算穩定常數 C2。

    v1 = 2.0 * sigma12 + c2  # 計算 SSIM 分子的一部分。
    v2 = sigma1_sq + sigma2_sq + c2  # 計算 SSIM 分母的一部分。
    cs = torch.mean(v1 / v2)  # 計算對比度敏感度（Contrast Sensitivity）。

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)  # 計算 SSIM 映射圖。

    if size_average:  # 如果需要平均。
        ret = ssim_map.mean()  # 計算 SSIM 均值。
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)  # 對每個樣本單獨計算均值。

    if full:  # 如果需要完整輸出。
        return ret, cs  # 返回 SSIM 和 CS。
    return ret, ssim_map  # 返回 SSIM 和 SSIM 映射圖。


class SSIM(torch.nn.Module):  # 定義 SSIM 模組類，繼承自 nn.Module。
    def __init__(self, window_size=11, size_average=True, val_range=None):  # 初始化函數。
        super(SSIM, self).__init__()  # 調用父類構造函數。
        self.window_size = window_size  # 保存窗口大小。
        self.size_average = size_average  # 保存平均標誌。
        self.val_range = val_range  # 保存值範圍。

        # Assume 1 channel for SSIM
        self.channel = 1  # 假設初始通道數為 1。
        self.window = create_window(window_size).cuda()  # 創建高斯窗口並移動到 CUDA（如果可用）。注意這裡強制了 .cuda()，可能需要根據實際情況調整設備。

    def forward(self, img1, img2):  # 定義前向傳播。
        (_, channel, _, _) = img1.size()  # 獲取輸入圖像的通道數。

        if channel == self.channel and self.window.dtype == img1.dtype:  # 如果通道數和數據類型匹配。
            window = self.window  # 使用緩存的窗口。
        else:  # 如果不匹配。
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)  # 重新創建適合的窗口。
            self.window = window  # 更新緩存。
            self.channel = channel  # 更新通道數。

        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)  # 計算 SSIM。
        return 1.0 - s_score  # 返回 DSSIM (1 - SSIM) 作為損失值。
