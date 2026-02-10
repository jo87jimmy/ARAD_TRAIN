"""
此模組定義了用於模型訓練的自定義損失函數。
包含 Focal Loss 和 SSIM (Structural Similarity Index) Loss。

原因：
- Focal Loss 用於解決類別不平衡問題，透過降低易分類樣本的權重，使模型更專注於難分類的樣本。
- SSIM Loss 用於衡量圖像結構的相似度，相較於 MSE 更符合人類視覺感知，適合用於圖像重建或增強任務。
"""

from math import exp  # 從 math 模組導入 exp，用於數學指數運算。
import numpy as np  # 導入 numpy 並命名為 np，用於數值計算。
import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
from torch import nn  # 導入神經網絡模組並命名為 nn。
from torch.nn import (
    functional,
)  # 導入函數式接口並命名為 functional，包含各種激活函數和損失函數等。為避免變數名稱衝突，建議使用完整名稱。


class FocalLoss(nn.Module):  # 定義 FocalLoss 類，繼承自 nn.Module。
    """
    複製自https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    這是 Focal Loss 的實現，支持平滑標籤交叉熵，該實現由以下論文提出：
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss = -1 * alpha * (1 - pt) * log(pt)
    :param alpha: (tensor) 3D 或 4D 此準則的標量因子
    :param gamma: (float, double) gamma > 0 會降低分類良好的樣本 (p > 0.5) 的相對損失，從而更關注難以分類的樣本
    :param smooth: (float, double) 交叉熵的平滑值
    :param balance_index: (int) 平衡類別索引，當 alpha 為浮點數時應明確指定
    :param size_average: (bool可選) 預設情況下，損失值在批次中的每個損失元素上取平均值。
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):  # 初始化函數。
        super(FocalLoss, self).__init__()  # 調用父類構造函數。
        self.apply_nonlin = apply_nonlin  # 保存可選的非線性變換函數（如 softmax）。
        self.alpha = alpha  # 保存 alpha 參數，用於類別加權。
        self.gamma = gamma  # 保存 gamma 參數，用於調節對難分類樣本的關注度。
        self.balance_index = balance_index  # 保存平衡類別的索引。
        self.smooth = smooth  # 保存標籤平滑的參數。
        self.size_average = size_average  # 保存是否對損失進行平均的標誌。

        if self.smooth is not None:  # 如果設置了平滑參數。
            if (
                self.smooth < 0 or self.smooth > 1.0
            ):  # 檢查平滑參數是否在合法範圍 [0, 1] 內。
                raise ValueError(
                    "smooth value should be in [0,1]"
                )  # 如果不在範圍內，拋出錯誤。

    def forward(self, logit, target):
        """
        計算 Focal Loss 的前向傳播。

        Args:
            logit (torch.Tensor): 模型的輸出預測值。
            target (torch.Tensor): 真實標籤。

        Returns:
            torch.Tensor: 計算得到的損失值。

        原因：
        - 實作 Focal Loss 的核心計算邏輯。
        - 根據論文公式 -alpha * (1 - pt)^gamma * log(pt) 計算損失。
        - 支援多維度輸入 (如 2D 影像分割)，自動調整維度以進行計算。
        - 處理 Alpha 平衡因子與 Gamma 調節因子，以解決樣本不平衡與難易分類問題。
        """
        if self.apply_nonlin is not None:  # 如果定義了非線性變換。
            logit = self.apply_nonlin(logit)  # 對輸入 logit 應用非線性變換。
        num_class = logit.shape[1]  # 獲取類別數量。

        if logit.dim() > 2:  # 如果輸入 logit 的維度大於 2 (例如 [N, C, H, W])。
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)  # 將空間維度展平。
            logit = logit.permute(
                0, 2, 1
            ).contiguous()  # 交換維度，使其變為 [N, m, C]。
            logit = logit.view(-1, logit.size(-1))  # 展平為 [N*m, C]。
        target = torch.squeeze(target, 1)  # 移除 target 的通道維度（如果是 1）。
        target = target.view(-1, 1)  # 將 target 展平為 [N*m, 1]。
        alpha = self.alpha  # 獲取 alpha 參數。

        if alpha is None:  # 如果未指定 alpha。
            alpha = torch.ones(num_class, 1)  # 創建全 1 的 alpha 張量。
        elif isinstance(alpha, (list, np.ndarray)):  # 如果 alpha 是列表或數組。
            assert len(alpha) == num_class  # 斷言 alpha 長度等於類別數。
            alpha = torch.FloatTensor(alpha).view(
                num_class, 1
            )  # 轉換為張量並調整形狀。
            alpha = alpha / alpha.sum()  # 歸一化 alpha。
        elif isinstance(alpha, float):  # 如果 alpha 是浮點數。
            alpha = torch.ones(num_class, 1)  # 創建全 1 的 alpha 張量。
            alpha = alpha * (1 - self.alpha)  # 使用 (1-alpha) 填充。
            alpha[self.balance_index] = self.alpha  # 設置平衡類別的 alpha 值。

        else:  # 如果 alpha 類型不支持。
            raise TypeError("Not support alpha type")  # 拋出類型錯誤。

        if alpha.device != logit.device:  # 確保 alpha 和 logit 在同一個設備上。
            alpha = alpha.to(logit.device)  # 移動 alpha 到目標設備。

        idx = target.cpu().long()  # 獲取目標索引，並轉換為 long 類型。

        one_hot_key = torch.FloatTensor(
            target.size(0), num_class
        ).zero_()  # 創建全零的 one-hot 張量。
        one_hot_key = one_hot_key.scatter_(1, idx, 1)  # 根據索引填充 one-hot 張量。
        if one_hot_key.device != logit.device:  # 確保 one-hot 張量在正確的設備上。
            one_hot_key = one_hot_key.to(logit.device)  # 移動 one-hot 張量。

        if self.smooth:  # 如果啟用了標籤平滑。
            one_hot_key = torch.clamp(  # 對 one-hot 標籤進行平滑處理。
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth  # 計算預測概率 pt。
        logpt = pt.log()  # 計算 log(pt)。

        gamma = self.gamma  # 獲取 gamma 參數。

        alpha = alpha[idx]  # 獲取對應樣本的 alpha 值。
        alpha = torch.squeeze(alpha)  # 壓縮維度。
        loss = (
            -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        )  # 計算 Focal Loss 公式：-alpha * (1-pt)^gamma * log(pt)。

        if self.size_average:  # 如果需要平均損失。
            loss = loss.mean()  # 計算平均值。
        return loss  # 返回損失值。


def gaussian(window_size, sigma):
    """
    生成一維高斯分佈窗口。

    Args:
        window_size (int): 高斯窗口的大小。
        sigma (float): 高斯分佈的標準差。

    Returns:
        torch.Tensor: 歸一化後的一維高斯窗口張量。

    原因：
    - 用於構建 SSIM 計算所需的高斯加權窗口。
    - 高斯分佈能模擬人眼對中心區域的關注度。
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )  # 計算一維高斯分布。
    return gauss / gauss.sum()  # 歸一化高斯核。


def create_window(window_size, channel=1):
    """
    創建二維高斯窗口。

    Args:
        window_size (int): 窗口大小。
        channel (int): 圖像通道數。

    Returns:
        torch.Tensor: 擴展到指定通道數的二維高斯窗口。

    原因：
    - SSIM 計算是在二維圖像上進行的，需要二維的高斯核。
    - 透過外積將一維高斯核轉換為二維。
    - 根據通道數擴展窗口，以便對多通道圖像進行卷積操作。
    """
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)  # 創建一維窗口並增加維度。
    window_2d = (
        window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    )  # 通過矩陣乘法創建二維窗口，並增加 batch 和 channel 維度。
    window = window_2d.expand(
        channel, 1, window_size, window_size
    ).contiguous()  # 擴展窗口以匹配通道數。
    return window  # 返回高斯窗口。


def ssim(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=None,
):
    """
    計算兩張圖像之間的結構相似性 (SSIM) 指標。

    Args:
        img1 (torch.Tensor): 第一張圖像 (通常是預測值)。
        img2 (torch.Tensor): 第二張圖像 (通常是真實值)。
        window_size (int): 高斯窗口大小。 default: 11
        window (torch.Tensor, optional): 預定義的高斯窗口。 default: None
        size_average (bool): 是否對結果取平均。 default: True
        full (bool): 是否返回完整的 SSIM 圖和對比度敏感度。 default: False
        val_range (float, optional): 圖像像素值的範圍 (如 255 或 1)。 default: None

    Returns:
        tuple or torch.Tensor: 根據參數返回 SSIM 值、SSIM 圖或 CS 值。

    原因：
    - SSIM 是一種衡量兩張圖像結構相似度的指標，比 MSE 更符合人類視覺感知。
    - 分別計算亮度、對比度和結構的相似度。
    - 使用卷積計算局部均值和方差，實現滑動窗口式的 SSIM 計算。
    """
    if val_range is None:  # 如果未指定值範圍。
        if torch.max(img1) > 128:  # 判斷圖像是否為 0-255 範圍。
            max_val = 255
        else:
            max_val = 1  # 否則是 0-1 範圍。

        if torch.min(img1) < -0.5:  # 判斷是否歸一化到 -1 到 1。
            min_val = -1
        else:
            min_val = 0
        dynamic_range = max_val - min_val  # 計算動態範圍。
    else:  # 如果指定了值範圍。
        dynamic_range = val_range  # 使用指定的值範圍。

    padd = window_size // 2  # 計算卷積的填充大小。
    (_, channel, height, width) = img1.size()  # 獲取圖像尺寸。
    if window is None:  # 如果未提供窗口。
        real_size = min(window_size, height, width)  # 計算實際窗口大小。
        window = create_window(real_size, channel=channel).to(
            img1.device
        )  # 創建高斯窗口並移動到設備。

    mu1 = functional.conv2d(
        img1, window, padding=padd, groups=channel
    )  # 使用 functional.conv2d 計算 img1 的局部均值（高斯模糊）。避免使用缩寫 F 以防止變數名稱衝突。
    mu2 = functional.conv2d(
        img2, window, padding=padd, groups=channel
    )  # 使用 functional.conv2d 計算 img2 的局部均值。

    mu1_sq = mu1.pow(2)  # 計算均值的平方。
    mu2_sq = mu2.pow(2)  # 計算均值的平方。
    mu1_mu2 = mu1 * mu2  # 計算均值的乘積。

    sigma1_sq = (
        functional.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    )  # 計算 img1 的局部方差。使用 functional.conv2d 替代 F.conv2d。
    sigma2_sq = (
        functional.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    )  # 計算 img2 的局部方差。
    sigma12 = (
        functional.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    )  # 計算協方差。

    c1 = (0.01 * dynamic_range) ** 2  # 計算穩定常數 C1。
    c2 = (0.03 * dynamic_range) ** 2  # 計算穩定常數 C2。

    v1 = 2.0 * sigma12 + c2  # 計算 SSIM 分子的一部分。
    v2 = sigma1_sq + sigma2_sq + c2  # 計算 SSIM 分母的一部分。
    cs = torch.mean(v1 / v2)  # 計算對比度敏感度（Contrast Sensitivity）。

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / (
        (mu1_sq + mu2_sq + c1) * v2
    )  # 計算 SSIM 映射圖。

    if size_average:  # 如果需要平均。
        ret = ssim_map.mean()  # 計算 SSIM 均值。
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)  # 對每個樣本單獨計算均值。

    if full:  # 如果需要完整輸出。
        return ret, cs  # 返回 SSIM 和 CS。
    return ret, ssim_map  # 返回 SSIM 和 SSIM 映射圖。


class SSIM(torch.nn.Module):
    """
    SSIM (Structural Similarity) 損失函數模組。

    原因：
    - 封裝 SSIM 計算邏輯為 PyTorch 模組，方便在神經網絡訓練中使用。
    - 自動管理高斯窗口的創建和設備移動，提高易用性。
    """

    def __init__(
        self, window_size=11, size_average=True, val_range=None
    ):  # 初始化函數。
        super(SSIM, self).__init__()  # 調用父類構造函數。
        self.window_size = window_size  # 保存窗口大小。
        self.size_average = size_average  # 保存平均標誌。
        self.val_range = val_range  # 保存值範圍。

        # Assume 1 channel for SSIM
        self.channel = 1  # 假設初始通道數為 1。
        self.window = create_window(
            window_size
        ).cuda()  # 創建高斯窗口並移動到 CUDA（如果可用）。注意這裡強制了 .cuda()，可能需要根據實際情況調整設備。

    def forward(self, img1, img2):
        """
        計算 SSIM 損失的前向傳播。

        Args:
            img1 (torch.Tensor): 預測圖像。
            img2 (torch.Tensor): 目標圖像。

        Returns:
            torch.Tensor: 計算得到的 DSSIM 損失值 (1 - SSIM)。

        原因：
        - 在訓練過程中計算預測圖像與目標圖像的結構差異。
        - 返回 1 - SSIM 作為損失值，因為優化器通常最小化損失，而 SSIM 越高越好。
        """
        (_, channel, _, _) = img1.size()  # 獲取輸入圖像的通道數。

        if (
            channel == self.channel and self.window.dtype == img1.dtype
        ):  # 如果通道數和數據類型匹配。
            window = self.window  # 使用緩存的窗口。
        else:  # 如果不匹配。
            window = (
                create_window(self.window_size, channel)
                .to(img1.device)
                .type(img1.dtype)
            )  # 重新創建適合的窗口。
            self.window = window  # 更新緩存。
            self.channel = channel  # 更新通道數。

        s_score, ssim_map = ssim(
            img1,
            img2,
            window=window,
            window_size=self.window_size,
            size_average=self.size_average,
        )  # 計算 SSIM。
        return 1.0 - s_score  # 返回 DSSIM (1 - SSIM) 作為損失值。
