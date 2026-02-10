"""
U-Net 模型模組
================

本模組實現了用於異常檢測的 U-Net 架構變體。
主要包含兩個子網絡：
1. 重建子網絡 (ReconstructiveSubNetwork): 負責將輸入圖像編碼並重建，用於學習正常樣本的特徵。
2. 判別子網絡 (DiscriminativeSubNetwork): 接收原始圖像和重建圖像，負責分割出異常區域。

此架構結合了重建誤差和分割網絡，適用於 MVTec AD 等數據集的異常檢測任務。
"""

import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
from torch import nn  # 導入 PyTorch 的神經網絡模組。


class ReconstructiveSubNetwork(nn.Module):  # 定義重建子網絡類，繼承自 nn.Module。
    """
    重建子網絡 (Reconstructive SubNetwork)

    負責將輸入圖像進行編碼 (Encoder) 提取特徵，再解碼 (Decoder) 重建回原始圖像。
    此子網絡的目標是學習正常樣本的特徵分佈，通過重建誤差來輔助異常檢測。
    """

    def __init__(
        self, in_channels=3, out_channels=3, base_width=128
    ):  # 初始化函數，設置輸入輸出通道和基礎寬度。
        super().__init__()  # 調用父類構造函數。
        self.encoder = EncoderReconstructive(
            in_channels, base_width
        )  # 實例化重建編碼器。
        self.decoder = DecoderReconstructive(
            base_width, out_channels=out_channels
        )  # 實例化重建解碼器。

    def forward(self, x):
        """
        前向傳播函數。

        Args:
            x (Tensor): 輸入圖像張量。

        Returns:
            Tensor: 重建後的圖像張量。
        """
        b5 = self.encoder(x)  # 編碼器提取特徵。
        output = self.decoder(b5)  # 解碼器重建圖像。
        return output  # 返回輸出。


class EncoderReconstructive(nn.Module):  # 定義重建網絡的編碼器。
    """
    重建網絡編碼器 (Encoder Reconstructive)

    負責對輸入圖像進行下採樣 (Downsampling) 和特徵提取。
    通過一系列卷積層和池化層，將圖像轉換為高維特徵表示 (Latent Representation)，供解碼器使用。
    """

    def __init__(self, in_channels, base_width):  # 初始化函數。
        super().__init__()  # 調用父類構造函數。
        self.block1 = nn.Sequential(  # 第一個卷積塊。
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block2 = nn.Sequential(  # 第二個卷積塊。
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block3 = nn.Sequential(  # 第三個卷積塊。
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block4 = nn.Sequential(  # 第四個卷積塊。
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block5 = nn.Sequential(  # 第五個卷積塊。
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        前向傳播函數。

        Args:
            x (Tensor): 輸入圖像張量。

        Returns:
            Tensor: 編碼後的特徵張量 (b5)。
        """
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)  # 只返回最後一層特徵。
        return b5


class DecoderReconstructive(nn.Module):  # 定義重建網絡的解碼器。
    """
    重建網絡解碼器 (Decoder Reconstructive)

    負責將編碼器提取的特徵 (Latent Representation) 進行上採樣 (Upsampling) 和解碼。
    通過一系列上採樣層和卷積層，將特徵還原為原始圖像尺寸，完成圖像重建任務。
    """

    def __init__(self, base_width, out_channels=1):  # 初始化函數。
        super().__init__()  # 調用父類構造函數。

        self.up1 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),  # 沒有跳躍連接拼接。
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 8, base_width * 4, kernel_size=3, padding=1
            ),  # 通道數減半。
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 4, base_width * 2, kernel_size=3, padding=1
            ),  # 通道數減半。
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        # cat with base*1
        self.db3 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_width * 2, base_width * 1, kernel_size=3, padding=1
            ),  # 通道數減半。
            nn.BatchNorm2d(base_width * 1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        )  # 最終輸出。
        # self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):  # 定義前向傳播。
        """
        前向傳播函數。

        Args:
            b5 (Tensor): 編碼器輸出的瓶頸特徵 (Bottleneck Features)。

        Returns:
            Tensor: 重建後的圖像張量。
        """
        up1 = self.up1(b5)  # 上採樣。
        db1 = self.db1(up1)  # 解碼。

        up2 = self.up2(db1)  # 上採樣。
        db2 = self.db2(up2)  # 解碼。

        up3 = self.up3(db2)  # 上採樣。
        db3 = self.db3(up3)  # 解碼。

        up4 = self.up4(db3)  # 上採樣。
        db4 = self.db4(up4)  # 解碼。

        out = self.fin_out(db4)  # 輸出。
        return out  # 返回結果。


class DiscriminativeSubNetwork(
    nn.Module
):  # 定義判別子網絡類（用於分割），繼承自 nn.Module。
    """
    判別子網絡 (Discriminative SubNetwork)

    負責接收原始圖像和重建圖像的拼接輸入，並輸出異常區域的分割掩碼 (Segmentation Mask)。
    此子網絡透過比較原始圖像與重建圖像的差異，來定位圖像中的異常部分。
    """

    def __init__(
        self, in_channels=3, out_channels=3, base_channels=64, out_features=False
    ):  # 初始化函數。
        super().__init__()  # 調用父類構造函數。
        base_width = base_channels  # 設置基礎寬度。
        self.encoder_segment = EncoderDiscriminative(
            in_channels, base_width
        )  # 實例化判別編碼器。
        self.decoder_segment = DecoderDiscriminative(
            base_width, out_channels=out_channels
        )  # 實例化判別解碼器。
        # self.segment_act = torch.nn.Sigmoid()  # 註釋掉的 Sigmoid 激活層。
        self.out_features = out_features  # 是否輸出中間特徵。

    def forward(self, x):  # 定義前向傳播。
        """
        前向傳播函數。

        Args:
            x (Tensor): 輸入圖像張量。通常是原始圖像和重建圖像的拼接。

        Returns:
            Tensor or tuple: 分割掩碼張量 (output_segment)。
                            如果 out_features 為 True，則返回 (output_segment, features...) 元組。
        """
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)  # 編碼器提取多層特徵。
        output_segment = self.decoder_segment(
            b1, b2, b3, b4, b5, b6
        )  # 解碼器生成分割掩碼。
        if self.out_features:  # 如果需要輸出特徵。
            return output_segment, b2, b3, b4, b5, b6  # 返回分割結果和中間特徵。
        else:
            return output_segment  # 僅返回分割結果。


class EncoderDiscriminative(nn.Module):  # 定義判別網絡的編碼器。
    """
    判別網絡編碼器 (Encoder Discriminative)

    負責對輸入圖像(通常是原始圖像與重建圖像的拼接)進行特徵提取。
    通過一系列卷積層和池化層，提取多尺度的特徵圖 (Feature Maps)，並將其傳遞給解碼器進行分割。
    """

    def __init__(self, in_channels, base_width):  # 初始化函數。
        super().__init__()  # 調用父類構造函數。
        self.block1 = nn.Sequential(  # 第一個卷積塊。
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),  # 卷積層。
            nn.BatchNorm2d(base_width),  # 批次歸一化。
            nn.ReLU(inplace=True),  # ReLU 激活。
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),  # 卷積層。
            nn.BatchNorm2d(base_width),  # 批次歸一化。
            nn.ReLU(inplace=True),
        )  # ReLU 激活。
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層，下採樣。
        self.block2 = nn.Sequential(  # 第二個卷積塊。
            nn.Conv2d(
                base_width, base_width * 2, kernel_size=3, padding=1
            ),  # 通道數加倍。
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block3 = nn.Sequential(  # 第三個卷積塊。
            nn.Conv2d(
                base_width * 2, base_width * 4, kernel_size=3, padding=1
            ),  # 通道數加倍。
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block4 = nn.Sequential(  # 第四個卷積塊。
            nn.Conv2d(
                base_width * 4, base_width * 8, kernel_size=3, padding=1
            ),  # 通道數加倍。
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block5 = nn.Sequential(  # 第五個卷積塊。
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),  # 通道數不變。
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block6 = nn.Sequential(  # 第六個卷積塊。
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),  # 通道數不變。
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # 定義前向傳播。
        """
        前向傳播函數。

        Args:
            x (Tensor): 輸入圖像張量。

        Returns:
            tuple: 包含各個層級特徵圖 (b1, b2, b3, b4, b5, b6) 的元組。
        """
        b1 = self.block1(x)  # 通過 block1。
        mp1 = self.mp1(b1)  # 下採樣。
        b2 = self.block2(mp1)  # 通過 block2。
        mp2 = self.mp2(b2)  # 下採樣。
        b3 = self.block3(mp2)  # 通過 block3。
        mp3 = self.mp3(b3)  # 下採樣。
        b4 = self.block4(mp3)  # 通過 block4。
        mp4 = self.mp4(b4)  # 下採樣。
        b5 = self.block5(mp4)  # 通過 block5。
        mp5 = self.mp5(b5)  # 下採樣。
        b6 = self.block6(mp5)  # 通過 block6。
        return b1, b2, b3, b4, b5, b6  # 返回所有層級的特徵。


class DecoderDiscriminative(nn.Module):  # 定義判別網絡的解碼器。
    """
    判別網絡解碼器 (Decoder Discriminative)

    負責接收判別編碼器 (Encoder Discriminative) 的多尺度特徵，
    通過跳躍連接 (Skip Connections) 和上採樣 (Upsampling) 生成最終的異常分割掩碼。
    """

    def __init__(self, base_width, out_channels=1):  # 初始化函數。
        super().__init__()  # 調用父類構造函數。

        self.up_b = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(
                base_width * 8, base_width * 8, kernel_size=3, padding=1
            ),  # 卷積。
            nn.BatchNorm2d(base_width * 8),  # 批次歸一化。
            nn.ReLU(inplace=True),
        )  # ReLU。
        self.db_b = nn.Sequential(  # 解碼塊。
            nn.Conv2d(
                base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1
            ),  # 接收拼接後的特徵（跳躍連接）。
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(
                base_width * 8, base_width * 4, kernel_size=3, padding=1
            ),  # 通道數減半。
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(
                base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1
            ),  # 拼接後卷積。
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # 上採樣。
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        )  # 最終輸出層。

    def forward(self, b1, b2, b3, b4, b5, b6):
        """
        前向傳播函數。

        Args:
            b1 (Tensor): 編碼器第一層特徵圖。
            b2 (Tensor): 編碼器第二層特徵圖。
            b3 (Tensor): 編碼器第三層特徵圖。
            b4 (Tensor): 編碼器第四層特徵圖。
            b5 (Tensor): 編碼器第五層特徵圖。
            b6 (Tensor): 編碼器第六層特徵圖。

        Returns:
            Tensor: 最終生成的異常分割掩碼。
        """
        up_b = self.up_b(b6)  # 上採樣 b6。
        cat_b = torch.cat((up_b, b5), dim=1)  # 與 b5 拼接。
        db_b = self.db_b(cat_b)  # 解碼塊處理。

        up1 = self.up1(db_b)  # 上採樣。
        cat1 = torch.cat((up1, b4), dim=1)  # 與 b4 拼接。
        db1 = self.db1(cat1)  # 解碼塊處理。

        up2 = self.up2(db1)  # 上採樣。
        cat2 = torch.cat((up2, b3), dim=1)  # 與 b3 拼接。
        db2 = self.db2(cat2)  # 解碼塊處理。

        up3 = self.up3(db2)  # 上採樣。
        cat3 = torch.cat((up3, b2), dim=1)  # 與 b2 拼接。
        db3 = self.db3(cat3)  # 解碼塊處理。

        up4 = self.up4(db3)  # 上採樣。
        cat4 = torch.cat((up4, b1), dim=1)  # 與 b1 拼接。
        db4 = self.db4(cat4)  # 解碼塊處理。

        out = self.fin_out(db4)  # 生成最終輸出。
        return out  # 返回結果。


class AnomalyDetectionModel(nn.Module):
    """
    異常檢測模型組合類別 (Anomaly Detection Model Wrapper)

    將重建子網絡 (Reconstructive SubNetwork) 與判別子網絡 (Discriminative SubNetwork)
    封裝在一個 nn.Module 中，方便進行端到端 (End-to-End) 的推論與管理。
    主要流程：
    1. 輸入圖像通過重建子網絡生成的重建圖像。
    2. 原始圖像與重建圖像拼接。
    3. 拼接後的圖像通過判別子網絡生成異常分割掩碼。
    """

    def __init__(
        self,
        recon_in=3,
        recon_out=3,
        recon_base=64,
        disc_in=6,
        disc_out=2,
        disc_base=64,
    ):
        """
        初始化異常檢測模型。

        Args:
            recon_in (int): 重建子網絡輸入通道數 (預設: 3)。
            recon_out (int): 重建子網絡輸出通道數 (預設: 3)。
            recon_base (int): 重建子網絡基礎寬度 (預設: 64)。
            disc_in (int): 判別子網絡輸入通道數 (預設: 6, 通常為原始圖像+重建圖像)。
            disc_out (int): 判別子網絡輸出通道數 (預設: 2, 例如前景/背景)。
            disc_base (int): 判別子網絡基礎寬度 (預設: 64)。
        """
        super(AnomalyDetectionModel, self).__init__()
        # 初始化重建子網絡
        self.reconstructive = ReconstructiveSubNetwork(
            in_channels=recon_in, out_channels=recon_out, base_width=recon_base
        )
        # 初始化判別子網絡
        self.discriminative = DiscriminativeSubNetwork(
            in_channels=disc_in,
            out_channels=disc_out,
            base_channels=disc_base,
            out_features=False,  # 預設不輸出特徵，符合 eval.py 的使用場景
        )

    def forward(self, x, return_feats=False):
        """
        前向傳播函數

        Args:
            x (Tensor): 輸入圖像 Tensor [Batch, Channel, Height, Width]
            return_feats (bool): 是否返回中間特徵 (目前主要兼容介面，具體依賴子網絡實作)

        Returns:
            tuple: (重建圖像, 分割 Logits)
        """
        # 1. 重建圖像
        recon = self.reconstructive(x)

        # 2. 拼接重建圖像與原始圖像作為判別網絡輸入
        # 在 channel 維度 (dim=1) 上拼接
        joined_in = torch.cat((recon, x), dim=1)

        # 3. 生成分割 Logits
        seg_out = self.discriminative(joined_in)

        # 處理可能的特徵返回 (雖然目前 DiscriminativeSubNetwork 設為 False)
        if isinstance(seg_out, tuple):
            if return_feats:
                return recon, seg_out[0]  # 这里假设外部只需要 seg_out[0]
            seg_out = seg_out[0]

        return recon, seg_out
