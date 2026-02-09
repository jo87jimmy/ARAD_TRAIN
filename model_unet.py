import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
import torch.nn as nn  # 導入 PyTorch 的神經網絡模組。


class ReconstructiveSubNetwork(nn.Module):  # 定義重建子網絡類，繼承自 nn.Module。
    def __init__(self,in_channels=3, out_channels=3, base_width=128):  # 初始化函數，設置輸入輸出通道和基礎寬度。
        super(ReconstructiveSubNetwork, self).__init__()  # 調用父類構造函數。
        self.encoder = EncoderReconstructive(in_channels, base_width)  # 實例化重建編碼器。
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)  # 實例化重建解碼器。

    def forward(self, x):  # 定義前向傳播。
        b5 = self.encoder(x)  # 編碼器提取特徵。
        output = self.decoder(b5)  # 解碼器重建圖像。
        return output  # 返回輸出。

class EncoderReconstructive(nn.Module):  # 定義重建網絡的編碼器。
    def __init__(self, in_channels, base_width):  # 初始化函數。
        super(EncoderReconstructive, self).__init__()  # 調用父類構造函數。
        self.block1 = nn.Sequential(  # 第一個卷積塊。
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block2 = nn.Sequential(  # 第二個卷積塊。
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block3 = nn.Sequential(  # 第三個卷積塊。
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block4 = nn.Sequential(  # 第四個卷積塊。
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))  # 池化。
        self.block5 = nn.Sequential(  # 第五個卷積塊。
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):  # 定義前向傳播。
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
    def __init__(self, base_width, out_channels=1):  # 初始化函數。
        super(DecoderReconstructive, self).__init__()  # 調用父類構造函數。

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),  # 沒有跳躍連接拼接。
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),  # 通道數減半。
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),  # 通道數減半。
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width * 2, base_width*2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width*2),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*1, kernel_size=3, padding=1),  # 通道數減半。
            nn.BatchNorm2d(base_width*1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))  # 最終輸出。
        #self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):  # 定義前向傳播。
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

class DiscriminativeSubNetwork(nn.Module):  # 定義判別子網絡類（用於分割），繼承自 nn.Module。
    def __init__(self,in_channels=3, out_channels=3, base_channels=64, out_features=False):  # 初始化函數。
        super(DiscriminativeSubNetwork, self).__init__()  # 調用父類構造函數。
        base_width = base_channels  # 設置基礎寬度。
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)  # 實例化判別編碼器。
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)  # 實例化判別解碼器。
        #self.segment_act = torch.nn.Sigmoid()  # 註釋掉的 Sigmoid 激活層。
        self.out_features = out_features  # 是否輸出中間特徵。
    def forward(self, x):  # 定義前向傳播。
        b1,b2,b3,b4,b5,b6 = self.encoder_segment(x)  # 編碼器提取多層特徵。
        output_segment = self.decoder_segment(b1,b2,b3,b4,b5,b6)  # 解碼器生成分割掩碼。
        if self.out_features:  # 如果需要輸出特徵。
            return output_segment, b2, b3, b4, b5, b6  # 返回分割結果和中間特徵。
        else:
            return output_segment  # 僅返回分割結果。

class EncoderDiscriminative(nn.Module):  # 定義判別網絡的編碼器。
    def __init__(self, in_channels, base_width):  # 初始化函數。
        super(EncoderDiscriminative, self).__init__()  # 調用父類構造函數。
        self.block1 = nn.Sequential(  # 第一個卷積塊。
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),  # 卷積層。
            nn.BatchNorm2d(base_width),  # 批次歸一化。
            nn.ReLU(inplace=True),  # ReLU 激活。
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),  # 卷積層。
            nn.BatchNorm2d(base_width),  # 批次歸一化。
            nn.ReLU(inplace=True))  # ReLU 激活。
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層，下採樣。
        self.block2 = nn.Sequential(  # 第二個卷積塊。
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),  # 通道數加倍。
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block3 = nn.Sequential(  # 第三個卷積塊。
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),  # 通道數加倍。
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block4 = nn.Sequential(  # 第四個卷積塊。
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),  # 通道數加倍。
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block5 = nn.Sequential(  # 第五個卷積塊。
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),  # 通道數不變。
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))  # 最大池化層。
        self.block6 = nn.Sequential(  # 第六個卷積塊。
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),  # 通道數不變。
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):  # 定義前向傳播。
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
        return b1,b2,b3,b4,b5,b6  # 返回所有層級的特徵。

class DecoderDiscriminative(nn.Module):  # 定義判別網絡的解碼器。
    def __init__(self, base_width, out_channels=1):  # 初始化函數。
        super(DecoderDiscriminative, self).__init__()  # 調用父類構造函數。

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),  # 卷積。
                                 nn.BatchNorm2d(base_width * 8),  # 批次歸一化。
                                 nn.ReLU(inplace=True))  # ReLU。
        self.db_b = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),  # 接收拼接後的特徵（跳躍連接）。
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),  # 通道數減半。
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),  # 拼接後卷積。
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 上採樣。
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(  # 解碼塊。
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )



        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))  # 最終輸出層。

    def forward(self, b1,b2,b3,b4,b5,b6):  # 定義前向傳播，接收編碼器的各層特徵。
        up_b = self.up_b(b6)  # 上採樣 b6。
        cat_b = torch.cat((up_b,b5),dim=1)  # 與 b5 拼接。
        db_b = self.db_b(cat_b)  # 解碼塊處理。

        up1 = self.up1(db_b)  # 上採樣。
        cat1 = torch.cat((up1,b4),dim=1)  # 與 b4 拼接。
        db1 = self.db1(cat1)  # 解碼塊處理。

        up2 = self.up2(db1)  # 上採樣。
        cat2 = torch.cat((up2,b3),dim=1)  # 與 b3 拼接。
        db2 = self.db2(cat2)  # 解碼塊處理。

        up3 = self.up3(db2)  # 上採樣。
        cat3 = torch.cat((up3,b2),dim=1)  # 與 b2 拼接。
        db3 = self.db3(cat3)  # 解碼塊處理。

        up4 = self.up4(db3)  # 上採樣。
        cat4 = torch.cat((up4,b1),dim=1)  # 與 b1 拼接。
        db4 = self.db4(cat4)  # 解碼塊處理。

        out = self.fin_out(db4)  # 生成最終輸出。
        return out  # 返回結果。

