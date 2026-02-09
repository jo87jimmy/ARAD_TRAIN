import torch  # 導入 PyTorch 庫，這是深度學習的主要框架。
import math  # 導入 math 模組，用於數學運算。
import numpy as np  # 導入 numpy 並命名為 np，用於數值計算。


def lerp_np(x, y, w):  # 定義線性插值函數 (NumPy 版本)。
    fin_out = (y - x) * w + x  # 計算插值結果：x + w * (y - x)。
    return fin_out  # 返回結果。


def generate_fractal_noise_2d(
    shape, res, octaves=1, persistence=0.5
):  # 定義生成 2D 分形噪聲的函數。
    noise = np.zeros(shape)  # 初始化噪聲數組為全零。
    frequency = 1  # 初始頻率為 1。
    amplitude = 1  # 初始振幅為 1。
    for _ in range(octaves):  # 遍歷每個倍頻程。
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1])
        )  # 累加 Perlin 噪聲。
        frequency *= 2  # 頻率翻倍。
        amplitude *= persistence  # 振幅衰減。
    return noise  # 返回最終的分形噪聲。


def generate_perlin_noise_2d(shape, res):  # 定義生成 2D Perlin 噪聲的函數。
    def f(t):  # 定義緩入緩出函數 (Fade function)。
        return 6 * t**5 - 15 * t**4 + 10 * t**3  # 改進的平滑函數。

    delta = (res[0] / shape[0], res[1] / shape[1])  # 計算網格步長。
    d = (shape[0] // res[0], shape[1] // res[1])  # 計算每個網格單元的大小。
    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    )  # 生成網格坐標並取小數部分。
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)  # 生成隨機梯度角度。
    gradients = np.dstack((np.cos(angles), np.sin(angles)))  # 計算梯度向量 (cos, sin)。
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)  # 左上角梯度平鋪。
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)  # 右上角梯度平鋪。
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)  # 左下角梯度平鋪。
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)  # 右下角梯度平鋪。
    # Ramps
    n00 = np.sum(grid * g00, 2)  # 計算點積（左上）。
    n10 = np.sum(
        np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2
    )  # 計算點積（右上）。
    n01 = np.sum(
        np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2
    )  # 計算點積（左下）。
    n11 = np.sum(
        np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2
    )  # 計算點積（右下）。
    # Interpolation
    t = f(grid)  # 計算平滑係數。
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10  # 在 x 方向插值。
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11  # 在 x 方向插值。
    return np.sqrt(2) * (
        (1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1
    )  # 在 y 方向插值並返回結果。


def rand_perlin_2d_np(
    shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
):  # 定義隨機 2D Perlin 噪聲函數 (NumPy 版本)。
    delta = (res[0] / shape[0], res[1] / shape[1])  # 計算步長。
    d = (shape[0] // res[0], shape[1] // res[1])  # 計算單元大小。
    grid = (
        np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    )  # 生成網格。

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)  # 生成隨機角度。
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)  # 生成梯度。

    def tile_grads(slice1, slice2):
        """定義梯度平鋪函數。將梯度在兩個軸上重複擴展。"""
        return np.repeat(
            np.repeat(
                gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0
            ),
            d[1],
            axis=1,
        )

    def dot(grad, shift):
        """定義點積函數。計算梯度與位移向量的點積。"""
        return (
            np.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                axis=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])  # 計算左上角點積。
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])  # 計算右上角點積。
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])  # 計算左下角點積。
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])  # 計算右下角點積。
    t = fade(grid[: shape[0], : shape[1]])  # 計算插值係數。
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )  # 雙線性插值並返回。


def rand_perlin_2d(
    shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
):  # 定義隨機 2D Perlin 噪聲函數 (PyTorch 版本)。
    delta = (res[0] / shape[0], res[1] / shape[1])  # 計算步長。
    d = (shape[0] // res[0], shape[1] // res[1])  # 計算單元大小。

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])
            ),
            dim=-1,
        )
        % 1
    )  # 生成網格。
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)  # 生成隨機角度。
    gradients = torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )  # 生成梯度。

    def tile_grads(slice1, slice2):
        """定義梯度平鋪函數。將梯度在兩個軸上重複擴展 (PyTorch 版本)。"""
        return (
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
            .repeat_interleave(d[0], 0)
            .repeat_interleave(d[1], 1)
        )

    def dot(grad, shift):
        """定義點積函數。計算梯度與位移向量的點積 (PyTorch 版本)。"""
        return (
            torch.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                dim=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])  # 計算左上角點積。

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])  # 計算右上角點積。
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])  # 計算左下角點積。
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])  # 計算右下角點積。
    t = fade(grid[: shape[0], : shape[1]])  # 計算插值係數。
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
    )  # 雙線性插值並返回。


def rand_perlin_2d_octaves(
    shape, res, octaves=1, persistence=0.5
):  # 定義帶倍頻程的 Perlin 噪聲生成函數 (PyTorch 版本)。
    noise = torch.zeros(shape)  # 初始化噪聲。
    frequency = 1  # 初始頻率。
    amplitude = 1  # 初始振幅。
    for _ in range(octaves):  # 遍歷倍頻程。
        noise += amplitude * rand_perlin_2d(
            shape, (frequency * res[0], frequency * res[1])
        )  # 累加 Perlin 噪聲。
        frequency *= 2  # 頻率翻倍。
        amplitude *= persistence  # 振幅衰減。
    return noise  # 返回結果。
