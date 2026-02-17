"""
model.py

Замена diffusers-модели на SpaceTimeUnet из make-a-video-pytorch.

Архитектура:
  - Координаты хранятся внутри модели как register_buffer (не конкатенируются каждый раз)
  - Вход: (B, 2, T, 320, 256) — 2 физических канала, T кадров
  - Выход: (B, 2, T, 320, 256) — предсказание v
  - Flash attention через PyTorch SDPA (PyTorch >= 2.0)
  - Temporal attention между кадрами

Установка:
  pip install make-a-video-pytorch einops
"""

import torch
import torch.nn as nn
from make_a_video_pytorch import SpaceTimeUnet
from utils import make_normalized_xy_grid 


class VideoDiffusionModel(nn.Module):
    """
    Обёртка вокруг SpaceTimeUnet с встроенными координатами.

    Параметры
    ---------
    in_channels : int
        Число физических каналов данных. У тебя 2.
    dim : int
        Базовая ширина сети. Увеличь для большей ёмкости (64 → 128 → 256).
    dim_mults : tuple[int]
        Множители каналов по уровням UNet.
        (1, 2, 4, 8) при H=320, W=256 → downsample на 3 раза → 320/8=40, 256/8=32 ✓
    temporal_compression : tuple[bool]
        На каком уровне UNet применять temporal downsampling.
        Для начала лучше False везде — не сжимаем по времени.
    num_frames : int
        Число входных кадров (T). Используется только для документации,
        реальный T определяется входным тензором.
    coord_embed_dim : int
        Размерность, в которую проецируются координатыVi перед сложением.
        Должна совпадать с dim.
    """

    def __init__(
        self,
        in_channels: int = 2,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4, 8),
        temporal_compression: tuple = (False, False, False, False),
        num_frames: int = 8,
        coord_embed_dim: int = None,   # по умолчанию = dim
    ):
        super().__init__()

        coord_embed_dim = coord_embed_dim or dim

        # ── Координатный эмбеддинг ──────────────────────────────────────────
        # Вычисляем координаты один раз и храним внутри модели.
        # register_buffer:
        #   - не является обучаемым параметром
        #   - автоматически переезжает на GPU вместе с моделью (.to(device))
        #   - сохраняется в state_dict
        coords = make_normalized_xy_grid()   # (2, 320, 256)
        self.register_buffer('coords', coords)

        # Проекция координат (2 → coord_embed_dim) через 1×1 conv
        # Это learnable: модель сама учится как использовать координаты
        self.coord_proj = nn.Conv2d(
            in_channels=2,
            out_channels=coord_embed_dim,
            kernel_size=1,
            bias=True,
        )

        # ── Проекция входных данных (2 → dim) ──────────────────────────────
        # Входные данные и координатный эмбеддинг суммируются в пространстве dim
        self.input_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # ── Основная модель: SpaceTimeUnet ──────────────────────────────────
        # channels=dim: принимает тензор с dim каналами (после input_proj)
        # flash_attn=True: использует PyTorch SDPA (FlashAttention если доступно)
        # condition_on_timestep=True: модель принимает timestep как conditioning
        self.unet = SpaceTimeUnet(
            dim=dim,
            channels=dim,
            dim_mult=dim_mults,
            temporal_compression=temporal_compression,
            condition_on_timestep=True,
            flash_attn=True,
            attn_pos_bias=False,   # несовместимо с flash_attn — отключаем
        )

        # ── Выходная проекция (dim → in_channels) ──────────────────────────
        self.output_proj = nn.Conv2d(
            in_channels=dim,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.in_channels = in_channels
        self.dim = dim

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ---------
        x : torch.Tensor, shape (B, C, T, H, W)
            Зашумлённые кадры. C=2, T — число кадров, H=320, W=256.
        timesteps : torch.Tensor, shape (B,)
            Шаги диффузии в диапазоне [0, 1] или [0, 1000] — зависит от твоего шедулера.

        Возвращает
        ----------
        torch.Tensor, shape (B, C, T, H, W)
            Предсказание (шум или v — зависит от твоего add_noise).
        """
        B, C, T, H, W = x.shape

        # ── 1. Координатный эмбеддинг ───────────────────────────────────────
        # self.coords: (2, H, W) — один раз, без копирования по батчу
        coord_emb = self.coord_proj(self.coords)          # (dim, H, W)
        coord_emb = coord_emb.unsqueeze(0)                # (1, dim, H, W) — broadcast

        # ── 2. Проекция входных данных ──────────────────────────────────────
        # Применяем input_proj и coord_proj поканально для каждого кадра.
        # Flatten по времени → применяем conv → unflatten обратно.
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
        feat_flat = self.input_proj(x_flat)                           # (B*T, dim, H, W)
        feat_flat = feat_flat + coord_emb                             # broadcast по B*T — 0 копий!
        feat = feat_flat.reshape(B, T, self.dim, H, W)
        feat = feat.permute(0, 2, 1, 3, 4)                           # (B, dim, T, H, W)

        # ── 3. SpaceTimeUnet ────────────────────────────────────────────────
        # Spatial attention: по H×W для каждого кадра независимо
        # Temporal attention: по T для каждого пространственного патча
        out = self.unet(feat, timesteps)                              # (B, dim, T, H, W)

        # ── 4. Выходная проекция ────────────────────────────────────────────
        out_flat = out.permute(0, 2, 1, 3, 4).reshape(B * T, self.dim, H, W)
        pred_flat = self.output_proj(out_flat)                        # (B*T, C, H, W)
        pred = pred_flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        return pred