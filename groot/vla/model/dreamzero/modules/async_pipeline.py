"""Async preprocessing and VAE pipeline for overlapping with diffusion.

Key idea: run video preprocessing + VAE encode on a separate CUDA stream
while the previous chunk's diffusion is still running.

Usage:
    pipe = AsyncPreprocessPipeline(vae, device)

    # Submit preprocessing (returns immediately, runs on background stream)
    pipe.submit_preprocess(videos, state_features)

    # ... diffusion from previous chunk runs on default stream ...

    # Get results (waits for background stream to finish)
    videos_processed, latent, state_bf16 = pipe.get_results()
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class AsyncPreprocessPipeline:
    """Overlaps video preprocessing + VAE with compute on the default stream."""

    def __init__(self, vae: nn.Module, device: torch.device,
                 tiled: bool = True,
                 tile_size: tuple = (34, 34),
                 tile_stride: tuple = (18, 16),
                 normalize_mean: tuple = (0.5, 0.5, 0.5),
                 normalize_std: tuple = (0.5, 0.5, 0.5)):
        self.vae = vae
        self.device = device
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride

        # Pre-allocate normalization constants on GPU
        self._mean = torch.tensor(normalize_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(normalize_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        # Background stream for preprocessing
        self._stream = torch.cuda.Stream(device=device)
        self._result = None
        self._submitted = False

    def preprocess_video_sync(self, videos: torch.Tensor) -> torch.Tensor:
        """Fast video preprocessing: uint8 → normalized bfloat16.

        Optimized: single-pass normalize + dtype convert.
        """
        # videos: [B, C, T, H, W] uint8 or float
        if videos.dtype == torch.uint8:
            # Fused: uint8 → float32 → normalize → bfloat16
            b, c, t, h, w = videos.shape
            # Reshape for normalize: [B*T, C, H, W]
            v = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
            v = v / 255.0
            v = (v - self._mean) / self._std
            v = v.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [B, C, T, H, W]
            return v.to(dtype=torch.bfloat16)
        else:
            return videos.to(dtype=torch.bfloat16)

    def submit_preprocess_and_vae(self, videos: torch.Tensor,
                                   state_features: torch.Tensor,
                                   need_vae: bool = True,
                                   num_frame_per_block: int = 2) -> None:
        """Submit preprocessing + VAE to background stream. Returns immediately."""
        self._submitted = True

        # Record event on default stream so background stream waits for data transfer
        default_stream = torch.cuda.current_stream(self.device)
        transfer_done = torch.cuda.Event()
        transfer_done.record(default_stream)

        with torch.cuda.stream(self._stream):
            # Wait for input data to be ready
            self._stream.wait_event(transfer_done)

            # Preprocess video
            videos_processed = self.preprocess_video_sync(videos)
            state_bf16 = state_features.to(dtype=torch.bfloat16)

            # VAE encode (the expensive part: ~0.074s)
            latent = None
            if need_vae:
                # Prepare frames for VAE
                first_frame = videos_processed[:, :, 0:1]
                vae_input = torch.cat([first_frame, videos_processed], dim=2)
                with torch.no_grad():
                    latent = self.vae.encode(
                        vae_input,
                        tiled=self.tiled,
                        tile_size=self.tile_size,
                        tile_stride=self.tile_stride,
                    )

            self._result = (videos_processed, latent, state_bf16)

    def get_results(self) -> tuple:
        """Wait for background preprocessing to finish and return results."""
        if not self._submitted:
            raise RuntimeError("No preprocessing submitted")

        # Wait for background stream to finish
        event = torch.cuda.Event()
        event.record(self._stream)
        torch.cuda.current_stream(self.device).wait_event(event)

        self._submitted = False
        return self._result

    def submit_vae_only(self, videos_processed: torch.Tensor,
                        num_frame_per_block: int = 2) -> None:
        """Submit just VAE encode to background stream."""
        self._submitted = True
        default_stream = torch.cuda.current_stream(self.device)
        transfer_done = torch.cuda.Event()
        transfer_done.record(default_stream)

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(transfer_done)
            first_frame = videos_processed[:, :, 0:1]
            vae_input = torch.cat([first_frame, videos_processed], dim=2)
            with torch.no_grad():
                latent = self.vae.encode(
                    vae_input,
                    tiled=self.tiled,
                    tile_size=self.tile_size,
                    tile_stride=self.tile_stride,
                )
            self._result = latent

    def get_vae_result(self) -> torch.Tensor:
        """Wait for VAE to finish."""
        if not self._submitted:
            raise RuntimeError("No VAE submitted")
        event = torch.cuda.Event()
        event.record(self._stream)
        torch.cuda.current_stream(self.device).wait_event(event)
        self._submitted = False
        return self._result
