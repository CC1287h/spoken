import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import DatasetConfig, get_dataloaders
from model import UNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def istft_reconstruct(complex_spec):
    win_type = DatasetConfig.window.lower()
    if win_type == "hann":
        window = torch.hann_window(DatasetConfig.n_fft, device=complex_spec.device)
    elif win_type == "hamming":
        window = torch.hamming_window(DatasetConfig.n_fft, device=complex_spec.device)
    elif win_type  == "rectangular":
        window = torch.ones(DatasetConfig.n_fft, device=complex_spec.device)
    else:
        raise ValueError(f"Unknown window type: {DatasetConfig.window}")

    wav = torch.istft(
        complex_spec,
        n_fft=DatasetConfig.n_fft,
        hop_length=DatasetConfig.hop_length,
        win_length=DatasetConfig.win_length,
        window=window,
        length=None
    )
    return wav


def si_sdr_loss(pred, target, eps=1e-12, reduction="mean"):
    # pred, target: (B, T)

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    # projection
    dot = torch.sum(pred * target, dim=1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + eps

    scale = dot / target_energy
    proj = scale * target

    noise = pred - proj

    ratio = torch.sum(proj ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps)

    si_sdr = 10 * torch.log10(ratio + eps)

    loss = -si_sdr

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be 'mean' or 'none'")


# ===== 训练函数 =====
def train_epoch(model, train_loader, optimizer, device, lambda1=1.0, lambda2=0.1, lambda3=1.0):
    model.train()
    total_loss = 0

    for noisy, clean, phase, mask, _ in tqdm(train_loader, desc="Train"):
        noisy = noisy.to(device)
        clean = clean.to(device)
        phase = phase.to(device)
        mask = mask.to(device)

        # 前向
        pred_mask = model(noisy)
        enhanced = pred_mask * noisy

        enhanced = enhanced.squeeze(1)
        clean = clean.squeeze(1)
        phase = phase.squeeze(1)

        enhanced_complex = torch.complex(
            enhanced * torch.cos(phase),
            enhanced * torch.sin(phase)
        )
        clean_complex = torch.complex(
            clean * torch.cos(phase),
            clean * torch.sin(phase)
        )

        enhanced_wave = istft_reconstruct(enhanced_complex)
        clean_wave = istft_reconstruct(clean_complex)

        enhanced_mag = torch.abs(enhanced_complex)
        clean_mag = torch.abs(clean_complex)

        enhanced_log = torch.log(enhanced_mag + EPS)
        clean_log = torch.log(clean_mag + EPS)

        sisdr_loss  = si_sdr_loss(enhanced_wave, clean_wave, EPS)

        spec_loss = torch.abs(enhanced_log - clean_log)
        spec_loss = spec_loss * mask
        spec_loss = spec_loss.sum() / mask.sum()

        complex_loss = torch.mean(torch.abs(enhanced_complex - clean_complex))

        loss = lambda1 * sisdr_loss + lambda2 * spec_loss + lambda3 * complex_loss 

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ===== 验证函数 =====
@torch.no_grad()
def evaluate(model, test_loader, device, lambda1=1.0, lambda2=0.1, lambda3=1.0):
    model.eval()
    total_loss = 0

    for noisy, clean, phase, mask, _ in tqdm(test_loader, desc="Eval"):
        noisy = noisy.to(device)
        clean = clean.to(device)
        phase = phase.to(device)
        mask = mask.to(device)

        pred_mask = model(noisy)
        enhanced = pred_mask * noisy

        enhanced = enhanced.squeeze(1)
        clean = clean.squeeze(1)
        phase = phase.squeeze(1)

        enhanced_complex = torch.complex(
            enhanced.squeeze(1) * torch.cos(phase),
            enhanced.squeeze(1) * torch.sin(phase)
        )
        clean_complex = torch.complex(
            clean.squeeze(1) * torch.cos(phase),
            clean.squeeze(1) * torch.sin(phase)
        )

        enhanced_wave = istft_reconstruct(enhanced_complex)
        clean_wave = istft_reconstruct(clean_complex)

        enhanced_mag = torch.abs(enhanced_complex)
        clean_mag = torch.abs(clean_complex)

        enhanced_log = torch.log(enhanced_mag + EPS)
        clean_log = torch.log(clean_mag + EPS)

        sisdr_loss  = si_sdr_loss(enhanced_wave, clean_wave, EPS)

        spec_loss = torch.abs(enhanced_log - clean_log)
        spec_loss = spec_loss * mask
        spec_loss = spec_loss.sum() / mask.sum()

        complex_loss = torch.mean(torch.abs(enhanced_complex - clean_complex))

        loss = lambda1 * sisdr_loss + lambda2 * spec_loss + lambda3 * complex_loss 

        total_loss += loss.item()

    return total_loss / len(test_loader)


# ===== 训练主循环 =====
def train(model, train_loader, test_loader, optimizer, device):
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, test_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # ===== 保存模型 =====
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("ckpt", exist_ok=True)

            torch.save(model.state_dict(), "ckpt/best_model_mag.pth")
            print("Saved best model")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(model, train_loader, test_loader, optimizer, device)


if __name__ == "__main__":
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LR = 1e-3
    EPOCHS = 20
    EPS = 1e-12

    set_seed(42)

    main()
