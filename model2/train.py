import argparse
import numpy as np
import os
from pathlib import Path
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
    elif win_type == "rectangular":
        window = torch.ones(DatasetConfig.n_fft, device=complex_spec.device)
    else:
        raise ValueError(f"Unknown window type: {DatasetConfig.window}")

    wav = torch.istft(
        complex_spec,
        n_fft=DatasetConfig.n_fft,
        hop_length=DatasetConfig.hop_length,
        win_length=DatasetConfig.win_length,
        window=window,
        length=None,
    )
    return wav


def si_sdr_loss(pred, target, eps=1e-12, reduction="mean"):
    # pred, target: (B, T)

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    # projection
    dot = torch.sum(pred * target, dim=1, keepdim=True)
    target_energy = torch.sum(target**2, dim=1, keepdim=True) + eps

    scale = dot / target_energy
    proj = scale * target

    noise = pred - proj

    ratio = torch.sum(proj**2, dim=1) / (torch.sum(noise**2, dim=1) + eps)

    si_sdr = 10 * torch.log10(ratio + eps)

    loss = -si_sdr

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be 'mean' or 'none'")


# ===== train a epoch =====
def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    lambda1=1.0,
    lambda2=0.005,
    lambda3=5.0,
    eps=1e-12,
):
    model.train()
    total_loss = 0

    for noisy, clean, mask, _ in tqdm(train_loader, desc="Train"):
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        # forward
        pred_mask = model(noisy)
        pred_mask = pred_mask.squeeze(1)

        # complex
        noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
        clean_complex = torch.complex(clean[:, 0], clean[:, 1])
        enhanced_complex = pred_mask * noisy_complex

        # waveform
        clean_wave = istft_reconstruct(clean_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)

        clean_mag = torch.abs(clean_complex)
        enhanced_mag = torch.abs(enhanced_complex)

        clean_log = torch.log(clean_mag + eps)
        enhanced_log = torch.log(enhanced_mag + eps)

        sisdr_loss = si_sdr_loss(enhanced_wave, clean_wave, eps)

        spec_loss = torch.abs(enhanced_log - clean_log)
        spec_loss = spec_loss * mask
        spec_loss = spec_loss.sum() / mask.sum()

        num = torch.real(enhanced_complex * torch.conj(clean_complex))
        den = torch.abs(enhanced_complex) * torch.abs(clean_complex) + eps
        phase_loss = 1 - (num / den)
        phase_loss = phase_loss.mean()
        mag_gt = torch.abs(clean_complex)
        phase_mask = (mag_gt > mag_gt.mean()).float()
        phase_loss = phase_loss * phase_mask
        phase_loss = phase_loss.sum() / (phase_mask.sum() + eps)

        loss = lambda1 * sisdr_loss + lambda2 * spec_loss + lambda3 * phase_loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ===== evaluate =====
@torch.no_grad()
def evaluate(
    model, test_loader, device, lambda1=1.0, lambda2=0.005, lambda3=5.0, eps=1e-12
):
    model.eval()
    total_loss = 0

    for noisy, clean, mask, _ in tqdm(test_loader, desc="Eval"):
        noisy = noisy.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        # forward
        pred_mask = model(noisy)
        pred_mask = pred_mask.squeeze(1)

        # complex
        noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1]).to(device)
        clean_complex = torch.complex(clean[:, 0], clean[:, 1]).to(device)
        enhanced_complex = pred_mask * noisy_complex

        # waveform
        clean_wave = istft_reconstruct(clean_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)

        clean_mag = torch.abs(clean_complex)
        enhanced_mag = torch.abs(enhanced_complex)

        clean_log = torch.log(clean_mag + eps)
        enhanced_log = torch.log(enhanced_mag + eps)

        sisdr_loss = si_sdr_loss(enhanced_wave, clean_wave, eps)

        spec_loss = torch.abs(enhanced_log - clean_log)
        spec_loss = spec_loss * mask
        spec_loss = spec_loss.sum() / mask.sum()

        num = torch.real(enhanced_complex * torch.conj(clean_complex))
        den = torch.abs(enhanced_complex) * torch.abs(clean_complex) + eps
        phase_loss = 1 - (num / den)
        phase_loss = phase_loss.mean()
        mag_gt = torch.abs(clean_complex)
        phase_mask = (mag_gt > mag_gt.mean()).float()
        phase_loss = phase_loss * phase_mask
        phase_loss = phase_loss.sum() / (phase_mask.sum() + eps)

        loss = lambda1 * sisdr_loss + lambda2 * spec_loss + lambda3 * phase_loss

        total_loss += loss.item()

    return total_loss / len(test_loader)


# ===== train =====
def train(
    args, model, train_loader, test_loader, optimizer, device, save_path, eps=1e-12
):
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda1,
            args.lambda2,
            args.lambda3,
            eps,
        )
        val_loss = evaluate(
            model, test_loader, device, args.lambda1, args.lambda2, args.lambda3, eps
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # ===== save best model =====
        if val_loss < best_loss:
            best_loss = val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_path = CKPT_DIR / f"best_model_{args.exp_name}.pth"

    train(args, model, train_loader, test_loader, optimizer, device, save_path, EPS)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.0005)
    parser.add_argument("--lambda3", type=float, default=5.0)

    parser.add_argument("--exp_name", type=str, default="complex")

    return parser.parse_args()


if __name__ == "__main__":
    EPS = 1e-12

    BASE_DIR = Path(__file__).resolve().parent.parent
    CKPT_DIR = BASE_DIR / "ckpt"

    set_seed(42)

    args = parse_args()

    main()
