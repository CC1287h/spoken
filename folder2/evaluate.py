import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datetime import datetime
import librosa
import pandas as pd
from pathlib import Path
import pesq
import pystoi

from model import UNet
from dataset import DatasetConfig, get_dataloaders, load_subset


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
    

def compute_snr_batch(clean, test, eps=1e-12):
    # clean, test: [B, T]

    clean = clean - clean.mean(dim=1, keepdim=True)
    test = test - test.mean(dim=1, keepdim=True)

    noise = test - clean

    signal_power = torch.sum(clean ** 2, dim=1)
    noise_power = torch.sum(noise ** 2, dim=1)

    snr = 10 * torch.log10(signal_power / (noise_power + eps))
    return snr


def align_signals(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    min_length = min(len(signal) for signal in signals)
    return tuple(np.asarray(signal[:min_length], dtype=np.float32) for signal in signals)


def normalize(x: np.ndarray, eps: float=1e-12):
    return x / (np.max(np.abs(x)) + eps)


def snr(clean: np.ndarray, estimate: np.ndarray, eps: float=1e-12):
    noise = clean - estimate
    return 10 * np.log10(
        np.sum(clean ** 2) / (np.sum(noise ** 2) + eps)
    )


def mae(clean: np.ndarray, estimate: np.ndarray):
    return np.mean(np.abs(clean - estimate))


def mse(clean: np.ndarray, estimate: np.ndarray):
    return np.mean((clean - estimate) ** 2)


def rmse(clean: np.ndarray, estimate: np.ndarray):
    return np.sqrt(mse(clean, estimate))


def _resample_for_perceptual_metric(clean: np.ndarray, estimate: np.ndarray, sample_rate: int, target_rate: int = 16000):
    clean, estimate = align_signals(clean, estimate)

    if sample_rate != target_rate:
        clean = librosa.resample(clean, orig_sr=sample_rate, target_sr=target_rate)
        estimate = librosa.resample(estimate, orig_sr=sample_rate, target_sr=target_rate)

    return clean.astype(np.float32), estimate.astype(np.float32), target_rate


def pesq_wb(clean: np.ndarray, estimate: np.ndarray, sample_rate: int):
    try:
        clean_16k, est_16k, sr = _resample_for_perceptual_metric(
            clean, estimate, sample_rate
        )
        return float(pesq.pesq(sr, clean_16k, est_16k, "wb"))
    except Exception:
        return None


def stoi(clean: np.ndarray, estimate: np.ndarray, sample_rate: int):
    try:
        clean_16k, est_16k, sr = _resample_for_perceptual_metric(
            clean, estimate, sample_rate
        )
        return float(pystoi.stoi(clean_16k, est_16k, sr, extended=False))
    except Exception:
        return None


def save_to_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{ts}.csv")
        df.to_csv(fallback, index=False)
        return fallback


# evaluation + playback
@torch.no_grad()
def evaluate_and_play(model, test_loader, device, num_examples=4, lambda1=1.0, lambda2=0.5, lambda3=0.1):
    model.eval()

    examples = 0

    for noisy, clean, mask, _ in test_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        pred_mask = model(noisy)
        pred_mask = pred_mask.squeeze(1)

        noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
        clean_complex = torch.complex(clean[:, 0], clean[:, 1])

        enhanced_complex = pred_mask * noisy_complex

        noisy_wave = istft_reconstruct(noisy_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)
        clean_wave = istft_reconstruct(clean_complex)

        enhanced_mag = torch.abs(enhanced_complex)
        clean_mag = torch.abs(clean_complex)

        enhanced_log = torch.log(enhanced_mag + EPS)
        clean_log = torch.log(clean_mag + EPS)

        sisdr_loss_per_sample  = si_sdr_loss(enhanced_wave, clean_wave, EPS, reduction="none")
        sisdr_loss = sisdr_loss_per_sample.mean()

        spec_loss_map = torch.abs(enhanced_log - clean_log)
        spec_loss_map = spec_loss_map * mask
        spec_loss_per_sample = (
            spec_loss_map.sum(dim=(1,2,3)) /
            (mask.sum(dim=(1,2,3)) + EPS)
        )
        spec_loss = spec_loss_map.sum() / mask.sum()

        complex_loss_map = torch.abs(enhanced_complex - clean_complex)
        complex_loss_per_sample = torch.mean(
            complex_loss_map,
            dim=(1, 2)
            )
        complex_loss = torch.mean(complex_loss_map)

        loss = lambda1 * sisdr_loss + lambda2 * spec_loss + lambda3 * complex_loss 

        print(f"\nbatch loss: {loss.item():.4f}")

        snr_noisy_batch = compute_snr_batch(clean_wave, noisy_wave, EPS)
        snr_enh_batch = compute_snr_batch(clean_wave, enhanced_wave, EPS)
        snr_improve_batch = snr_enh_batch - snr_noisy_batch

        print(f"Avg ΔSNR:   {snr_improve_batch.mean():.4f}")

        # 取 batch 中一个样本播放
        for i in range(noisy.shape[0]):

            noisy_wav = noisy_wave[i].cpu().numpy()
            clean_wav = clean_wave[i].cpu().numpy()
            enh_wav = enhanced_wave[i].cpu().numpy()

            noisy_wav = normalize(noisy_wav)
            clean_wav = normalize(clean_wav)
            enh_wav = normalize(enh_wav)

            print(f"\nsample {examples+1}:")

            print("  Playing Noisy...")
            sd.play(noisy_wav, samplerate=DatasetConfig.sample_rate)
            sd.wait()

            print("  Playing Enhanced...")
            sd.play(enh_wav, samplerate=DatasetConfig.sample_rate)
            sd.wait()

            print("  Playing Clean...")
            sd.play(clean_wav, samplerate=DatasetConfig.sample_rate)
            sd.wait()

            print(f"  SI-SDR loss: {sisdr_loss_per_sample[i].item():.4f}")
            print(f"  Spec loss:   {spec_loss_per_sample[i].item():.4f}")
            print(f"  Comp loss:   {complex_loss_per_sample[i].item():.4f}")
            print(f"  SNR noisy:   {snr_noisy_batch[i].item():.2f} dB")
            print(f"  SNR enhced:  {snr_enh_batch[i].item():.2f} dB")
            print(f"  ΔSNR:        {snr_improve_batch[i].item():.2f} dB")

            examples += 1
            if examples >= num_examples:
                return


@torch.no_grad()
def evaluate_full(model, test_loader, device, sample_rate, eps=1e-12):
    model.eval()

    all_snr = []
    all_snr_improve = []
    all_mae = []
    all_mse = []
    all_rmse = []
    all_pesq = []
    all_stoi = []

    for noisy, clean, _ in tqdm(test_loader, desc="Full Eval"):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # ===== forward =====
        mask = model(noisy)              # [B,1,F,T]
        mask = mask.squeeze(1)           # [B,F,T]

        noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
        clean_complex = torch.complex(clean[:, 0], clean[:, 1])

        enhanced_complex = mask * noisy_complex

        # ===== ISTFT =====
        noisy_wave = istft_reconstruct(noisy_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)
        clean_wave = istft_reconstruct(clean_complex)

        B = noisy_wave.shape[0]

        for i in range(B):
            clean_wav = clean_wave[i].cpu().numpy()
            noisy_wav = noisy_wave[i].cpu().numpy()
            enh_wav   = enhanced_wave[i].cpu().numpy()

            # ===== normalize（很关键！）=====
            def norm(x):
                return x / (np.max(np.abs(x)) + eps)

            clean_wav = norm(clean_wav)
            noisy_wav = norm(noisy_wav)
            enh_wav   = norm(enh_wav)

            # ===== SNR =====
            snr_noisy = 10 * np.log10(
                np.sum(clean_wav**2) / (np.sum((clean_wav - noisy_wav)**2) + eps)
            )
            snr_enh = 10 * np.log10(
                np.sum(clean_wav**2) / (np.sum((clean_wav - enh_wav)**2) + eps)
            )

            all_snr.append(snr_enh)
            all_snr_improve.append(snr_enh - snr_noisy)

            # ===== MAE / MSE / RMSE =====
            mae = np.mean(np.abs(clean_wav - enh_wav))
            mse = np.mean((clean_wav - enh_wav) ** 2)
            rmse = np.sqrt(mse)

            all_mae.append(mae)
            all_mse.append(mse)
            all_rmse.append(rmse)

            # ===== PESQ（注意采样率）=====
            # try:
            #     if sample_rate == 16000:
            #         pesq_score = pesq(16000, clean_wav, enh_wav, 'wb')
            #     elif sample_rate == 8000:
            #         pesq_score = pesq(8000, clean_wav, enh_wav, 'nb')
            #     else:
            #         pesq_score = np.nan
            #     all_pesq.append(pesq_score)
            # except:
            #     all_pesq.append(np.nan)

            # # ===== STOI =====
            # try:
            #     stoi_score = stoi(clean_wav, enh_wav, sample_rate, extended=False)
            #     all_stoi.append(stoi_score)
            # except:
            #     all_stoi.append(np.nan)

    # ===== 汇总 =====
    def summarize(x):
        x = np.array(x)
        return np.nanmean(x), np.nanstd(x)

    print("\n===== FINAL RESULTS =====")

    print(f"SNR:        {summarize(all_snr)[0]:.3f} ± {summarize(all_snr)[1]:.3f}")
    print(f"ΔSNR:       {summarize(all_snr_improve)[0]:.3f} ± {summarize(all_snr_improve)[1]:.3f}")

    print(f"MAE:        {summarize(all_mae)[0]:.6f}")
    print(f"MSE:        {summarize(all_mse)[0]:.6f}")
    print(f"RMSE:       {summarize(all_rmse)[0]:.6f}")

    # print(f"PESQ:       {summarize(all_pesq)[0]:.3f}")
    # print(f"STOI:       {summarize(all_stoi)[0]:.3f}")

    return {
        "snr": summarize(all_snr),
        "snr_improve": summarize(all_snr_improve),
        "mae": summarize(all_mae),
        "mse": summarize(all_mse),
        "rmse": summarize(all_rmse),
        # "pesq": summarize(all_pesq),
        # "stoi": summarize(all_stoi),
    }


# main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subset = load_subset(
        DatasetConfig.test_txt,
        noise_type=None,
        snr=7.5
    )

    _, test_loader = get_dataloaders(
        # test_files=subset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = UNet().to(device)
    model.load_state_dict(torch.load("ckpt/best_model_com.pth", map_location=device))

    print("Loaded best model")

    # evaluate_and_play(model, test_loader, device)
    evaluate_full(model, test_loader, device, EPS)


if __name__ == "__main__":
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    EPS = 1e-12

    main()
