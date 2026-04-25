import argparse
import gc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sounddevice as sd
import soundfile as sf
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
from dataset import DatasetConfig, get_testloader, load_subset


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

    wave = torch.istft(
        complex_spec,
        n_fft=DatasetConfig.n_fft,
        hop_length=DatasetConfig.hop_length,
        win_length=DatasetConfig.win_length,
        window=window,
        length=None,
    )
    return wave


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


def compute_snr_batch(clean, test, eps=1e-12):
    # clean, test: [B, T]

    clean = clean - clean.mean(dim=1, keepdim=True)
    test = test - test.mean(dim=1, keepdim=True)

    noise = test - clean

    signal_power = torch.sum(clean**2, dim=1)
    noise_power = torch.sum(noise**2, dim=1)

    snr = 10 * torch.log10(signal_power / (noise_power + eps))
    return snr


def align_signals(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    min_length = min(len(signal) for signal in signals)
    return tuple(
        np.asarray(signal[:min_length], dtype=np.float32) for signal in signals
    )


def normalize(x: np.ndarray, eps: float = 1e-12):
    return x / (np.max(np.abs(x)) + eps)


def compute_snr(clean: np.ndarray, estimate: np.ndarray, eps: float = 1e-12):
    noise = clean - estimate
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + eps))


def compute_mae(clean: np.ndarray, estimate: np.ndarray):
    return np.mean(np.abs(clean - estimate))


def compute_mse(clean: np.ndarray, estimate: np.ndarray):
    return np.mean((clean - estimate) ** 2)


def compute_rmse(clean: np.ndarray, estimate: np.ndarray):
    return np.sqrt(compute_mse(clean, estimate))


def _resample_for_perceptual_metric(
    clean: np.ndarray, estimate: np.ndarray, sample_rate: int, target_rate: int = 16000
):
    clean, estimate = align_signals(clean, estimate)

    if sample_rate != target_rate:
        clean = librosa.resample(clean, orig_sr=sample_rate, target_sr=target_rate)
        estimate = librosa.resample(
            estimate, orig_sr=sample_rate, target_sr=target_rate
        )

    return clean.astype(np.float32), estimate.astype(np.float32), target_rate


def compute_pesq(clean: np.ndarray, estimate: np.ndarray, sample_rate: int):
    try:
        clean_16k, est_16k, sr = _resample_for_perceptual_metric(
            clean, estimate, sample_rate
        )
        return float(pesq.pesq(sr, clean_16k, est_16k, "wb"))
    except Exception:
        return None


def compute_stoi(clean: np.ndarray, estimate: np.ndarray, sample_rate: int):
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


def plot_waveform(clean, noisy, enhanced, sample_rate, output_path, title):
    signals = [("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, y=0.98)

    for index, (axis, (label, samples)) in enumerate(zip(axes, signals)):
        times = np.arange(len(samples)) / sample_rate
        axis.plot(times, samples, linewidth=0.7)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if index < len(signals) - 1:
            axis.tick_params(labelbottom=False)
            axis.set_xlabel("")

    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.24)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_spectrogram(
    clean, noisy, enhanced, sample_rate, output_path, title, n_fft=1024, hop_length=256
):
    signals = [("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, sharey=True)
    fig.suptitle(title, y=0.98)

    for index, (axis, (label, samples)) in enumerate(zip(axes, signals)):
        spectrum = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
        db = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)
        image = librosa.display.specshow(
            db,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="hz",
            ax=axis,
        )
        axis.set_title(label)
        if index < len(signals) - 1:
            axis.tick_params(labelbottom=False)
            axis.set_xlabel("")
        else:
            axis.set_xlabel("Time (s)")

    fig.subplots_adjust(top=0.89, bottom=0.08, right=0.88, hspace=0.20)
    fig.colorbar(image, ax=axes, format="%+2.0f dB", fraction=0.035, pad=0.06)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# evaluation + playback
@torch.no_grad()
def evaluate_and_play(
    model,
    test_loader,
    device,
    num_examples=4,
    lambda1=1.0,
    lambda2=0.005,
    lambda3=5,
    eps=1e-12,
):
    model.eval()

    examples = 0

    for noisy, clean, mask, lengths in test_loader:
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
        noisy_wave = istft_reconstruct(noisy_complex)
        clean_wave = istft_reconstruct(clean_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)

        clean_mag = torch.abs(clean_complex)
        enhanced_mag = torch.abs(enhanced_complex)

        clean_log = torch.log(clean_mag + eps)
        enhanced_log = torch.log(enhanced_mag + eps)

        sisdr_loss_per_sample = si_sdr_loss(
            enhanced_wave, clean_wave, eps, reduction="none"
        )
        sisdr_loss = sisdr_loss_per_sample.mean()

        spec_loss_map = torch.abs(enhanced_log - clean_log)
        spec_loss_map = spec_loss_map * mask
        spec_loss_per_sample = spec_loss_map.sum(dim=(1, 2, 3)) / (
            mask.sum(dim=(1, 2, 3)) + eps
        )
        spec_loss = spec_loss_map.sum() / mask.sum()

        num = torch.real(enhanced_complex * torch.conj(clean_complex))
        den = torch.abs(enhanced_complex) * torch.abs(clean_complex) + eps
        phase_loss_map = 1 - (num / den)
        mag_gt = torch.abs(clean_complex)
        phase_mask = (mag_gt > mag_gt.mean(dim=(1, 2), keepdim=True)).float()
        phase_loss_per_sample = (phase_loss_map * phase_mask).sum(dim=(1, 2)) / (
            phase_mask.sum(dim=(1, 2)) + 1e-8
        )
        phase_loss = phase_loss_per_sample.mean()

        loss = lambda1 * sisdr_loss + lambda2 * spec_loss + lambda3 * phase_loss

        snr_noisy_batch = compute_snr_batch(clean_wave, noisy_wave, eps)
        snr_enh_batch = compute_snr_batch(clean_wave, enhanced_wave, eps)
        snr_improve_batch = snr_enh_batch - snr_noisy_batch

        print(f"\nBatch loss: {loss.item():.4f}")
        print(f"Avg SNRI:   {snr_improve_batch.mean():.4f}")

        B = noisy_wave.shape[0]

        for i in range(B):
            L = min(
                int(lengths[i]),
                clean_wave.shape[1],
                enhanced_wave.shape[1],
                noisy_wave.shape[1],
            )

            noisy_wave_i = noisy_wave[i][:L].cpu().numpy()
            clean_wave_i = clean_wave[i][:L].cpu().numpy()
            enhanced_wave_i = enhanced_wave[i][:L].cpu().numpy()

            noisy_wave_i = normalize(noisy_wave_i)
            clean_wave_i = normalize(clean_wave_i)
            enhanced_wave_i = normalize(enhanced_wave_i)

            print(f"\nsample {examples+1}:")

            print("  Playing Noisy...")
            sd.play(noisy_wave_i, samplerate=DatasetConfig.sample_rate)
            sd.wait()

            print("  Playing Enhanced...")
            sd.play(enhanced_wave_i, samplerate=DatasetConfig.sample_rate)
            sd.wait()

            print("  Playing Clean...")
            sd.play(clean_wave_i, samplerate=DatasetConfig.sample_rate)
            sd.wait()

            print()
            print("  Loss breakdown:")
            print(f"    SI-SDR loss: {sisdr_loss_per_sample[i].item():.4f}")
            print(f"    Spec loss:   {spec_loss_per_sample[i].item():.4f}")
            print(f"    Phase loss:  {phase_loss_per_sample[i].item():.4f}")
            print("  SNR:")
            print(f"    Noisy:       {snr_noisy_batch[i].item():.2f} dB")
            print(f"    Enhanced:    {snr_enh_batch[i].item():.2f} dB")
            print(f"    Improvement: {snr_improve_batch[i].item():.2f} dB")

            examples += 1
            if examples >= num_examples:
                return


@torch.no_grad()
def evaluate_full(model, test_loader, device, sample_rate, eps=1e-12):
    model.eval()

    results = []

    for noisy, clean, _, lengths in tqdm(test_loader, desc="Full Eval"):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # forward
        pred_mask = model(noisy)
        pred_mask = pred_mask.squeeze(1)

        # complex
        noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
        clean_complex = torch.complex(clean[:, 0], clean[:, 1])
        enhanced_complex = pred_mask * noisy_complex

        # waveform
        noisy_wave = istft_reconstruct(noisy_complex)
        clean_wave = istft_reconstruct(clean_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)

        B = noisy_wave.shape[0]

        for i in range(B):
            L = min(
                int(lengths[i]),
                clean_wave.shape[1],
                enhanced_wave.shape[1],
                noisy_wave.shape[1],
            )

            noisy_wave_i = noisy_wave[i][:L].cpu().numpy()
            clean_wave_i = clean_wave[i][:L].cpu().numpy()
            enhanced_wave_i = enhanced_wave[i][:L].cpu().numpy()

            # normalize
            noisy_wave_i = normalize(noisy_wave_i, eps)
            clean_wave_i = normalize(clean_wave_i, eps)
            enhanced_wave_i = normalize(enhanced_wave_i, eps)

            # metrics
            snr_noisy = compute_snr(clean_wave_i, noisy_wave_i, eps)
            snr_enh = compute_snr(clean_wave_i, enhanced_wave_i, eps)

            row = {
                "snr_noisy": snr_noisy,
                "snr_enh": snr_enh,
                "snr_improve": snr_enh - snr_noisy,
                "mae": compute_mae(clean_wave_i, enhanced_wave_i),
                "mse": compute_mse(clean_wave_i, enhanced_wave_i),
                "rmse": compute_rmse(clean_wave_i, enhanced_wave_i),
                "pesq": compute_pesq(clean_wave_i, enhanced_wave_i, sample_rate),
                "stoi": compute_stoi(clean_wave_i, enhanced_wave_i, sample_rate),
            }

            results.append(row)

    # dataframe
    df = pd.DataFrame(results)

    summary = df.mean(numeric_only=True).to_dict()

    print("\n===== FINAL RESULTS =====")
    print(f"SNR:   {summary['snr_enh']:.6f}")
    print(f"SNRI:  {summary['snr_improve']:.6f}")
    print(f"MAE:   {summary['mae']:.6f}")
    print(f"MSE:   {summary['mse']:.6f}")
    print(f"RMSE:  {summary['rmse']:.6f}")
    print(f"PESQ:  {summary['pesq']:.6f}")
    print(f"STOI:  {summary['stoi']:.6f}")

    return df, summary


@torch.no_grad()
def enhance_and_save(model, device, test_files, infer_loader, suffix):
    model.eval()

    save_dir = RES_DIR / suffix
    save_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0

    for noisy, _, _, lengths in infer_loader:
        noisy = noisy.to(device)

        # forward
        pred_mask = model(noisy)
        pred_mask = pred_mask.squeeze(1)

        # complex
        noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
        enhanced_complex = pred_mask * noisy_complex

        # waveform
        enhanced_wave = istft_reconstruct(enhanced_complex)

        B = enhanced_wave.shape[0]

        for i in range(B):
            L = min(int(lengths[i]), enhanced_wave.shape[1])
            enhanced_wave_i = enhanced_wave[i][:L].cpu().numpy()

            enhanced_wave_i = normalize(enhanced_wave_i)

            if global_idx < len(test_files):
                orig_name = Path(test_files[global_idx]).stem
            else:
                orig_name = f"sample_{global_idx}"

            sf.write(
                save_dir / f"{orig_name}_{suffix}.wav",
                enhanced_wave_i,
                DatasetConfig.sample_rate,
            )

            global_idx += 1

    print(f"Saved enhanced files to: {save_dir}")


@torch.no_grad()
def enhance_and_plot(model, device, test_files, infer_loader, suffix):
    model.eval()

    fig_wave_dir = FIG_DIR / "waveform_comparison" / suffix
    fig_spec_dir = FIG_DIR / "spectrogram_comparison" / suffix

    fig_wave_dir.mkdir(parents=True, exist_ok=True)
    fig_spec_dir.mkdir(parents=True, exist_ok=True)

    global_idx = 0

    for noisy, clean, _, lengths in infer_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # forward
        pred_mask = model(noisy)
        pred_mask = pred_mask.squeeze(1)

        # complex
        noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
        clean_complex = torch.complex(clean[:, 0], clean[:, 1])
        enhanced_complex = pred_mask * noisy_complex

        # waveform
        noisy_wave = istft_reconstruct(noisy_complex)
        clean_wave = istft_reconstruct(clean_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)

        B = enhanced_wave.shape[0]

        for i in range(B):
            L = min(
                int(lengths[i]),
                noisy_wave.shape[1],
                clean_wave.shape[1],
                enhanced_wave.shape[1],
            )

            noisy_wave_i = normalize(noisy_wave[i][:L].cpu().numpy())
            clean_wave_i = normalize(clean_wave[i][:L].cpu().numpy())
            enhanced_wave_i = normalize(enhanced_wave[i][:L].cpu().numpy())

            if global_idx < len(test_files):
                orig_name = Path(test_files[global_idx]).stem
            else:
                orig_name = f"sample_{global_idx}"

            # plot waveform
            plot_waveform(
                clean_wave_i,
                noisy_wave_i,
                enhanced_wave_i,
                DatasetConfig.sample_rate,
                fig_wave_dir / f"{orig_name}.png",
                title=f"{suffix}: {orig_name}.wav",
            )

            # plot spectrogram
            plot_spectrogram(
                clean_wave_i,
                noisy_wave_i,
                enhanced_wave_i,
                DatasetConfig.sample_rate,
                fig_spec_dir / f"{orig_name}.png",
                title=f"{suffix}: {orig_name}.wav",
            )

            global_idx += 1

    print(f"Saved figures to: {FIG_DIR}")


def run_model(args, model, test_loader, infer_loader, infer_files, device, eps=1e-12):
    # playback / quick eval
    if args.mode in ["play", "all"]:
        evaluate_and_play(
            model, test_loader, device, num_examples=args.num_examples, eps=eps
        )

    # full evaluation
    if args.mode in ["eval", "all"]:
        df, summary = evaluate_full(
            model, test_loader, device, DatasetConfig.sample_rate, eps=eps
        )
        save_to_csv(df, RES_DIR / f"eval_full_{args.exp_name}.csv")

    # save audio
    if args.mode in ["save", "all"]:
        enhance_and_save(model, device, infer_files, infer_loader, suffix=args.exp_name)

    # plot
    if args.mode in ["plot", "all"]:
        enhance_and_plot(model, device, infer_files, infer_loader, suffix=args.exp_name)

    return summary if args.mode in ["full", "all"] else None


# main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== argument sanity check =====
    if args.multi_run and args.exp_name != "complex":
        print(
            "[Warning] --multi_run enabled: using configs, ignoring exp_name / single-model loading"
        )

    valid_num_examples_modes = ["play", "all"]

    if args.mode not in valid_num_examples_modes and args.num_examples != 4:
        print(
            f"[Warning] --num_examples is ignored because mode='{args.mode}' "
            f"(only {valid_num_examples_modes} uses num_examples)"
        )

    valid_subset_modes = ["play", "eval", "all"]

    if args.use_subset and args.mode not in valid_subset_modes:
        print(
            f"[Warning] --use_subset is ignored because mode='{args.mode}' "
            f"(only {valid_subset_modes} support subset)"
        )
        use_subset = False
    else:
        use_subset = args.use_subset

    if not use_subset and args.snr != 12.5:
        print("[Warning] --snr has no effect because subset is not enabled")

    # ===== test set =====
    if use_subset:
        print(f"Using subset with SNR={args.snr}")
        test_files = load_subset(DatasetConfig.test_txt, noise_type=None, snr=args.snr)
    else:
        print("Using full test set")
        test_files = None

    test_loader = get_testloader(
        test_files=test_files, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ===== infer set =====
    infer_files = ["p232_006", "p232_290", "p257_098"]

    infer_loader = get_testloader(
        test_files=infer_files, batch_size=args.batch_size, num_workers=args.num_workers
    )

    print(f"\n====== Running: {args.exp_name} ======")

    # ===== model =====
    model = UNet().to(device)

    ckpt_path = CKPT_DIR / f"best_model_{args.exp_name}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("Loaded best model")

    # ===== run pipeline =====
    run_model(args, model, test_loader, infer_loader, infer_files, device, EPS)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["play", "eval", "save", "plot", "all"],
    )

    parser.add_argument("--exp_name", type=str, default="complex")

    parser.add_argument(
        "--use_subset", action="store_true", help="Use subset instead of full test set"
    )

    parser.add_argument(
        "--snr", type=float, default=12.5, help="SNR for subset generation"
    )

    return parser.parse_args()


if __name__ == "__main__":
    EPS = 1e-12

    BASE_DIR = Path(__file__).resolve().parent.parent
    CKPT_DIR = BASE_DIR / "ckpt"
    RES_DIR = BASE_DIR / "results"
    FIG_DIR = BASE_DIR / "figures"

    args = parse_args()

    main(args)
