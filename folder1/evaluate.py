import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
from scipy import signal
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

    wave = torch.istft(
        complex_spec,
        n_fft=DatasetConfig.n_fft,
        hop_length=DatasetConfig.hop_length,
        win_length=DatasetConfig.win_length,
        window=window,
        length=None
    )
    return wave


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


def compute_snr(clean: np.ndarray, estimate: np.ndarray, eps: float=1e-12):
    noise = clean - estimate
    return 10 * np.log10(
        np.sum(clean ** 2) / (np.sum(noise ** 2) + eps)
    )


def compute_mae(clean: np.ndarray, estimate: np.ndarray):
    return np.mean(np.abs(clean - estimate))


def compute_mse(clean: np.ndarray, estimate: np.ndarray):
    return np.mean((clean - estimate) ** 2)


def compute_rmse(clean: np.ndarray, estimate: np.ndarray):
    return np.sqrt(compute_mse(clean, estimate))


def _resample_for_perceptual_metric(clean: np.ndarray, estimate: np.ndarray, sample_rate: int, target_rate: int = 16000):
    clean, estimate = align_signals(clean, estimate)

    if sample_rate != target_rate:
        clean = librosa.resample(clean, orig_sr=sample_rate, target_sr=target_rate)
        estimate = librosa.resample(estimate, orig_sr=sample_rate, target_sr=target_rate)

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


def plot_spectrogram(clean, noisy, enhanced, sample_rate, output_path, title, n_fft = 1024, hop_length = 256):
    signals = [("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, sharey=True)
    fig.suptitle(title, y=0.98)

    for index, (axis, (label, samples)) in enumerate(zip(axes, signals)):
        spectrum = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
        db = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)
        image = librosa.display.specshow(db, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="hz", ax=axis)
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
def evaluate_and_play(model, test_loader, device, num_examples=4,
                      lambda1=1.0, lambda2=0.005, lambda3=0.0, eps=1e-12):
    model.eval()

    examples = 0

    for noisy, clean, phase, mask, lengths in test_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        phase = phase.to(device)
        mask = mask.to(device)

        # forward
        pred_mask = model(noisy)
        enhanced = pred_mask * noisy

        noisy = noisy.squeeze(1)
        clean = clean.squeeze(1)
        enhanced = enhanced.squeeze(1)
        phase = phase.squeeze(1)

        # complex
        noisy_complex = torch.complex(
            noisy * torch.cos(phase),
            noisy * torch.sin(phase)
        )
        clean_complex = torch.complex(
            clean * torch.cos(phase),
            clean * torch.sin(phase)
        )
        enhanced_complex = torch.complex(
            enhanced * torch.cos(phase),
            enhanced * torch.sin(phase)
        )

        # waveform
        noisy_wave = istft_reconstruct(noisy_complex)
        clean_wave = istft_reconstruct(clean_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)

        clean_mag = torch.abs(clean_complex)
        enhanced_mag = torch.abs(enhanced_complex)

        clean_log = torch.log(clean_mag + eps)
        enhanced_log = torch.log(enhanced_mag + eps)

        sisdr_loss_per_sample  = si_sdr_loss(enhanced_wave, clean_wave, eps, reduction="none")
        sisdr_loss = sisdr_loss_per_sample.mean()

        spec_loss_map = torch.abs(enhanced_log - clean_log)
        spec_loss_map = spec_loss_map * mask
        spec_loss_per_sample = (
            spec_loss_map.sum(dim=(1,2,3)) /
            (mask.sum(dim=(1,2,3)) + eps)
        )
        spec_loss = spec_loss_map.sum() / mask.sum()

        complex_loss_map = torch.abs(enhanced_complex - clean_complex)
        complex_loss_per_sample = torch.mean(
            complex_loss_map,
            dim=(1, 2)
            )
        complex_loss = torch.mean(complex_loss_map)

        loss = lambda1 * sisdr_loss + lambda2 * spec_loss + lambda3 * complex_loss

        snr_noisy_batch = compute_snr_batch(clean_wave, noisy_wave, eps)
        snr_enh_batch = compute_snr_batch(clean_wave, enhanced_wave, eps)
        snr_improve_batch = snr_enh_batch - snr_noisy_batch

        print(f"\nbatch loss: {loss.item():.4f}")
        print(f"Avg ΔSNR:   {snr_improve_batch.mean():.4f}")

        B = noisy_wave.shape[0]

        for i in range(B):
            L = min(
                int(lengths[i]),
                clean_wave.shape[1],
                enhanced_wave.shape[1],
                noisy_wave.shape[1]
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

    results = []

    for noisy, clean, phase, _, lengths in tqdm(test_loader, desc="Full Eval"):
        noisy = noisy.to(device)
        clean = clean.to(device)
        phase = phase.to(device)

        # forward
        pred_mask = model(noisy)
        enhanced = pred_mask * noisy

        noisy = noisy.squeeze(1)
        clean = clean.squeeze(1)
        enhanced = enhanced.squeeze(1)
        phase = phase.squeeze(1)

        # complex
        noisy_complex = torch.complex(
            noisy * torch.cos(phase),
            noisy * torch.sin(phase)
        )
        clean_complex = torch.complex(
            clean * torch.cos(phase),
            clean * torch.sin(phase)
        )
        enhanced_complex = torch.complex(
            enhanced * torch.cos(phase),
            enhanced * torch.sin(phase)
        )

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
                noisy_wave.shape[1]
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
    print(f"ΔSNR:  {summary['snr_improve']:.6f}")
    print(f"MAE:   {summary['mae']:.6f}")
    print(f"MSE:   {summary['mse']:.6f}")
    print(f"RMSE:  {summary['rmse']:.6f}")
    print(f"PESQ:  {summary['pesq']:.6f}")
    print(f"STOI:  {summary['stoi']:.6f}")

    return df, summary


@torch.no_grad()
def enhance_and_save(
    model,
    device,
    test_files,
    infer_loader,
    suffix
):
    model.eval()

    save_dir = os.path.join("results", suffix)
    os.makedirs(save_dir, exist_ok=True)

    global_idx = 0

    for noisy, _, phase, _, lengths in infer_loader:

        noisy = noisy.to(device)
        phase = phase.to(device)

        # forward
        pred_mask = model(noisy)
        enhanced = pred_mask * noisy

        noisy = noisy.squeeze(1)
        enhanced = enhanced.squeeze(1)
        phase = phase.squeeze(1)

        # complex
        enhanced_complex = torch.complex(
            enhanced * torch.cos(phase),
            enhanced * torch.sin(phase)
        )

        # wave
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

            out_name = f"{orig_name}_{suffix}.wav"
            out_path = os.path.join(save_dir, out_name)

            sf.write(out_path, enhanced_wave_i, DatasetConfig.sample_rate)

            global_idx += 1

    print(f"Saved enhanced files to: {save_dir}")


@torch.no_grad()
def enhance_and_plot(
    model,
    device,
    test_files,
    infer_loader,
    suffix
):
    model.eval()

    fig_wave_dir = os.path.join("figures", "waveform_comparison", suffix)
    fig_spec_dir = os.path.join("figures", "spectrogram_comparison", suffix)

    os.makedirs(fig_wave_dir, exist_ok=True)
    os.makedirs(fig_spec_dir, exist_ok=True)

    global_idx = 0

    for noisy, clean, phase, _, lengths in infer_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        phase = phase.to(device)

        # forward
        pred_mask = model(noisy)
        enhanced = pred_mask * noisy

        noisy = noisy.squeeze(1)
        clean = clean.squeeze(1)
        enhanced = enhanced.squeeze(1)
        phase = phase.squeeze(1)

        # complex
        noisy_complex = torch.complex(
            noisy * torch.cos(phase),
            noisy * torch.sin(phase)
        )
        clean_complex = torch.complex(
            clean * torch.cos(phase),
            clean * torch.sin(phase)
        )
        enhanced_complex = torch.complex(
            enhanced * torch.cos(phase),
            enhanced * torch.sin(phase)
        )

        # wave
        noisy_wave = istft_reconstruct(noisy_complex)
        clean_wave = istft_reconstruct(clean_complex)
        enhanced_wave = istft_reconstruct(enhanced_complex)

        B = enhanced_wave.shape[0]

        for i in range(B):
            L = min(
                int(lengths[i]),
                noisy_wave.shape[1],
                clean_wave.shape[1],
                enhanced_wave.shape[1]
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
                os.path.join(fig_wave_dir, f"{orig_name}.png"),
                title=f"{suffix}: {orig_name}.wav"
            )

            # plot spectrogram
            plot_spectrogram(
                clean_wave_i,
                noisy_wave_i,
                enhanced_wave_i,
                DatasetConfig.sample_rate,
                os.path.join(fig_spec_dir, f"{orig_name}.png"),
                title=f"{suffix}: {orig_name}.wav"
            )

            global_idx += 1

    print(f"Saved figures to: figures/")


# main
def main(configs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subset = load_subset(
        DatasetConfig.test_txt,
        noise_type=None,
        snr=12.5
    )

    # _, test_loader = get_dataloaders(
    #     # test_files=subset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS
    # )

    # if configs is None:
    #     model = UNet().to(device)
    #     model.load_state_dict(torch.load("ckpt/best_model_baseline.pth", map_location=device))

    #     print("Loaded best model")

    #     # evaluate_and_play(model, test_loader, device, eps=EPS)

    #     df, summary = evaluate_full(model, test_loader, device, DatasetConfig.sample_rate, EPS)

    #     save_to_csv(df, "results/eval_full_baseline.csv")
    # else:
    #     all_results = {}

    #     for config in configs:
    #         print(f"\n====== Evaluating: {config['name']} ======")

    #         model = UNet(
    #             use_ca=config["use_ca"],
    #             use_sa=config["use_sa"]
    #         ).to(device)

    #         ckpt_path = f"ckpt/best_model_{config['name']}.pth"

    #         model.load_state_dict(torch.load(ckpt_path, map_location=device))
    #         print(f"Loaded {ckpt_path}")

    #         df, summary = evaluate_full(
    #             model,
    #             test_loader,
    #             device,
    #             DatasetConfig.sample_rate,
    #             EPS
    #         )

    #         save_path = f"results/eval_full_{config['name']}.csv"
    #         save_to_csv(df, save_path)

    #         all_results[config["name"]] = summary

    #         del model
    #         released = gc.collect()
    #         torch.cuda.empty_cache()

    #         print(f"\nCollected {released} objects.")
        
    #     summary_df = pd.DataFrame(all_results).T
    #     save_to_csv(summary_df, "results/summary_compare.csv")

    infer_files = ['p232_006', 'p232_290', 'p257_098']

    # infer_loader
    _, infer_loader = get_dataloaders(
        test_files=infer_files,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # print("\n===== GENERATING ENHANCED AUDIO =====")

    # if configs is None:
    #     model = UNet().to(device)
    #     model.load_state_dict(torch.load("ckpt/best_model_baseline.pth", map_location=device))

    #     enhance_and_save(
    #         model,
    #         device,
    #         infer_files,
    #         infer_loader,
    #         suffix="baseline"
    #     )

    # else:
    #     for config in configs:
    #         print(f"\n------ Enhancing: {config['name']} ------")

    #         model = UNet(
    #             use_ca=config["use_ca"],
    #             use_sa=config["use_sa"]
    #         ).to(device)

    #         ckpt_path = f"ckpt/best_model_{config['name']}.pth"
    #         model.load_state_dict(torch.load(ckpt_path, map_location=device))

    #         enhance_and_save(
    #             model,
    #             device,
    #             infer_files,
    #             infer_loader,
    #             suffix=config["name"]
    #         )

    #         del model
    #         released = gc.collect()
    #         torch.cuda.empty_cache()

    #         print(f"Collected {released} objects.")

    print("\n===== GENERATING COMPARE PLOT =====")

    if configs is None:
        model = UNet().to(device)
        model.load_state_dict(torch.load("ckpt/best_model_baseline.pth", map_location=device))

        enhance_and_plot(
            model,
            device,
            infer_files,
            infer_loader,
            suffix="baseline"
        )

    else:
        for config in configs:
            print(f"\n------ Enhancing: {config['name']} ------")

            model = UNet(
                use_ca=config["use_ca"],
                use_sa=config["use_sa"]
            ).to(device)

            ckpt_path = f"ckpt/best_model_{config['name']}.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            enhance_and_plot(
                model,
                device,
                infer_files,
                infer_loader,
                suffix=config["name"]
            )

            del model
            released = gc.collect()
            torch.cuda.empty_cache()

            print(f"Collected {released} objects.")


if __name__ == "__main__":
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    EPS = 1e-12

    configs = [
        {"name": "baseline", "use_ca": False, "use_sa": False},
        {"name": "ca", "use_ca": True, "use_sa": False},
        {"name": "sa", "use_ca": False, "use_sa": True},
        {"name": "ca_sa", "use_ca": True, "use_sa": True},
    ]

    main(configs)
