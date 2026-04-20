import numpy as np
import os
import random
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import torchaudio


class DatasetConfig:
    # Audio format
    sample_rate = 16000

    # STFT
    n_fft = 400
    hop_length = 160
    win_length = 400
    window = "hann"   # hann / hamming / rectangular

    # Dataset paths
    train_clean_dir = r"D:\Final Project\Dataset\clean_trainset_28spk_wav\clean_trainset_28spk_wav"
    train_noisy_dir = r"D:\Final Project\Dataset\noisy_trainset_28spk_wav\noisy_trainset_28spk_wav"

    test_clean_dir = r"D:\Final Project\Dataset\clean_testset_wav\clean_testset_wav"
    test_noisy_dir = r"D:\Final Project\Dataset\noisy_testset_wav\noisy_testset_wav"

    train_txt = r"D:\Final Project\Dataset\logfiles\log_trainset_28spk.txt"
    test_txt = r"D:\Final Project\Dataset\logfiles\log_testset.txt"


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noisy_dir, file_list=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir

        if file_list is None:
            self.files = sorted(os.listdir(clean_dir))
        else:
            self.files = [self.resolve_wav(x) for x in file_list]

        self.lengths = []
        for f in self.files:
            c = sf.read(os.path.join(clean_dir, f))[0]
            n = sf.read(os.path.join(noisy_dir, f))[0]

            if c.ndim > 1:
                c = c.mean(axis=1)
            if n.ndim > 1:
                n = n.mean(axis=1)

            self.lengths.append(min(len(c), len(n)))

        self.resamplers = {}

        win_type = DatasetConfig.window.lower()
        if win_type == "hann":
            self.window = torch.hann_window(DatasetConfig.n_fft)
        elif win_type == "hamming":
            self.window = torch.hamming_window(DatasetConfig.n_fft)
        elif win_type  == "rectangular":
            self.window = torch.ones(DatasetConfig.n_fft)
        else:
            raise ValueError(f"Unknown window type: {DatasetConfig.window}")

        self.n_fft = DatasetConfig.n_fft
        self.hop_length = DatasetConfig.hop_length


    def resolve_wav(self, filename):
        return filename if filename.endswith(".wav") else filename + ".wav"


    def load_wav(self, path, eps=1e-12):
        wav, sr = sf.read(path)

        wav = torch.tensor(wav, dtype=torch.float32)

        # mono
        if wav.ndim > 1:
            wav = wav.mean(dim=1)

        # resample
        if sr != DatasetConfig.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=DatasetConfig.sample_rate
                )

            wav = self.resamplers[sr](wav.unsqueeze(0)).squeeze(0)

        # normalize
        wav = wav / (wav.abs().max() + eps)

        return wav


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        filename = self.resolve_wav(self.files[idx])

        noisy = self.load_wav(os.path.join(self.noisy_dir, filename))
        clean = self.load_wav(os.path.join(self.clean_dir, filename))

        T = min(clean.shape[0], noisy.shape[0])
        noisy = noisy[:T]
        clean = clean[:T]

        noisy = noisy.unsqueeze(0)
        clean = clean.unsqueeze(0)

        noisy_spec = torch.stft(noisy[0], n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                window=self.window,
                                return_complex=True)

        clean_spec = torch.stft(clean[0], n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                window=self.window,
                                return_complex=True)

        noisy_mag = torch.abs(noisy_spec)
        clean_mag = torch.abs(clean_spec)
        noisy_phase = torch.angle(noisy_spec)

        return {
            "noisy_mag": noisy_mag,
            "clean_mag": clean_mag,
            "noisy_phase": noisy_phase,
            "length": noisy_mag.shape[-1]
            }


def collate_fn(batch):
    lengths = [x["length"] for x in batch]
    max_len = max(lengths)

    def pad(x):
        return F.pad(x, (0, max_len - x.shape[-1]))

    noisy_mag = torch.stack([pad(x["noisy_mag"]) for x in batch])
    clean_mag = torch.stack([pad(x["clean_mag"]) for x in batch])
    noisy_phase = torch.stack([pad(x["noisy_phase"]) for x in batch])

    noisy_mag = noisy_mag.unsqueeze(1)
    clean_mag = clean_mag.unsqueeze(1)
    noisy_phase = noisy_phase.unsqueeze(1)

    mask = torch.zeros(len(batch), 1, 1, max_len)
    for i, l in enumerate(lengths):
        mask[i, :, :, :l] = 1.0

    return noisy_mag, clean_mag, noisy_phase, mask, lengths


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False, bucket_size=50, seed=42):
        """
        lengths: list[int] 每个样本的 T
        batch_size: batch大小
        bucket_size: 每个 bucket 内排序范围
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = bucket_size

        self.indices = list(range(len(lengths)))
        self.rng = random.Random(seed)

    def _make_buckets(self):
        sorted_indices = sorted(self.indices, key=lambda i: self.lengths[i])

        buckets = []
        for i in range(0, len(sorted_indices), self.bucket_size):
            buckets.append(sorted_indices[i:i + self.bucket_size])

        return buckets

    def __iter__(self):
        buckets = self._make_buckets()

        if self.shuffle:
            self.rng.shuffle(buckets)

        batch_list = []

        for bucket in buckets:
            if self.shuffle:
                self.rng.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]

                if len(batch) == self.batch_size or not self.drop_last:
                    batch_list.append(batch)

        if self.shuffle:
            self.rng.shuffle(batch_list)

        for batch in batch_list:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def worker_init_fn(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataloaders(
    train_clean_dir=DatasetConfig.train_clean_dir,
    train_noisy_dir=DatasetConfig.train_noisy_dir,
    test_clean_dir=DatasetConfig.test_clean_dir,
    test_noisy_dir=DatasetConfig.test_noisy_dir,
    train_files=None,
    test_files=None,
    batch_size=8,
    num_workers=4
):
    train_dataset = SpeechDataset(train_clean_dir, train_noisy_dir, train_files)
    test_dataset = SpeechDataset(test_clean_dir, test_noisy_dir, test_files)

    sampler = BucketBatchSampler(
        lengths=train_dataset.lengths,
        batch_size=batch_size,
        bucket_size=100,
        shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    return train_loader, test_loader


def load_subset(txt_path, noise_type=None, snr=None):
    items = []

    with open(txt_path, 'r') as f:
        for line in f:
            name, noise, snr_val = line.strip().split()

            if noise_type and noise != noise_type:
                continue

            if snr and float(snr_val) != snr:
                continue

            items.append(name)

    return items
