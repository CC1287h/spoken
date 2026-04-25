# Speech Enhancement Coursework

This repository contains a complete implementation for a speech enhancement course assignment. The overall approach is to first implement classical speech enhancement methods to establish a traditional baseline, then introduce neural network methods, and finally compare all methods using unified metrics and provide a summary conclusion.

## Assignment Content

This assignment mainly consists of two parts:

1. Classical speech enhancement methods
2. Neural network speech enhancement methods

The classical methods implemented include:

- Spectral Subtraction
- Wavelet Denoising
- Frequency Masking

For the neural network part, two main results are retained:

- Magnitude Mask U-Net
- Complex Mask U-Net

## Dataset

The experiments are based on the paired noisy speech database provided by Edinburgh DataShare. The raw data is not included in this repository and must be downloaded separately.

Dataset page:
https://datashare.ed.ac.uk/handle/10283/2791

The test set directory should be organized as:

```text
data/
  noisy_testset_wav/
  clean_testset_wav/
```

## Final Results Table

Results for 6 methods:

| Method | SNRI | MAE | RMSE | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| Noisy Input | 0.000 | 0.024 | 0.030 | 1.967 | 0.921 |
| Spectral Subtraction | 5.445 | 0.010 | 0.0156 | 2.2948 | 0.912 |
| Wavelet Denoising | 0.220 | 0.023 | 0.029 | 2.094 | 0.915 |
| Frequency Masking | 4.223 | 0.013 | 0.019 | 2.233 | 0.919 |
| Magnitude Mask U-Net | 11.062 | 0.009 | 0.013 | 2.850 | 0.947 |
| Complex Mask U-Net | 8.545 | 0.015 | 0.020 | 2.595 | 0.935 |

## Conclusion

- Among classical methods, `Spectral Subtraction` is the best-performing traditional baseline.
- `Wavelet Denoising` is the weakest, providing limited improvement.
- `Frequency Masking` also outperforms the noisy input, but overall is still weaker than the best neural network method.
- Neural network methods generally outperform classical methods.
- Among the retained neural network results, `Magnitude Mask U-Net` achieves the best performance and is the optimal solution for this coursework.
- Although `Complex Mask U-Net` surpasses classical methods, it does not outperform `Magnitude Mask U-Net`.

## Repository Notes

### Classical Method Code

- `src/spoken_denoise/`
- `scripts/run_parameter_search.py`
- `scripts/run_all_methods.py`
- `scripts/evaluate_results.py`
- `scripts/make_figures.py`

### Neural Network Code

- `model1/`: Magnitude Mask U-Net pipeline
- `model2/`: Complex Mask U-Net pipeline
- `ckpt/`: Trained weights

### Results and Documentation

- `results/final_summary.csv`: Summary results for classical methods
- `results/eval_full_baseline.csv`: Source of Magnitude Mask U-Net results
- `results/eval_full_com.csv`: Source of Complex Mask U-Net results
- `docs/experiment_report.md`: Full experiment report
- `docs/code_structure.md`: Code structure description

## Reproduction Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run classical methods:

```bash
python scripts/run_parameter_search.py
python scripts/run_all_methods.py
python scripts/evaluate_results.py
```

Generate demonstration figures:

```bash
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise --target-sr 16000
```

## Magnitude Model Training Usage Summary

### 1. Overview
This script trains a U-Net based speech enhancement model using **magnitude masking**.
The model operates on **STFT spectrograms** (not raw audio) and reconstructs waveform via ISTFT only for loss computation.

### 2. Usage Modes

#### (1) Single Model Training
Train one model with manually specified architecture:

```bash
python model1/train.py --exp_name baseline
```

Examples:
```bash
python model1/train.py --use_ca --exp_name ca
python model1/train.py --use_sa --exp_name sa
python model1/train.py --use_ca --use_sa --exp_name ca_sa
```

Behavior:
- Model structure is determined by `--use_ca` and `--use_sa`
- Model is saved as: `ckpt/best_model_{exp_name}.pth`

#### (2) Multi-Model Training (Ablation)
Train multiple predefined models automatically:

```bash
python model1/train.py --multi_run
```

This runs the following configurations:
- baseline (no attention)
- ca (channel attention)
- sa (skip attention)
- ca_sa (both)

Behavior:
- Ignores: `--use_ca`, `--use_sa`, `--exp_name`
- Saves each model separately
- Prints final loss comparison

### 3. Argument Explanation (argparse)

#### Basic Training Parameters

`--batch_size (int, default=8)`
Batch size for training and testing.

`--num_workers (int, default=4)`
Number of workers for dataloader.

`--lr (float, default=1e-3)`
Learning rate for Adam optimizer.

`--epochs (int, default=10)`
Number of training epochs.

#### Mode Control

`--multi_run (flag)`
If enabled:
- Runs multiple predefined experiments
- Ignores manual architecture settings

#### Model Architecture

`--use_ca (flag)`
Enable Channel Attention module in UNet.

`--use_sa (flag)`
Enable Spatial Attention module in UNet.

Note:
- Only effective when NOT using `--multi_run`

#### Loss Weights

Total loss:
$$\mathcal{L}={\lambda}_{1}\cdot\mathcal{L}_{SI\text{-}SDR}+{\lambda}_{2}\cdot\mathcal{L}_{spec}+{\lambda}_{3}\cdot\mathcal{L}_{complex}$$

`--lambda1 (float, default=1.0)`
Weight for SI-SDR loss (waveform domain).

`--lambda2 (float, default=0.0005)`
Weight for spectrogram loss (log-magnitude, masked).

`--lambda3 (float, default=0.0)`
Weight for complex spectrogram loss.

#### Experiment Naming

`--exp_name (str, default="baseline")`
Used ONLY for naming the checkpoint file.

Saved model path:
`ckpt/best_model_{exp_name}.pth`

Example:
`ckpt/best_model_ca.pth`

### 4. Data Format

Dataloader returns spectrogram-domain data:
- noisy: noisy magnitude spectrogram
- clean: clean magnitude spectrogram
- phase: phase information (for reconstruction)
- mask: time-frequency mask for valid regions

Important:
- STFT is already applied in dataset
- Model predicts magnitude mask only
- Phase is reused (not learned)

### 5. Training Pipeline

For each batch:
1. Predict mask from noisy spectrogram
2. Apply mask: enhanced = mask × noisy
3. Combine magnitude with phase → complex spectrogram
4. ISTFT → waveform
5. Compute losses:
   - SI-SDR (waveform)
   - Spectrogram loss (log magnitude)
   - Optional complex loss
6. Backpropagation

### 6. Model Saving

- Best model (lowest validation loss) is saved
- Path: `ckpt/best_model_{exp_name}.pth`

### 7. Notes

- Training is spectrogram-based, not waveform-based
- Phase is fixed (from noisy input)
- SI-SDR is the dominant loss term
- Spec loss is masked and usually small-weighted
- `--multi_run` is recommended for ablation experiments

### 8. Recommended Usage

Ablation study:
```bash
python model1/train.py --multi_run
```

Manual experiment:
```bash
python model1/train.py --use_sa --epochs 10
```

## Magnitude Model Evaluation Script Usage Summary

### 1. Overview
This script evaluates trained magnitude-mask-based U-Net models for speech enhancement.

Key points:
- Input from dataloader is **STFT spectrograms** (magnitude + phase), NOT raw waveform
- Model predicts **a magnitude mask**
- Waveform is reconstructed via ISTFT only for evaluation metrics
- Supports playback, full evaluation, saving audio, and plotting

### 2. Usage Modes

#### (1) Single Model Evaluation

Evaluate one trained model:

```bash
python model1/evaluate.py --exp_name baseline
```

Examples:

```bash
python model1/evaluate.py --exp_name ca --mode eval
python model1/evaluate.py --exp_name sa --mode play
python model1/evaluate.py --exp_name ca_sa --mode all
```

Behavior:
- Loads checkpoint: ckpt/best_model_{exp_name}.pth
- Model architecture is inferred from configs (or manual override if enabled)
- Runs selected evaluation mode(s)

#### (2) Multi-Model Evaluation (Comparison)

Evaluate all predefined models:

```bash
python model1/evaluate.py --multi_run --mode eval
```

Behavior:
- Runs:
  - baseline
  - ca
  - sa
  - ca_sa
- Automatically loads corresponding checkpoints
- Produces comparison results across models

### 3. Evaluation Modes (--mode)

Controls evaluation behavior:

- `play`
  - Plays noisy / enhanced / clean audio
  - Prints per-sample metrics
  - Limited by `--num_examples`

- `eval`
  - Full dataset evaluation
  - Computes:
    - SNR / SNRI
    - MAE / MSE / RMSE
    - PESQ / STOI
  - Saves CSV results

- `save`
  - Saves enhanced audio (`.wav` files)

- `plot`
  - Saves waveform and spectrogram figures

- `all (default)`
  - Runs all above modes

### 4. Argument Explanation (argparse)

#### Basic Settings

`--batch_size (int, default=8)`
Batch size for evaluation.

`--num_workers (int, default=4)`
Number of dataloader workers.

#### Mode Control

`--mode (str, default="all")`
Evaluation mode:
`play | eval | save | plot | all`

#### Multi-Model Evaluation

`--multi_run (flag)`
If enabled:
- Evaluates all predefined configurations
- Ignores manual architecture settings

#### Model Architecture

`--manual_arch (flag)`
Enables manual override of model structure.

`--use_ca (flag)`
Enable Channel Attention module.

`--use_sa (flag)`
Enable Spatial Attention module.

Note:
- Only used when `--manual_arch` is enabled

#### Experiment Name

`--exp_name (str, default="baseline")`
Used for:
- Selecting checkpoint file
- Naming outputs

Checkpoint path:
`ckpt/best_model_{exp_name}.pth`

#### Playback Settings

`--num_examples (int, default=4)`
Number of audio samples to play in play mode.

#### Dataset Control

`--use_subset (flag)`
Use subset of test data instead of full dataset.

`--snr (float, default=12.5)`
SNR level for subset generation.

Note:
- Only effective when `--use_subset` is enabled
- Mainly used for controlled evaluation conditions

### 5. Outputs

#### Evaluation Metrics (eval mode)
Printed and saved:

- SNR / SNRI
- MAE / MSE / RMSE
- PESQ / STOI

Saved to:
`results/eval_full_{exp_name}.csv`

#### Audio Output (save mode)

Saved enhanced audio:
`outputs/{exp_name}/*.wav`

#### Figures (plot mode)

Waveform:
`figures/waveform_comparison/{exp_name}/`

Spectrogram:
`figures/spectrogram_comparison/{exp_name}/`

### 6. Important Notes

- Model operates on STFT spectrograms, not waveform
- Phase is reused from noisy input
- ISTFT is used only for:
  - SI-SDR computation
  - perceptual metrics (PESQ/STOI)
  - visualization
- Multi-run mode ensures consistent comparison across architectures
- PESQ/STOI may fail for invalid audio lengths and return None

## Complex Model Training Usage Summary

### 1. Overview
This script trains a U-Net based speech enhancement model using **complex spectrogram masking**.

Unlike magnitude-only models, this version directly operates on **complex STFT representations**, where both magnitude and phase are used during training.

The model predicts a **complex-domain mask**, and waveform reconstruction is performed via ISTFT only for loss computation.

---

### 2. Usage Modes

#### (1) Single Model Training
Train one model with default complex configuration:

```bash
python model2/train.py --exp_name complex
```

Behavior:
- Uses a standard U-Net (no attention variants in this version)
- Saves model as:
  `ckpt/best_model_complex.pth`

---

### 3. Argument Explanation (argparse)

#### Basic Training Parameters

`--batch_size (int, default=8)`
Batch size for training and testing.

`--num_workers (int, default=4)`
Number of dataloader worker processes.

`--lr (float, default=1e-3)`
Learning rate for Adam optimizer.

`--epochs (int, default=10)`
Number of training epochs.

---

#### Loss Weights

Total loss is defined as:

$$\mathcal{L}={\lambda}_{1}\cdot\mathcal{L}_{SI\text{-}SDR}+{\lambda}_{2}\cdot\mathcal{L}_{spec}+{\lambda}_{3}\cdot\mathcal{L}_{phase}$$

`--lambda1 (float, default=1.0)`
Weight for SI-SDR loss (time-domain waveform quality).

`--lambda2 (float, default=0.0005)`
Weight for log-magnitude spectrogram loss.

`--lambda3 (float, default=5.0)`
Weight for phase-consistency loss (complex domain alignment).

#### Experiment Naming

`--exp_name (str, default="complex")`
Used for naming checkpoint files.

Saved model path:
`ckpt/best_model_{exp_name}.pth`

Example:
`ckpt/best_model_complex.pth`


### 4. Data Format

The dataloader returns **complex spectrogram-related components**, NOT raw audio:

- `noisy`: noisy STFT representation
- `clean`: clean STFT representation
- `mask`: time-frequency mask

Important notes:
- STFT is already computed in the dataset
- Model operates on complex-valued representations
- Waveform is reconstructed only for loss computation

### 5. Complex Representation

Complex spectrogram is constructed as:

$$X={X}_{r}+j{X}_{i}$$

In code:
```python
noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
clean_complex = torch.complex(clean[:, 0], clean[:, 1])
```

### 6. Training Pipeline

For each batch:

#### Step 1: Predict mask

$$\hat{M}={f}_{\theta}(X)$$

#### Step 2: Apply complex mask

$$\hat{X}=\hat{M}\cdot X$$

#### Step 3: Waveform reconstruction

$$\hat{x}=\text{ISTFT}(\hat{X}),\quad x=\text{ISTFT}({X}_{clean})$$

#### Step 4: Loss computation

##### (1) SI-SDR Loss

$$\mathcal{L}_{SI\text{-}SDR} = -10 \log_{10} \left( \frac{\|x_{proj}\|^2}{\|x_{noise}\|^2 + \epsilon} \right)$$

##### (2) Log-Magnitude Spectrogram Loss
$$\mathcal{L}_{spec}=\frac{\sum|\log|\hat{X}|-\log|X||\cdot\text{mask}}{\sum\text{mask}}$$

##### (3) Phase Consistency Loss

Phase similarity is computed using normalized complex inner product:

$$\mathcal{L}_{phase}=1-\frac{\Re(\hat{X}\cdot X^*)}{|\hat{X}||X|+\epsilon}$$

Then spatially weighted:

$$\mathcal{L}_{phase}\leftarrow\text{mean over high-energy regions}$$

where mask is defined as:

$$\text{mask} = \mathbb{1}(|X| > \text{mean}(|X|))$$

### 7. Model Saving

- Best model is selected by validation loss
- Saved automatically during training

Path:
`ckpt/best_model_{exp_name}.pth`

### 8. Key Differences vs Magnitude Model

| Aspect | Magnitude Model | Complex Model |
|--------|----------------|---------------|
| Input | Magnitude spectrogram | Complex STFT |
| Output | Magnitude mask | Complex mask |
| Phase usage | Fixed reuse | Explicitly modeled |
| Loss terms | SI-SDR + Spec | SI-SDR + Spec + Phase |

### 9. Notes

- This model is more sensitive but more expressive
- Phase loss stabilizes reconstruction quality
- SI-SDR remains the dominant optimization signal
- Spectrogram loss is still magnitude-based but secondary
- Training stability depends heavily on ${\lambda}_{3}$ (phase weight)

### 10. Recommended Usage

Single experiment:

```bash
python model2/train.py --exp_name complex
```

Tuning loss balance:

```bash
python model2/train.py --lambda3 2.0
python model2/train.py --lambda3 10.0
```

Recommended default:
- lambda1 = 1.0
- lambda2 = 0.0005
- lambda3 = 5.0

## Complex Model Evaluation Usage Summary

### 1. Overview
This script evaluates a trained **complex spectrogram mask-based U-Net** for speech enhancement.

Key characteristics:
- Input is **complex STFT representation** (real + imaginary channels)
- Model predicts a **complex-domain mask**
- Enhanced signal is reconstructed via ISTFT for evaluation and visualization
- Supports:
  - Audio playback
  - Full dataset evaluation
  - Audio saving
  - Waveform / spectrogram visualization

### 2. Usage Modes

#### (1) Single Model Evaluation

Evaluate a single trained complex model:

```bash
python model2/evaluate.py --exp_name complex
```

Examples:

```bash
python model2/evaluate.py --exp_name complex --mode eval
python model2/evaluate.py --exp_name complex --mode play
python model2/evaluate.py --exp_name complex --mode all
```

Behavior:
- Loads checkpoint:
  `ckpt/best_model_{exp_name}.pth`
- Evaluates using selected mode
- Uses predefined UNet architecture (no attention variants)

### 3. Evaluation Modes (--mode)

Controls what evaluation pipeline runs:

- `play`
  - Plays noisy / enhanced / clean audio
  - Prints per-sample metrics and loss breakdown
  - Limited by `--num_examples`

- `eval`
  - Full dataset evaluation
  - Computes:
    - SNR / SNRI
    - MAE / MSE / RMSE
    - PESQ / STOI
  - Saves results to CSV

- `save`
  - Saves enhanced audio as `.wav`

- `plot`
  - Saves waveform and spectrogram visualizations

- `all (default)`
  - Runs all evaluation modes above

### 4. Argument Explanation (argparse)

#### Basic Settings

`--batch_size (int, default=8)`
Batch size used during evaluation.

`--num_workers (int, default=4)`
Number of dataloader worker processes.

#### Mode Control

`--mode (str, default="all")`
Evaluation mode selection:
`play | eval | save | plot | all`

#### Experiment Name / Checkpoint

`--exp_name (str, default="complex")`
Specifies which model checkpoint to load.

Checkpoint path:
`ckpt/best_model_{exp_name}.pth`

Used also for naming outputs.

#### Dataset Control

`--use_subset (flag)`
Use a subset of test data instead of full dataset.

`--snr (float, default=12.5)`
Controls SNR level of subset dataset.

Note:
- Only effective when `--use_subset` is enabled
- Used for controlled evaluation settings

#### Playback Control

`--num_examples (int, default=4)`
Number of samples played in `play` mode.

### 5. Evaluation Outputs

#### (1) Full Evaluation (eval mode)

Outputs per-sample and averaged metrics including:
- SNR / SNRI
- MAE / MSE / RMSE
- PESQ / STOI

Saved to:
`results/eval_full_{exp_name}.csv`

#### (2) Audio Output (save mode)

Enhanced audio is saved as:
`outputs/{exp_name}/*.wav`

#### (3) Visualization Output (plot mode)

- Waveform plots:
  `figures/waveform_comparison/{exp_name}/`

- Spectrogram plots:
  `figures/spectrogram_comparison/{exp_name}/`

### 6. Important Notes

- Model operates on **complex STFT inputs**
- Phase is explicitly modeled via complex representation
- ISTFT is used only for:
  - waveform reconstruction for evaluation
  - perceptual metric computation (PESQ / STOI)
  - visualization

- Evaluation supports both full dataset and subset-controlled testing
- PESQ/STOI may fail on invalid signals and return `None`
- Outputs are automatically normalized before saving or playback

### 7. Recommended Usage

#### Full evaluation pipeline:
```bash
python model2/evaluate.py --exp_name complex --mode all
```

#### Quick qualitative check:
```bash
python model2/evaluate.py --exp_name complex --mode play
```

#### Generate results + plots:
```bash
python model2/evaluate.py --exp_name complex --mode eval
python model2/evaluate.py --exp_name complex --mode plot
```

# 语音增强课程作业

本仓库是一份完整的语音增强课程作业实现。整体思路是先完成经典语音增强方法，建立传统 baseline；再引入神经网络方法；最后使用统一指标对所有方法进行比较，并给出总结结论。

## 作业内容

本作业主要完成了两部分：

1. 经典语音增强方法
2. 神经网络语音增强方法

经典方法部分实现了：

- Spectral Subtraction
- Wavelet Denoising
- Frequency Masking

神经网络部分实现了：

- Magnitude Mask U-Net
- Complex Mask U-Net

## 数据集

实验基于 Edinburgh DataShare 提供的 Noisy speech database 配对语音数据开展。仓库未直接附带原始数据，需要自行下载。

数据集页面：
https://datashare.ed.ac.uk/handle/10283/2791

测试集目录应为：

```text
data/
  noisy_testset_wav/
  clean_testset_wav/
```

## 最终结果表

 6 个方法的结果：

| Method | SNRI | MAE | RMSE | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| Noisy Input | 0.000 | 0.024 | 0.030 | 1.967 | 0.921 |
| Spectral Subtraction | 5.445 | 0.010 | 0.0156 | 2.2948 | 0.912 |
| Wavelet Denoising | 0.220 | 0.023 | 0.029 | 2.094 | 0.915 |
| Frequency Masking | 4.223 | 0.013 | 0.019 | 2.233 | 0.919 |
| Magnitude Mask U-Net | 11.062 | 0.009 | 0.013 | 2.850 | 0.947 |
| Complex Mask U-Net | 8.545 | 0.015 | 0.020 | 2.595 | 0.935 |

## 结论

- 在经典方法中，`Spectral Subtraction` 是表现最好的传统 baseline。
- `Wavelet Denoising` 效果最弱，提升有限。
- `Frequency Masking` 也优于原始带噪输入，但整体仍弱于最优神经网络方法。
- 神经网络方法整体优于经典方法。
- 最终保留的神经网络结果中，`Magnitude Mask U-Net` 表现最好，是本次课程作业的最佳方案。
- `Complex Mask U-Net` 虽然优于经典方法，但仍未超过 `Magnitude Mask U-Net`。

## 仓库说明

### 经典方法代码

- `src/spoken_denoise/`
- `scripts/run_parameter_search.py`
- `scripts/run_all_methods.py`
- `scripts/evaluate_results.py`
- `scripts/make_figures.py`

### 神经网络代码

- `model1/`: Magnitude Mask U-Net 路线
- `model2/`: Complex Mask U-Net 路线
- `ckpt/`: 已训练权重

### 结果与文档

- `results/final_summary.csv`: 经典方法汇总结果
- `results/eval_full_baseline.csv`: Magnitude Mask U-Net 结果来源
- `results/eval_full_com.csv`: Complex Mask U-Net 结果来源
- `docs/experiment_report.md`: 完整实验报告
- `docs/code_structure.md`: 代码结构说明

## 复现命令

安装依赖：

```bash
pip install -r requirements.txt
```

运行经典方法：

```bash
python scripts/run_parameter_search.py
python scripts/run_all_methods.py
python scripts/evaluate_results.py
```

生成展示图像：

```bash
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise --target-sr 16000
```

## 幅度模型训练使用说明

### 1. 概述
本脚本用于训练基于 U-Net 的**幅度掩蔽**语音增强模型。
模型在 **STFT 频谱**（而非原始音频）上运行，仅在计算损失时通过 ISTFT 重建波形。

### 2. 使用模式

#### (1) 单模型训练
手动指定架构训练单个模型：

```bash
python model1/train.py --exp_name baseline
```

示例：
```bash
python model1/train.py --use_ca --exp_name ca
python model1/train.py --use_sa --exp_name sa
python model1/train.py --use_ca --use_sa --exp_name ca_sa
```

行为：
- 模型结构由 `--use_ca` 和 `--use_sa` 决定
- 模型保存为：`ckpt/best_model_{exp_name}.pth`

#### (2) 多模型训练（消融实验）
自动训练多个预定义模型：

```bash
python model1/train.py --multi_run
```

此命令运行以下配置：
- baseline（无注意力）
- ca（通道注意力）
- sa（跳跃注意力）
- ca_sa（两者都有）

行为：
- 忽略：`--use_ca`、`--use_sa`、`--exp_name`
- 分别保存每个模型
- 打印最终损失对比

### 3. 参数说明（argparse）

#### 基础训练参数

`--batch_size (int, 默认=8)`
训练和测试的批次大小。

`--num_workers (int, 默认=4)`
数据加载器的子进程数量。

`--lr (float, 默认=1e-3)`
Adam 优化器的学习率。

`--epochs (int, 默认=10)`
训练的轮数。

#### 模式控制

`--multi_run (flag)`
如果启用：
- 运行多个预定义实验
- 忽略手动设置的架构参数

#### 模型架构

`--use_ca (flag)`
启用 UNet 中的通道注意力模块。

`--use_sa (flag)`
启用 UNet 中的空间注意力模块。

注意：
- 仅当不使用 `--multi_run` 时生效

#### 损失权重

总损失：
$$\mathcal{L}={\lambda}_{1}\cdot\mathcal{L}_{SI\text{-}SDR}+{\lambda}_{2}\cdot\mathcal{L}_{spec}+{\lambda}_{3}\cdot\mathcal{L}_{complex}$$

`--lambda1 (float, 默认=1.0)`
SI-SDR 损失（波形域）的权重。

`--lambda2 (float, 默认=0.0005)`
频谱损失（对数幅度，使用掩蔽）的权重。

`--lambda3 (float, 默认=0.0)`
复数频谱损失的权重。

#### 实验命名

`--exp_name (str, 默认="baseline")`
仅用于命名检查点文件。

模型保存路径：
`ckpt/best_model_{exp_name}.pth`

示例：
`ckpt/best_model_ca.pth`

### 4. 数据格式

数据加载器返回频谱域的数据：
- noisy：带噪幅度频谱
- clean：纯净幅度频谱
- phase：相位信息（用于重建）
- mask：用于有效区域的时频掩蔽

重要说明：
- 数据集中已经应用了 STFT
- 模型仅预测幅度掩蔽
- 相位被复用（不被学习）

### 5. 训练流程

对于每个批次：
1. 从带噪频谱预测掩蔽
2. 应用掩蔽：增强后 = 掩蔽 × 带噪
3. 结合幅度与相位 → 复数频谱
4. ISTFT → 波形
5. 计算损失：
   - SI-SDR（波形）
   - 频谱损失（对数幅度）
   - 可选的复数损失
6. 反向传播

### 6. 模型保存

- 保存在验证集上损失最低的模型
- 路径：`ckpt/best_model_{exp_name}.pth`

### 7. 注意事项

- 训练是基于频谱的，而非基于波形
- 相位是固定的（来自带噪输入）
- SI-SDR 是主要的损失项
- 频谱损失带有掩蔽，权重通常较小
- 推荐使用 `--multi_run` 进行消融实验

### 8. 推荐用法

消融研究：
```bash
python model1/train.py --multi_run
```

手动实验：
```bash
python model1/train.py --use_sa --epochs 10
```

## 幅度模型评估脚本使用说明

### 1. 概述
本脚本用于评估已训练的基于幅度掩蔽的 U-Net 语音增强模型。

关键点：
- 数据加载器的输入是 **STFT 频谱**（幅度 + 相位），而非原始波形
- 模型预测**幅度掩蔽**
- 仅为了评估指标才会通过 ISTFT 重建波形
- 支持音频播放、完整评估、保存音频和绘图

### 2. 使用模式

#### (1) 单模型评估

评估单个已训练模型：

```bash
python model1/evaluate.py --exp_name baseline
```

示例：

```bash
python model1/evaluate.py --exp_name ca --mode eval
python model1/evaluate.py --exp_name sa --mode play
python model1/evaluate.py --exp_name ca_sa --mode all
```

行为：
- 加载检查点：ckpt/best_model_{exp_name}.pth
- 模型架构从配置推断（或通过手动覆盖启用）
- 运行所选的评估模式

#### (2) 多模型评估（对比）

评估所有预定义模型：

```bash
python model1/evaluate.py --multi_run --mode eval
```

行为：
- 运行：
  - baseline
  - ca
  - sa
  - ca_sa
- 自动加载对应的检查点
- 生成跨模型的对比结果

### 3. 评估模式（--mode）

控制评估行为：

- `play`
  - 播放带噪/增强/纯净音频
  - 打印每个样本的指标
  - 受 `--num_examples` 限制

- `eval`
  - 全数据集评估
  - 计算：
    - SNR / SNRI
    - MAE / MSE / RMSE
    - PESQ / STOI
  - 保存 CSV 结果

- `save`
  - 保存增强后的音频（`.wav` 文件）

- `plot`
  - 保存波形和频谱图

- `all (默认)`
  - 运行上述所有模式

### 4. 参数说明（argparse）

#### 基础设置

`--batch_size (int, 默认=8)`
评估时使用的批次大小。

`--num_workers (int, 默认=4)`
数据加载器的工作进程数。

#### 模式控制

`--mode (str, 默认="all")`
评估模式：
`play | eval | save | plot | all`

#### 多模型评估

`--multi_run (flag)`
如果启用：
- 评估所有预定义配置
- 忽略手动架构设置

#### 模型架构

`--manual_arch (flag)`
启用手动覆盖模型结构。

`--use_ca (flag)`
启用通道注意力模块。

`--use_sa (flag)`
启用空间注意力模块。

注意：
- 仅当启用 `--manual_arch` 时使用

#### 实验名称

`--exp_name (str, 默认="baseline")`
用于：
- 选择检查点文件
- 命名输出

检查点路径：
`ckpt/best_model_{exp_name}.pth`

#### 播放设置

`--num_examples (int, 默认=4)`
在播放模式下播放的音频样本数。

#### 数据集控制

`--use_subset (flag)`
使用测试数据的子集而非完整数据集。

`--snr (float, 默认=12.5)`
用于生成子集的 SNR 水平。

注意：
- 仅当启用 `--use_subset` 时生效
- 主要用于受控的评估条件

### 5. 输出

#### 评估指标（eval 模式）
打印并保存：

- SNR / SNRI
- MAE / MSE / RMSE
- PESQ / STOI

保存至：
`results/eval_full_{exp_name}.csv`

#### 音频输出（save 模式）

保存增强后的音频：
`outputs/{exp_name}/*.wav`

#### 图像（plot 模式）

波形图：
`figures/waveform_comparison/{exp_name}/`

频谱图：
`figures/spectrogram_comparison/{exp_name}/`

### 6. 重要注意事项

- 模型在 STFT 频谱上运行，而非波形
- 相位从带噪输入复用
- ISTFT 仅用于：
  - SI-SDR 计算
  - 感知指标（PESQ/STOI）
  - 可视化
- 多运行模式确保跨架构的一致比较
- PESQ/STOI 可能因无效音频长度而失败并返回 None

## 复数模型训练使用说明

### 1. 概述
本脚本用于训练基于 U-Net 的 **复数频谱掩蔽** 语音增强模型。

与仅使用幅度的模型不同，该版本直接操作 **复数 STFT 表示**，在训练过程中同时使用幅度和相位。

模型预测 **复数域掩蔽**，波形重建仅为了计算损失而通过 ISTFT 执行。

---

### 2. 使用模式

#### (1) 单模型训练
使用默认复数配置训练单个模型：

```bash
python model2/train.py --exp_name complex
```

行为：
- 使用标准的 U-Net（该版本无注意力变体）
- 模型保存为：
  `ckpt/best_model_complex.pth`

---

### 3. 参数说明（argparse）

#### 基础训练参数

`--batch_size (int, 默认=8)`
训练和测试的批次大小。

`--num_workers (int, 默认=4)`
数据加载器的工作进程数。

`--lr (float, 默认=1e-3)`
Adam 优化器的学习率。

`--epochs (int, 默认=10)`
训练的轮数。

---

#### 损失权重

总损失定义如下：

$$\mathcal{L}={\lambda}_{1}\cdot\mathcal{L}_{SI\text{-}SDR}+{\lambda}_{2}\cdot\mathcal{L}_{spec}+{\lambda}_{3}\cdot\mathcal{L}_{phase}$$

`--lambda1 (float, 默认=1.0)`
SI-SDR 损失（时域波形质量）的权重。

`--lambda2 (float, 默认=0.0005)`
对数幅度频谱损失的权重。

`--lambda3 (float, 默认=5.0)`
相位一致性损失（复数域对齐）的权重。

#### 实验命名

`--exp_name (str, 默认="complex")`
用于命名检查点文件。

模型保存路径：
`ckpt/best_model_{exp_name}.pth`

示例：
`ckpt/best_model_complex.pth`


### 4. 数据格式

数据加载器返回 **与复数频谱相关的分量**，而非原始音频：

- `noisy`：带噪 STFT 表示
- `clean`：纯净 STFT 表示
- `mask`：时频掩蔽

重要说明：
- STFT 已在数据集中计算完成
- 模型对复数值表示进行操作
- 仅为了损失计算才重建波形

### 5. 复数表示

复数频谱构造如下：

$$X={X}_{r}+j{X}_{i}$$

在代码中：
```python
noisy_complex = torch.complex(noisy[:, 0], noisy[:, 1])
clean_complex = torch.complex(clean[:, 0], clean[:, 1])
```

### 6. 训练流程

对于每个批次：

#### 第 1 步：预测掩蔽

$$\hat{M}={f}_{\theta}(X)$$

#### 第 2 步：应用复数掩蔽

$$\hat{X}=\hat{M}\cdot X$$

#### 第 3 步：波形重建

$$\hat{x}=\text{ISTFT}(\hat{X}),\quad x=\text{ISTFT}({X}_{clean})$$

#### 第 4 步：损失计算

##### (1) SI-SDR 损失

$$\mathcal{L}_{SI\text{-}SDR} = -10 \log_{10} \left( \frac{\|x_{proj}\|^2}{\|x_{noise}\|^2 + \epsilon} \right)$$

##### (2) 对数幅度频谱损失
$$\mathcal{L}_{spec}=\frac{\sum|\log|\hat{X}|-\log|X||\cdot\text{mask}}{\sum\text{mask}}$$

##### (3) 相位一致性损失

使用归一化复数内积计算相位相似度：

$$\mathcal{L}_{phase}=1-\frac{\Re(\hat{X}\cdot X^*)}{|\hat{X}||X|+\epsilon}$$

然后进行空间加权：

$$\mathcal{L}_{phase}\leftarrow\text{高能量区域的平均值}$$

其中掩蔽定义为：

$$\text{mask} = \mathbb{1}(|X| > \text{mean}(|X|))$$

### 7. 模型保存

- 根据验证损失选择最佳模型
- 训练过程中自动保存

路径：
`ckpt/best_model_{exp_name}.pth`

### 8. 与幅度模型的主要区别

| 方面 | 幅度模型 | 复数模型 |
|--------|----------------|---------------|
| 输入 | 幅度频谱 | 复数 STFT |
| 输出 | 幅度掩蔽 | 复数掩蔽 |
| 相位使用 | 固定复用 | 显式建模 |
| 损失项 | SI-SDR + Spec | SI-SDR + Spec + Phase |

### 9. 注意事项

- 该模型更敏感但更具表现力
- 相位损失能稳定重建质量
- SI-SDR 仍是主要的优化信号
- 频谱损失仍基于幅度，但处于次要地位
- 训练稳定性在很大程度上取决于 ${\lambda}_{3}$（相位权重）

### 10. 推荐用法

单次实验：

```bash
python model2/train.py --exp_name complex
```

调节损失平衡：

```bash
python model2/train.py --lambda3 2.0
python model2/train.py --lambda3 10.0
```

推荐默认值：
- lambda1 = 1.0
- lambda2 = 0.0005
- lambda3 = 5.0

## 复数模型评估使用说明

### 1. 概述
本脚本用于评估已训练的基于 **复数频谱掩蔽的 U-Net** 语音增强模型。

关键特性：
- 输入为 **复数 STFT 表示**（实部 + 虚部通道）
- 模型预测 **复数域掩蔽**
- 增强信号通过 ISTFT 重建，用于评估和可视化
- 支持：
  - 音频播放
  - 全数据集评估
  - 音频保存
  - 波形 / 频谱图可视化

### 2. 使用模式

#### (1) 单模型评估

评估单个已训练的复数模型：

```bash
python model2/evaluate.py --exp_name complex
```

示例：

```bash
python model2/evaluate.py --exp_name complex --mode eval
python model2/evaluate.py --exp_name complex --mode play
python model2/evaluate.py --exp_name complex --mode all
```

行为：
- 加载检查点：
  `ckpt/best_model_{exp_name}.pth`
- 使用所选模式进行评估
- 使用预定义的 UNet 架构（无注意力变体）

### 3. 评估模式（--mode）

控制运行的评估流程：

- `play`
  - 播放带噪/增强/纯净音频
  - 打印每个样本的指标和损失分解
  - 受 `--num_examples` 限制

- `eval`
  - 全数据集评估
  - 计算：
    - SNR / SNRI
    - MAE / MSE / RMSE
    - PESQ / STOI
  - 将结果保存至 CSV

- `save`
  - 将增强后的音频保存为 `.wav`

- `plot`
  - 保存波形和频谱图可视化

- `all (默认)`
  - 运行上述所有评估模式

### 4. 参数说明（argparse）

#### 基础设置

`--batch_size (int, 默认=8)`
评估时使用的批次大小。

`--num_workers (int, 默认=4)`
数据加载器的工作进程数。

#### 模式控制

`--mode (str, 默认="all")`
评估模式选择：
`play | eval | save | plot | all`

#### 实验名称 / 检查点

`--exp_name (str, 默认="complex")`
指定要加载哪个模型检查点。

检查点路径：
`ckpt/best_model_{exp_name}.pth`

也用于命名输出。

#### 数据集控制

`--use_subset (flag)`
使用测试数据的子集而非完整数据集。

`--snr (float, 默认=12.5)`
控制子集数据集的 SNR 水平。

注意：
- 仅当启用 `--use_subset` 时生效
- 用于受控的评估设置

#### 播放控制

`--num_examples (int, 默认=4)`
在 `play` 模式下播放的样本数。

### 5. 评估输出

#### (1) 完整评估（eval 模式）

输出每个样本及平均指标，包括：
- SNR / SNRI
- MAE / MSE / RMSE
- PESQ / STOI

保存至：
`results/eval_full_{exp_name}.csv`

#### (2) 音频输出（save 模式）

增强后的音频保存为：
`outputs/{exp_name}/*.wav`

#### (3) 可视化输出（plot 模式）

- 波形图：
  `figures/waveform_comparison/{exp_name}/`

- 频谱图：
  `figures/spectrogram_comparison/{exp_name}/`

### 6. 重要注意事项

- 模型对 **复数 STFT 输入** 进行操作
- 相位通过复数表示被显式建模
- ISTFT 仅用于：
  - 用于评估的波形重建
  - 感知指标计算（PESQ / STOI）
  - 可视化

- 评估支持完整数据集和基于子集的受控测试
- PESQ/STOI 可能在无效信号上失败并返回 `None`
- 输出在保存或播放前会自动归一化

### 7. 推荐用法

#### 完整评估流程：
```bash
python model2/evaluate.py --exp_name complex --mode all
```

#### 快速定性检查：
```bash
python model2/evaluate.py --exp_name complex --mode play
```

#### 生成结果和图表：
```bash
python model2/evaluate.py --exp_name complex --mode eval
python model2/evaluate.py --exp_name complex --mode plot
```
