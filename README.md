# Spoken Speech Enhancement Experiments

本仓库用于传统语音增强算法实验，对 Edinburgh DataShare 的成对数据集进行降噪、评估和可视化。

当前仓库已经包含：

- `outputs/`: 三种传统方法在官方测试集上的增强后 wav
- `results/final_results.csv`: 每条语音的详细指标
- `results/final_summary.csv`: 各方法平均指标
- `results/parameter_search.csv`: 参数搜索结果
- `figures/`: 最终展示用波形图和语谱图
- `docs/experiment_report.md`: 实验报告
- `docs/code_structure.md`: 代码结构说明

因此，如果组内同学只是需要：

- 写实验结果
- 看最终指标
- 插入图到 PPT 或文档

可以直接使用仓库中的 `results/`、`figures/`、`docs/` 和 `outputs/`，不需要重新跑完整实验。

## Dataset

原始测试数据集未随仓库上传，需要自行下载并解压：

数据集页面：
https://datashare.ed.ac.uk/handle/10283/2791

本实验只需要两个测试集压缩包：

- `data/noisy_testset_wav/`: 官方带噪测试语音
- `data/clean_testset_wav/`: 官方干净参考语音

解压后目录应为：

```text
data/
  noisy_testset_wav/
  clean_testset_wav/
```

## Install

```powershell
pip install -r requirements.txt
```

## Quick Use

如果只想查看最终实验结果，优先看：

- `results/final_summary.csv`
- `docs/experiment_report.md`
- `figures/spectrogram_comparison/`

如果只想听增强结果，直接查看：

- `outputs/spectral_subtraction/`
- `outputs/frequency_masking/`
- `outputs/wavelet_denoise/`

## Run

运行三种传统算法并生成最终指标：

```powershell
python scripts/run_all_methods.py
```

运行参数搜索：

```powershell
python scripts/run_parameter_search.py
```

根据输出 wav 重新评估：

```powershell
python scripts/evaluate_results.py
```

生成波形和语谱图对比：

```powershell
python scripts/make_figures.py
```

只生成最终展示图：

```powershell
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise
```

## Final Results

当前最终汇总结果如下：

| Method | SNR Improvement | MAE | RMSE | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| Noisy Input | 0.000000 | 0.023733 | 0.030315 | 1.967331 | 0.921063 |
| Spectral Subtraction | 5.445068 | 0.010806 | 0.015594 | 2.294208 | 0.911659 |
| Wavelet Denoising | 0.220296 | 0.023010 | 0.029433 | 2.093818 | 0.915433 |
| Frequency Masking | 4.223062 | 0.013048 | 0.018704 | 2.232847 | 0.919065 |

结论概括：

- `spectral_subtraction` 是整体最强的传统 baseline，`SNR Improvement` 最高，误差最低。
- `frequency_masking` 次之，语音质量提升也比较稳定。
- `wavelet_denoise` 提升较弱，更适合作为对照方法。

## Recommended Figures

推荐 PPT 重点使用以下语谱图：

- `figures/spectrogram_comparison/spectral_subtraction/p232_290.png`
- `figures/spectrogram_comparison/frequency_masking/p232_290.png`
- `figures/spectrogram_comparison/frequency_masking/p257_098.png`
- `figures/spectrogram_comparison/spectral_subtraction/p232_006.png`

其中：

- `p232_290.wav` 是成功样本
- `p257_098.wav` 是强噪声样本
- `p232_006.wav` 是失败/局限性样本

## Repository Contents

- `outputs/`: 降噪后的 wav 文件
- `results/final_results.csv`: 每条语音的详细指标
- `results/final_summary.csv`: 各方法的平均指标
- `results/parameter_search.csv`: 参数搜索结果
- `figures/`: 波形和语谱图对比图（可根据此写论文和ppt）
- `docs/experiment_report.md`: 实验报告（可根据此写论文和ppt）
- `docs/code_structure.md`: 代码结构说明
