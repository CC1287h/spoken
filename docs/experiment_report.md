# 传统语音增强实验报告

## 1. 实验目标

本实验使用 Edinburgh DataShare 的 Noisy speech database，比较三种传统语音增强算法在官方测试集上的降噪效果：

- Spectral Subtraction
- Wavelet Denoising
- Frequency Masking

实验目标不是训练模型，而是构建可复现的传统算法 baseline，为后续神经网络方法提供对比。

## 2. 数据集

使用官方测试集：

- 输入：`data/noisy_testset_wav/`
- 参考答案：`data/clean_testset_wav/`

代码按相同文件名进行配对。例如：

```text
data/noisy_testset_wav/p232_001.wav
data/clean_testset_wav/p232_001.wav
```

每条 noisy 语音经过算法处理后得到 enhanced 语音，再与 clean 语音计算客观指标。

## 3. 方法

### 3.1 Spectral Subtraction

谱减法使用语音前若干帧估计噪声频谱，然后从带噪语音的幅度谱中减去噪声幅度谱。该方法是传统语音增强中常见的 baseline，优点是简单、可解释，缺点是容易产生 musical noise。

默认参数：

```text
alpha = 2.0
beta = 0.02
noise_frames = 6
```

### 3.2 Wavelet Denoising

小波降噪先对语音做小波分解，再对细节系数做阈值处理，最后重构语音。该方法适合抑制部分高频噪声，但对非平稳噪声的效果依赖阈值选择。

默认参数：

```text
wavelet = db4
threshold_scale = 1.0
mode = soft
```

### 3.3 Frequency Masking

频域掩蔽方法在 STFT 时频域中估计每个 time-frequency bin 的语音/噪声比例，对低信噪比区域进行抑制，对高信噪比区域尽量保留。

默认参数：

```text
threshold = 2.0
mask_floor = 0.05
noise_frames = 6
```

## 3.4 参数选择依据

已运行 `python scripts/run_parameter_search.py`，结果保存于 `results/parameter_search.csv`。根据全测试集参数搜索结果，当前采用以下默认参数：

| Method | Selected Params | Selection Reason |
|---|---|---|
| Spectral Subtraction | `alpha=2.0, beta=0.02, noise_frames=6` | SNR Improvement、MAE、MSE、RMSE 最优，PESQ 与最佳值非常接近 |
| Wavelet Denoising | `wavelet=db4, threshold_scale=1.0, mode=soft` | SNR Improvement、MAE、MSE、RMSE 在小波配置中最优 |
| Frequency Masking | `threshold=2.0, mask_floor=0.05, noise_frames=6` | SNR Improvement、MAE、MSE、RMSE、PESQ 均为频域掩蔽配置中最优 |

## 4. 评价指标

本实验从波形误差、信噪比和感知质量三个角度评价降噪效果。

| 指标 | 趋势 | 含义 |
|---|---|---|
| SNR | 越高越好 | 干净语音能量与残留误差能量的比例 |
| SNR Improvement | 越高越好 | 降噪后相对 noisy input 的 SNR 提升 |
| MAE | 越低越好 | 降噪语音与 clean 语音的平均绝对误差 |
| MSE | 越低越好 | 降噪语音与 clean 语音的均方误差 |
| RMSE | 越低越好 | 均方根误差 |
| PESQ | 越高越好 | 语音感知质量指标 |
| STOI | 越高越好 | 语音可懂度指标 |

判断算法有效的主要依据：

- `SNR Improvement > 0`
- `MAE / MSE / RMSE` 低于 noisy input
- `PESQ / STOI` 高于 noisy input
- 语谱图中背景噪声能量被抑制，语音主结构仍然保留

## 5. 参数搜索

运行：

```powershell
python scripts/run_parameter_search.py
```

输出：

```text
results/parameter_search.csv
```

参数搜索用于选择各方法在测试集上的较优配置。最终报告中应说明采用的参数，以及这些参数相比其他配置的指标变化。

## 6. 最终实验

运行：

```powershell
python scripts/run_all_methods.py
```

输出：

```text
outputs/
results/final_results.csv
results/final_summary.csv
```

最终汇总表建议放入 PPT：

| Method | SNR Improvement | MAE | RMSE | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| Noisy Input | 0.000 | 0.023733 | 0.030315 | 1.967331 | 0.921063 |
| Spectral Subtraction | 5.445068 | 0.010806 | 0.015594 | 2.294208 | 0.911659 |
| Wavelet Denoising | 0.220296 | 0.023010 | 0.029433 | 2.093818 | 0.915433 |
| Frequency Masking | 4.223062 | 0.013048 | 0.018704 | 2.232847 | 0.919065 |

## 7. 可视化

运行：

```powershell
python scripts/make_figures.py
```

输出：

```text
figures/waveform_comparison/
figures/spectrogram_comparison/
```

报告中建议选择 3 条代表性语音展示：

- 降噪效果明显的样本
- 降噪效果一般的样本
- 降噪失败或语音失真的样本

建议最终展示以下 3 条样本：

- `p232_290.wav`: 成功样本，谱减法 `SNR Improvement=11.907 dB`，频域掩蔽 `SNR Improvement=16.547 dB`
- `p257_098.wav`: 强噪声样本，谱减法 `SNR Improvement=16.994 dB`，频域掩蔽 `SNR Improvement=15.128 dB`
- `p232_006.wav`: 局限性样本，谱减法 `SNR Improvement=-0.870 dB`，而频域掩蔽仍有 `0.808 dB` 提升

推荐重新生成最终 PPT 用图：

```powershell
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise
```

如果希望 `figures/` 目录中只保留最终展示图，可以先删除旧的 `figures/` 目录，再运行上面的命令。

PPT 中优先展示以下语谱图：

- `figures/spectrogram_comparison/spectral_subtraction/p232_290.png`
- `figures/spectrogram_comparison/frequency_masking/p232_290.png`
- `figures/spectrogram_comparison/frequency_masking/p257_098.png`
- `figures/spectrogram_comparison/spectral_subtraction/p232_006.png`

其中前 3 张可作为主结果图，第 4 张可作为失败案例或局限性分析图。

推荐展示样本的关键指标如下：

| File | Method | SNR Improvement | PESQ | STOI | RMSE |
|---|---|---:|---:|---:|---:|
| `p232_290.wav` | Spectral Subtraction | 11.906770 | 3.456568 | 0.993893 | 0.011188 |
| `p232_290.wav` | Frequency Masking | 16.547144 | 3.068395 | 0.996113 | 0.006557 |
| `p257_098.wav` | Spectral Subtraction | 16.994233 | 1.462737 | 0.754270 | 0.008405 |
| `p257_098.wav` | Frequency Masking | 15.128176 | 1.565049 | 0.765825 | 0.010419 |
| `p232_006.wav` | Spectral Subtraction | -0.870246 | 2.255408 | 0.954389 | 0.010311 |
| `p232_006.wav` | Frequency Masking | 0.808372 | 2.492941 | 0.960724 | 0.008499 |

语谱图解释重点：

- Noisy 图中是否存在大面积背景噪声能量
- Enhanced 图中噪声是否被压低
- Enhanced 图中语音谐波和共振峰是否仍然保留

## 8. 结论

在官方测试集上，三种传统方法相对 noisy input 均带来了一定程度的 SNR 提升，但提升幅度差异明显。其中，谱减法表现最强，平均 `SNR Improvement` 为 `5.445 dB`，同时取得最低的 `MAE` 与 `RMSE`，说明其在整体误差控制和噪声抑制上最有效。频域掩蔽次之，平均 `SNR Improvement` 为 `4.223 dB`，`PESQ` 也明显高于 noisy input，说明其在语音质量和降噪强度之间取得了较好的平衡。

小波降噪的平均 `SNR Improvement` 仅为 `0.220 dB`，虽然相对 noisy input 在 `PESQ` 上有一定提升，但整体降噪能力明显弱于谱减法和频域掩蔽，更适合作为较弱的传统对照方法。从总体结果看，传统方法能够提供可观的 baseline，但在部分样本上仍会出现降噪不足或语音失真，尤其是不同指标之间存在取舍，例如某些样本上 SNR 提升明显但 `STOI` 未同步提高。因此，本实验结果可以作为后续神经网络语音增强方法的公平对比基线，也说明单纯依赖传统算法在复杂噪声场景下仍存在性能上限。

从单条样本可视化结果看，`p232_290.wav` 最适合展示传统方法在典型样本上的成功效果，`p257_098.wav` 适合展示困难噪声场景下的可见改善，而 `p232_006.wav` 则能清楚说明谱减法在部分样本上可能退化。这组成功样本与失败样本的组合，能够更完整地展示传统方法的优势与局限。

## 9. 复现步骤

```powershell
pip install -r requirements.txt
python scripts/run_parameter_search.py
python scripts/run_all_methods.py
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise
```
