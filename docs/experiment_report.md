# 语音增强课程作业实验报告

## 1. 实验目标

本课程作业围绕带噪语音增强任务展开，目标是在保留语音主体信息的同时尽可能抑制背景噪声，并通过统一指标比较不同方法的表现。

本作业按照以下路线完成：

1. 先实现经典语音增强方法，建立 baseline。
2. 再实现神经网络方法，提升语音增强性能。
3. 最后对所有方法进行统一评估与总结。

## 2. 数据集

实验使用 Edinburgh DataShare 提供的 Noisy speech database。数据包含带噪语音与对应干净语音，可用于监督式语音增强实验。

本作业主要使用：

- `data/noisy_testset_wav/`
- `data/clean_testset_wav/`

输入为 noisy speech，输出为 enhanced speech，参考标签为 clean speech。

## 3. 评价指标

实验统一采用以下指标：

| 指标 | 趋势 | 含义 |
|---|---|---|
| SNRI | 越高越好 | 增强后相对带噪语音的信噪比提升 |
| MAE | 越低越好 | 与干净语音的平均绝对误差 |
| RMSE | 越低越好 | 与干净语音的均方根误差 |
| PESQ | 越高越好 | 感知语音质量 |
| STOI | 越高越好 | 语音可懂度 |

## 4. 经典方法实验

本作业实现了三种经典语音增强方法：

### 4.1 Spectral Subtraction

谱减法通过估计噪声频谱并从带噪语音频谱中进行减法抑制，是经典语音增强方法中的典型 baseline。

### 4.2 Wavelet Denoising

小波降噪通过小波分解和阈值处理压制噪声，但对复杂噪声的适应性相对有限。

### 4.3 Frequency Masking

频域掩蔽方法在时频域中对低信噪比区域进行抑制，相比简单谱减法更灵活。

### 4.4 经典方法结果

| Method | SNRI | MAE | RMSE | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| Noisy Input | 0.000 | 0.024 | 0.030 | 1.967 | 0.921 |
| Spectral Subtraction | 5.445 | 0.010 | 0.0156 | 2.2948 | 0.912 |
| Wavelet Denoising | 0.220 | 0.023 | 0.029 | 2.094 | 0.915 |
| Frequency Masking | 4.223 | 0.013 | 0.019 | 2.233 | 0.919 |

### 4.5 经典方法分析

- `Spectral Subtraction` 是三种经典方法中最好的 baseline。
- `Frequency Masking` 也优于原始带噪语音，但略弱于谱减法。
- `Wavelet Denoising` 的提升最有限，在本实验中表现最弱。

## 5. 神经网络实验

在完成经典方法后，本作业进一步实现神经网络语音增强模型，最终保留两类主结果。

### 5.1 Magnitude Mask U-Net

该方法在时频域中预测幅度谱掩码，再与带噪语音频谱相乘得到增强结果。它是本次作业中表现最好的神经网络方法。

### 5.2 Complex Mask U-Net

该方法尝试在复数频谱域中进行建模，显式考虑实部和虚部信息，希望利用更完整的频域表达。

### 5.3 神经网络结果

| Method | SNRI | MAE | RMSE | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| Magnitude Mask U-Net | 11.062 | 0.009 | 0.013 | 2.850 | 0.947 |
| Complex Mask U-Net | 8.545 | 0.015 | 0.020 | 2.595 | 0.935 |

### 5.4 神经网络分析

- 两种神经网络方法都明显优于经典方法。
- `Magnitude Mask U-Net` 的综合表现最好。
- `Complex Mask U-Net` 虽然也优于传统方法，但没有超过 `Magnitude Mask U-Net`。

## 6. 最终总对比

本课程作业最终只保留如下总表：

| Method | SNRI | MAE | RMSE | PESQ | STOI |
|---|---:|---:|---:|---:|---:|
| Noisy Input | 0.000 | 0.024 | 0.030 | 1.967 | 0.921 |
| Spectral Subtraction | 5.445 | 0.010 | 0.0156 | 2.2948 | 0.912 |
| Wavelet Denoising | 0.220 | 0.023 | 0.029 | 2.094 | 0.915 |
| Frequency Masking | 4.223 | 0.013 | 0.019 | 2.233 | 0.919 |
| Magnitude Mask U-Net | 11.062 | 0.009 | 0.013 | 2.850 | 0.947 |
| Complex Mask U-Net | 8.545 | 0.015 | 0.020 | 2.595 | 0.935 |

## 7. 最终结论

通过本次课程作业，可以得到以下结论：

1. 经典方法能够提供清晰、可解释的 baseline，其中 `Spectral Subtraction` 最优。
2. 神经网络方法整体显著优于经典方法，说明学习型方法在语音增强任务上更有优势。
3. 在最终保留的神经网络结果中，`Magnitude Mask U-Net` 是综合表现最好的方案。
4. `Complex Mask U-Net` 仍然优于传统方法，但未超过 `Magnitude Mask U-Net`。

因此，本作业的最终最佳方法为：

`Magnitude Mask U-Net`

## 8. 复现说明

经典方法相关命令如下：

```powershell
pip install -r requirements.txt
python scripts/run_parameter_search.py
python scripts/run_all_methods.py
python scripts/evaluate_results.py
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise --target-sr 16000
```

仓库中已保留主要神经网络结果文件与模型权重。
