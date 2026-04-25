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

神经网络部分最终保留两类主结果：

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

- `folder1/`: Magnitude Mask U-Net 路线
- `folder2/`: Complex Mask U-Net 路线
- `ckpt/`: 已训练权重

### 结果与文档

- `results/final_summary.csv`: 经典方法汇总结果
- `results/eval_full_baseline.csv`: Magnitude Mask U-Net 结果来源
- `results/eval_full_com.csv`: Complex Mask U-Net 结果来源
- `docs/experiment_report.md`: 完整实验报告
- `docs/code_structure.md`: 代码结构说明

## 复现命令

安装依赖：

```powershell
pip install -r requirements.txt
```

运行经典方法：

```powershell
python scripts/run_parameter_search.py
python scripts/run_all_methods.py
python scripts/evaluate_results.py
```

生成展示图像：

```powershell
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise --target-sr 16000
```
