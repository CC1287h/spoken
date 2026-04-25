# 代码结构说明

```text
spoken/
  README.md
  requirements.txt

  data/
    noisy_testset_wav/
    clean_testset_wav/

  src/
    spoken_denoise/
      audio.py
      datasets.py
      metrics.py
      methods.py
      visualization.py
      denoise/
        spectral_subtraction.py
        wavelet_denoise.py
        frequency_masking.py

  scripts/
    run_all_methods.py
    run_parameter_search.py
    evaluate_results.py
    make_figures.py

  ckpt/
    best_model_baseline.pth
    best_model_ca.pth
    best_model_ca_sa.pth
    best_model_complex.pth
    best_model_sa.pth

  folder1/
    dataset.py
    evaluate.py
    model.py
    train.py

  folder2/
    dataset.py
    evaluate.py
    model.py
    train.py

  outputs/
  results/
  figures/
  docs/
    experiment_report.md
    code_structure.md
```

## 核心模块

### `src/spoken_denoise/audio.py`

负责音频输入输出和长度对齐：

- `load_audio`: 读取单声道 wav
- `save_audio`: 保存 wav
- `align_signals`: 将 clean/noisy/enhanced 对齐到相同长度
- `peak_normalize`: 防止增强结果写入 wav 时削波

### `src/spoken_denoise/datasets.py`

负责测试集配对：

- `find_audio_pairs`: 根据相同相对路径匹配 noisy 和 clean wav

### `src/spoken_denoise/denoise/`

包含三种传统降噪算法：

- `spectral_subtraction.py`: 谱减法
- `wavelet_denoise.py`: 小波阈值降噪
- `frequency_masking.py`: 频域掩蔽

每个算法函数都遵循同一接口：

```python
enhanced = method(samples, sample_rate, **params)
```

### `src/spoken_denoise/metrics.py`

负责客观指标计算：

- `SNR`
- `SNR Improvement`
- `MAE`
- `MSE`
- `RMSE`
- `PESQ`
- `STOI`

`PESQ` 和 `STOI` 依赖 `pesq`、`pystoi`，如果计算失败会返回空值，不影响其他指标。

### `src/spoken_denoise/methods.py`

集中管理方法注册和参数：

- 默认实验参数：供 `run_all_methods.py` 使用
- 参数搜索网格：供 `run_parameter_search.py` 使用

如需修改算法参数，优先改这个文件。

### `src/spoken_denoise/visualization.py`

负责生成：

- 波形对比图
- 语谱图对比图

## 脚本说明

### `scripts/run_parameter_search.py`

用途：对三种算法跑简单参数搜索。

默认输入：

```text
data/noisy_testset_wav/
data/clean_testset_wav/
```

默认输出：

```text
results/parameter_search.csv
```

### `scripts/run_all_methods.py`

用途：使用默认参数运行最终实验。

默认输出：

```text
outputs/{method}/
results/final_results.csv
results/final_summary.csv
```

### `scripts/evaluate_results.py`

用途：如果已经有增强后的 wav，可重新计算指标，不需要重新跑算法。

默认输出：

```text
results/reevaluated_results.csv
results/reevaluated_summary.csv
```

### `scripts/make_figures.py`

用途：根据 `outputs/` 中的增强结果生成图像。

默认会以 `16000 Hz` 加载音频后再绘图，保证可视化采样率统一。

当前仓库中的 `figures/` 只保留最终展示用的 3 条代表性样本，对应命令为：

```powershell
python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise --target-sr 16000
```

脚本仍支持两种生成模式：

- `--num-files N`: 按顺序生成前 N 条样本的图
- `--files file1.wav file2.wav ...`: 只为指定样本生成图，适合最终 PPT 筛图

默认输出：

```text
figures/waveform_comparison/
figures/spectrogram_comparison/
```

## 额外合并内容

本次仓库还合并了另一条神经语音增强分支的产物：

- `ckpt/`: 若干 U-Net 变体训练得到的模型权重
- `folder1/`: 幅度谱掩码版本的训练与评估代码
- `folder2/`: 复数谱/相位相关版本的训练与评估代码

这些目录与 `src/spoken_denoise/` 下的传统方法实现是并列关系，主要用于作业展示与结果汇总，不影响原有传统方法脚本的运行。

## 推荐工作流

1. 下载并解压官方测试集。
2. 运行 `python scripts/run_parameter_search.py` 查看参数效果。
3. 根据参数搜索结果修改 `src/spoken_denoise/methods.py` 的默认参数。
4. 运行 `python scripts/run_all_methods.py` 生成最终 wav 和指标。
5. 运行 `python scripts/make_figures.py --files p232_290.wav p257_098.wav p232_006.wav --methods spectral_subtraction frequency_masking wavelet_denoise --target-sr 16000` 生成 PPT 图。
