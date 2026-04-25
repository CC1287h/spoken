# 代码结构与实验文件说明

## 1. 顶层结构

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

  folder1/
    dataset.py
    model.py
    train.py
    evaluate.py

  folder2/
    dataset.py
    model.py
    train.py
    evaluate.py

  ckpt/
    best_model_baseline.pth
    best_model_ca.pth
    best_model_sa.pth
    best_model_ca_sa.pth
    best_model_complex.pth

  outputs/
  results/
  figures/
  docs/
    experiment_report.md
    code_structure.md
```

## 2. 第一部分：传统方法代码

传统方法代码集中在 `src/spoken_denoise/` 与 `scripts/` 中，对应课程作业的第一阶段 baseline 实验。

### 2.1 核心模块

#### `src/spoken_denoise/audio.py`

负责音频处理基础功能：

- 读取音频
- 保存音频
- 多段音频长度对齐
- 峰值归一化

#### `src/spoken_denoise/datasets.py`

负责根据 noisy-clean 对应关系组织测试数据。

#### `src/spoken_denoise/metrics.py`

负责统一计算指标：

- `SNR`
- `SNR Improvement`
- `MAE`
- `MSE`
- `RMSE`
- `PESQ`
- `STOI`

#### `src/spoken_denoise/methods.py`

负责传统方法的注册、默认参数与参数搜索网格配置。

#### `src/spoken_denoise/visualization.py`

负责绘制：

- 波形对比图
- 语谱图对比图

### 2.2 传统方法实现

位于 `src/spoken_denoise/denoise/`：

- `spectral_subtraction.py`: 谱减法
- `wavelet_denoise.py`: 小波降噪
- `frequency_masking.py`: 频域掩蔽

### 2.3 传统方法脚本

#### `scripts/run_parameter_search.py`

对三种传统方法执行参数搜索，输出结果保存在：

```text
results/parameter_search.csv
```

#### `scripts/run_all_methods.py`

使用最终确定参数，对测试集批量执行传统方法，并输出：

```text
outputs/{method}/
results/final_results.csv
results/final_summary.csv
```

#### `scripts/evaluate_results.py`

当增强结果已经存在时，可重新计算指标而不必重新跑算法。

#### `scripts/make_figures.py`

根据增强后的音频生成波形图和语谱图。当前仓库中的图像主要保留了 3 条代表性样本。

## 3. 第二部分：神经网络代码

神经网络实验对应课程作业的第二阶段，分为两条路线。

### 3.1 `folder1/`: 幅度谱掩码 U-Net 系列

该目录包含基于幅度谱掩码的 U-Net 实现及其注意力变体。

#### 文件说明

- `dataset.py`: 数据加载、STFT 变换和 batch 组织
- `model.py`: U-Net 及注意力模块定义
- `train.py`: 训练主程序
- `evaluate.py`: 评估、保存增强语音、生成图像

#### 最终保留结果

- Magnitude Mask U-Net

#### 最终采用的结果文件

- `results/eval_full_baseline.csv` -> Magnitude Mask U-Net

#### 主要权重文件

- `ckpt/best_model_baseline.pth`

### 3.2 `folder2/`: 复数频谱建模路线

该目录对应 Complex U-Net 路线，模型直接学习复数域掩码。

#### 文件说明

- `dataset.py`: 复数频谱输入组织
- `model.py`: 复数域 U-Net 结构
- `train.py`: Complex U-Net 训练程序
- `evaluate.py`: Complex U-Net 评估与可视化

#### 对应结果

- `results/eval_full_com.csv` -> Complex U-Net
- `ckpt/best_model_complex.pth` -> Complex U-Net 最优权重

## 4. 第三部分：实验结果文件

### 4.1 传统方法结果

- `results/final_results.csv`: 每条样本的详细结果
- `results/final_summary.csv`: 传统方法平均结果
- `results/parameter_search.csv`: 传统方法参数搜索结果

### 4.2 神经网络结果

- `results/eval_full_baseline.csv`
- `results/eval_full_com.csv`

课程作业最终表格只采用这两个神经网络结果文件。

### 4.3 历史保留结果

仓库中还保留了：

- `results/eval_full_ca.csv`
- `results/eval_full_skip.csv`
- `results/eval_full_ca_skip.csv`
- `results/eval_full_mag.csv`

这些文件属于历史实验结果保留项，不作为最终主结论的核心依据，但可以作为补充参考。

## 5. 第四部分：音频与可视化产物

### 5.1 `outputs/`

保存传统方法生成的增强后音频。

### 5.2 `results/baseline/`、`results/ca/`、`results/skip/`、`results/ca_skip/`、`results/complex/`

保存神经网络模型输出的增强音频文件。

### 5.3 `figures/`

保存代表性样本的：

- `waveform_comparison/`: 波形对比图
- `spectrogram_comparison/`: 语谱图对比图