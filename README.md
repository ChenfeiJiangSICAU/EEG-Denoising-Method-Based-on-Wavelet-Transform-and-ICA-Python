## 1. 简介

`eeg_denoise.py` 用于对单个 EDF 头皮 EEG 进行离线去噪，并输出压缩 NPZ。

## 2. 使用场景与效果

- 适用于标准单极头皮 EEG、常规 10-20/10-10 通道布局和 0.5–45 Hz 分析。
- 可降低基线漂移、高频随机噪声及 ICLabel 判定的高噪声成分。
- 适合离线预处理、后续统计、建模及去噪前后质量检查。
- 不适合双极导联、颅内 EEG、实时处理或需要保留 45 Hz 以上活动的研究。
- 当前处理仅删除脑电概率 < 10% 且其余类别概率之和 > 50% 的成分，去噪相对温和；如需更强去噪，可自行调整删除规则并检查波形、频谱和 ICA 标签，具体详见`8.` 。

## 3. 主要流程

1. 使用 MNE 读取并预加载一个 EDF；目录输入只取排序后的第一个 EDF。
2. 清理通道名，匹配 `standard_1020`，删除非 EEG 和已标记坏道。
3. 执行 0.5 Hz 高通。
4. 各通道按 8 秒、50% 重叠进行 `sym8` 六层小波去噪。
5. 转为全脑平均参考，并在目标 1–100 Hz 分支拟合 extended Infomax ICA。
6. 使用 ICLabel 分类，删除脑电概率 < 10% 且其余类别概率之和 > 50% 的成分，其余成分全部保留。
7. 对重构 EEG 执行 45 Hz 低通，保存 NPZ，并可绘制前 40 秒对比图。

## 4. 输入格式与要求

- 输入：一个 `.edf` 文件，或包含 EDF 的目录。
- 目录模式只处理按名称排序后的第一个 EDF，不会批量处理全部文件。
- 通道名应能映射到 `standard_1020`；支持清理 `EEG/POLY` 前缀和常见参考后缀。
- 兼容 `T3→T7`、`T4→T8`、`T5→P7`、`T6→P8`。
- 清理及坏道剔除后至少需要 3 个有效 EEG 通道。
- 19、21、32、64 等不同通道数均可；通道越多，ICA 所需数据和计算量越大。
- ECG、EOG、EMG、Trigger 等未匹配通道会被删除，不进入输出。
- `Fp1-F7` 等双极导联、非标准名称及少于 3 个有效 EEG 的数据不可直接使用。
- EEG 不能包含 `NaN` 或 `Inf`，严重平坦、重复或桥接通道会降低 ICA 质量。
- ICA 要求 `ceil(n_times/ica_decim) >= 5×有效通道数`；实际应使用更多数据。

### 采样率

- 45 Hz 低通要求采样率严格高于 90 Hz。
- 8 秒 `sym8 level=6` 至少约需 960 点，因此综合最低采样率约为 120 Hz。
- 64 Hz、100 Hz 不可直接使用；120 Hz 是理论边界；128 Hz 及以上通常可用。
- 250/256 Hz 较合适；500/1000 Hz 可用但内存和计算成本更高。
- 脚本不重采样；非 256 Hz 数据的 D1–D6 频带会随采样率改变。

## 5. 输出格式

- 默认文件：`<输入名>_denoised.npz`；已有文件需用 `--overwrite` 覆盖。
- `eeg`：`float32`，形状 `[通道, 采样点]`，单位为伏特。
- `sfreq`、`ch_names`、`channel_types`：采样率、标准通道名和类型。
- `ic_labels`、`ic_label_confidence`：ICA 标签及分类置信度。
- `excluded_ica_components`、`kept_ica_components`：删除和保留的成分索引。
- `wavelet_detail_bands_hz`：当前采样率下 D1–D6 理论频带。
- 另保存源 EDF、测量日期、时长、通道映射、剔除通道和全部参数 JSON。

## 6. 小波参数

| 参数 | 数值 | 原因 |
|---|---:|---|
| 小波 | `sym8` | 平滑、近似对称，兼顾时频局部化和波形保真 |
| 层数 | `6` | 在 256 Hz 下覆盖 D1–D6，约 2–128 Hz |
| 边界模式 | `symmetric` | 镜像延拓，减少边界突变 |
| 噪声估计 | `D1_MAD` | 由最高频层稳健估计噪声，降低尖峰影响 |
| 基础阈值 | `sigma√(2lnN)` | 通用阈值随噪声和段长度自适应 |
| 阈值模式 | `garrote` | 比硬阈值连续，并减少大系数过度收缩 |
| 分段 | `8 s` | 兼顾局部估计、低频信息及六层分解 |
| 重叠 | `50%` | 配合 Hann 窗减少分段接缝 |

`sigma = median(|D1-median(D1)|) / 0.6744897501960817`。
各层阈值为 `lambda_j=lambda_1×2^(-(j-1)/2)`：

| 层 | 相对阈值 | 256 Hz 理论频带 |
|---|---:|---:|
| D1 | 1.000，最强 | 64–128 Hz |
| D2 | 0.707 | 32–64 Hz |
| D3 | 0.500 | 16–32 Hz |
| D4 | 0.354 | 8–16 Hz |
| D5 | 0.250 | 4–8 Hz |
| D6 | 0.177，最弱 | 2–4 Hz |

## 7. ICA、ICLabel 与其他参数

| 参数 | 数值 | 原因 |
|---|---:|---|
| ICA | extended Infomax | 分离不同统计分布的独立源 |
| `n_components` | `None` | 由 MNE 根据数据秩确定 |
| `max_iter` | `auto` | 由 MNE 管理最大迭代 |
| 随机种子 | `97` | 提高重复运行的一致性 |
| 拟合频带 | 目标 1–100 Hz | 减少慢漂移并满足 ICLabel 常见条件 |
| `ica_decim` | `1` | 默认使用全部点；增大可提速但降低拟合样本数 |
| 参考 | 全脑平均 | 匹配 ICLabel 常见训练条件 |
| ICLabel 规则 | 删除脑电概率 < 10% 且其余类别概率之和 > 50% 的成分 | 温和去噪；仅剔除高度确定的噪声成分，其余保留 |
| 最终滤波 | 45 Hz 低通 | 抑制高频残余 |
| 绘图 | 前 40 秒 | 黑色去噪前、红色去噪后 |

## 8. 按研究方向选择 ICLabel 标签

去噪强度可通过 ICA 的 `exclude` 自行调整，当前脚本默认删除脑电概率 < 10% 且其余类别概率之和 > 50% 的成分。
眼动、眨眼或眼—脑耦合任务可保留 `{"brain", "eye blink"}`，避免删除研究目标。
较温和的方案可只排除明确伪迹类别，并结合置信度、成分拓扑和频谱人工复核。
保留 `eye blink` 只保留 EEG 中分离出的眼源 ICA 成分，不会保留已被筛除的原始 EOG 通道。


<br>
<br>
<br>
<br>




## 1. Overview

`eeg_denoise.py` performs offline denoising of one scalp-EEG EDF and writes a compressed NPZ.


## 2. Use Cases and Effects

- Suitable for standard monopolar scalp EEG, conventional 10-20/10-10 layouts, and 0.5–45 Hz analysis.
- It can reduce baseline drift, high-frequency random noise, and components classified by ICLabel as high-noise.
- It supports offline preprocessing, downstream statistics, modeling, and before/after quality control.
- It is unsuitable for bipolar derivations, intracranial EEG, real-time use, or studies requiring activity above 45 Hz.
- The current processing removes only components with brain probability < 10% and summed non-brain probability > 50%, so denoising is relatively mild; check the waveform, spectrum, and ICA tags, and adjust the removal rule for your research direction. See `section 8` for details.

## 3. Main Pipeline

1. Read and preload one EDF with MNE; a directory supplies only its first sorted EDF.
2. Clean channel labels, match `standard_1020`, and remove non-EEG and marked-bad channels.
3. Apply a 0.5 Hz high-pass.
4. Denoise each channel with overlapping 8-second, six-level `sym8` wavelet segments.
5. Apply a common-average reference and fit extended Infomax ICA on a target 1–100 Hz branch.
6. Classify with ICLabel and remove components with brain probability < 10% and summed non-brain probability > 50%, keeping all others.
7. Apply a final 45 Hz low-pass, save NPZ, and optionally plot the first 40 seconds.

## 4. Input Format and Requirements

- Input: one `.edf` file or a directory containing EDF files.
- Directory mode processes only the first file after filename sorting; it is not a batch mode.
- Labels must map to `standard_1020`; `EEG/POLY` prefixes and common reference suffixes are cleaned.
- Legacy aliases are supported: `T3→T7`, `T4→T8`, `T5→P7`, and `T6→P8`.
- At least three valid EEG channels must remain after cleaning and bad-channel exclusion.
- Different counts such as 19, 21, 32, or 64 are supported; more channels require more data and computation.
- Unmatched ECG, EOG, EMG, trigger, and other channels are removed from the output.
- Bipolar labels such as `Fp1-F7`, unmapped names, and fewer than three usable EEG channels are unsupported.
- EEG must not contain `NaN` or `Inf`; flat, duplicate, or bridged channels reduce ICA quality.
- ICA requires `ceil(n_times/ica_decim) >= 5×valid_channels`; substantially more data is preferable.

### Sampling rate

- The 45 Hz low-pass requires a sampling rate strictly above 90 Hz.
- Eight-second `sym8 level=6` segments need about 960 samples, giving a combined minimum near 120 Hz.
- 64 and 100 Hz are unsupported; 120 Hz is a theoretical boundary; 128 Hz and above are generally usable.
- 250/256 Hz is well suited; 500/1000 Hz is usable but requires more RAM and computation.
- The script does not resample; D1–D6 bands change with sampling rate when it is not 256 Hz.

## 5. Output Format

- Default file: `<input_name>_denoised.npz`; use `--overwrite` to replace an existing file.
- `eeg`: `float32`, shape `[channels, samples]`, stored in volts.
- `sfreq`, `ch_names`, and `channel_types`: sampling rate, standard names, and types.
- `ic_labels` and `ic_label_confidence`: ICA labels and classification confidence.
- `excluded_ica_components` and `kept_ica_components`: removed and retained component indices.
- `wavelet_detail_bands_hz`: theoretical D1–D6 bands at the actual sampling rate.
- Source EDF, date, duration, channel mapping, dropped channels, and complete parameter JSON are also stored.

## 6. Wavelet Parameters

| Parameter | Value | Rationale |
|---|---:|---|
| Wavelet | `sym8` | Smooth and near-symmetric, balancing localization and waveform fidelity |
| Levels | `6` | Covers D1–D6, theoretically about 2–128 Hz at 256 Hz |
| Boundary mode | `symmetric` | Mirror extension reduces boundary discontinuities |
| Noise estimate | `D1_MAD` | Robust high-frequency estimate with reduced spike influence |
| Base threshold | `sigma√(2lnN)` | Universal threshold adapts to noise and segment length |
| Threshold mode | `garrote` | Continuous relative to hard thresholding with less large-coefficient shrinkage |
| Segment | `8 s` | Balances local estimation, low frequencies, and six-level feasibility |
| Overlap | `50%` | A Hann window reduces segment seams |

`sigma = median(|D1-median(D1)|) / 0.6744897501960817`.
Level thresholds use `lambda_j=lambda_1×2^(-(j-1)/2)`:

| Level | Relative threshold | Theoretical band at 256 Hz |
|---|---:|---:|
| D1 | 1.000, strongest | 64–128 Hz |
| D2 | 0.707 | 32–64 Hz |
| D3 | 0.500 | 16–32 Hz |
| D4 | 0.354 | 8–16 Hz |
| D5 | 0.250 | 4–8 Hz |
| D6 | 0.177, weakest | 2–4 Hz |

## 7. ICA, ICLabel, and Other Parameters

| Parameter | Value | Rationale |
|---|---:|---|
| ICA | Extended Infomax | Separates independent sources with different distributions |
| `n_components` | `None` | MNE determines it from data rank |
| `max_iter` | `auto` | MNE manages the maximum iterations |
| Random seed | `97` | Improves repeatability |
| Fitting band | Target 1–100 Hz | Reduces slow drift and matches common ICLabel conditions |
| `ica_decim` | `1` | Uses all samples; larger values are faster but provide fewer fitting samples |
| Reference | Common average | Matches common ICLabel training conditions |
| ICLabel rule | Remove components with brain probability < 10% and summed non-brain probability > 50% | Mild denoising; drops only high-confidence noise, keeps the rest |
| Final filter | 45 Hz low-pass | Suppresses high-frequency residuals |
| Plot | First 40 seconds | Black before and red after denoising |

## 8. Selecting ICLabel Labels by Research Goal

Denoising strength can be adjusted through ICA `exclude`; the current script removes only components with brain probability < 10% and summed non-brain probability > 50%.
Eye-movement, blink, or eye–brain coupling studies may retain `{"brain", "eye blink"}` to preserve the target signal.
A milder policy may exclude only explicit artifact classes and review confidence, topology, and spectra manually.
Retaining `eye blink` preserves ocular ICA sources separated from EEG, not original EOG channels removed during selection.
