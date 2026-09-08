# 基于小波变换与 ICA 的 EEG 去噪

## 概述

`eeg_denoise.py` 对单个头皮脑电 EDF 记录进行离线去噪。管线结合小波阈值收缩与 ICA/ICLabel 伪迹去除，输出压缩 NPZ 文件。

## 技术路径

1. **读取 EDF** — 使用 MNE 加载并预加载一个 EDF 文件。
2. **通道标准化** — 清理 EDF 通道名（去除 `EEG`/`POLY` 前缀和参考后缀），映射到 `standard_1020` 蒙太奇，应用旧版别名（`T3→T7`、`T4→T8`、`T5→P7`、`T6→P8`），删除非 EEG 和坏通道。
3. **高通滤波** — 0.5 Hz FIR 零相位双向高通。
4. **分段小波去噪** — 逐通道、8 秒分段、50% 重叠、`sym8` 小波、6 层分解、对称延拓、Hann 窗重叠相加。
5. **公共平均参考** — 在 ICA 前应用，匹配 ICLabel 训练条件。
6. **ICA + ICLabel** — 在 1–100 Hz 副本上拟合 Extended Infomax ICA；ICLabel 分类成分；删除脑电概率 < 10% 且噪声概率 > 50% 的成分。
7. **低通滤波** — 对重构 EEG 施加 45 Hz FIR 零相位双向低通。
8. **输出** — 保存压缩 NPZ（含去噪 EEG 和元数据）；可选展示 40 秒去噪前后对比图。

## 小波参数

| 参数 | 数值 |
|---|---|
| 小波 | `sym8` |
| 分解层数 | 6 |
| 边界模式 | `symmetric` |
| 噪声估计 | D1 层 MAD |
| 基础阈值 | `sigma * sqrt(2 * ln(N))`（通用阈值） |
| 层间衰减 | `lambda_j = lambda_1 * 2^(-(j-1)/2)` |
| 阈值模式 | `garrote` |
| 分段长度 | 8 秒 |
| 重叠 | 50% |
| 窗函数 | Hann |

噪声 sigma：`sigma = median(|D1 - median(D1)|) / 0.6744897501960817`

256 Hz 下各层相对阈值与理论频带：

| 层 | 相对阈值 | 频带 |
|---|---|---|
| D1 | 1.000 | 64–128 Hz |
| D2 | 0.707 | 32–64 Hz |
| D3 | 0.500 | 16–32 Hz |
| D4 | 0.354 | 8–16 Hz |
| D5 | 0.250 | 4–8 Hz |
| D6 | 0.177 | 2–4 Hz |

## ICA 与 ICLabel 参数

| 参数 | 数值 |
|---|---|
| ICA 方法 | Extended Infomax |
| n_components | 由 MNE 根据数据秩确定 |
| max_iter | `auto`（MNE 管理） |
| 随机种子 | 97 |
| ICA 拟合频带 | 1–100 Hz |
| 参考 | 公共平均 |
| ICLabel 删除规则 | 删除脑电概率 < 10% 且噪声概率 > 50% 的成分 |
| 最终低通 | 45 Hz |

## 运行方式

```bash
python eeg_denoise.py <输入edf> [-o 输出.npz] [选项]
```

### 参数

| 参数 | 说明 |
|---|---|
| `input` | `.edf` 文件路径，或目录（处理排序后的第一个 EDF） |
| `-o, --output` | 输出 NPZ 路径（默认：`<输入名>_denoised.npz`） |
| `--ica-decim` | ICA 拟合降采样步长（默认：1，使用全部采样点） |
| `--overwrite` | 覆盖已有输出文件 |
| `--no-plot` | 跳过交互式对比图 |
| `--log-level` | 日志级别：DEBUG、INFO（默认）、WARNING、ERROR |

### 示例

```bash
python eeg_denoise.py patient.edf -o patient_denoised.npz
```

## 输入要求

- 一个 `.edf` 文件或包含 EDF 文件的目录。
- 通道名需能映射到 `standard_1020`。自动清理 `EEG`/`POLY` 前缀和常见参考后缀（`REF`、`LE`、`RE`、`AVG`、`A1`、`A2`）。
- 清理和坏通道剔除后至少保留 3 个有效 EEG 通道。
- 采样率需高于 120 Hz（45 Hz 低通要求 > 90 Hz；8 秒 `sym8 level=6` 需约 960 个采样点）。推荐 256 Hz。
- 脚本不重采样；非 256 Hz 时 D1–D6 频带随采样率变化。
- EEG 不能包含 NaN 或 Inf。
- 不支持双极导联（如 `Fp1-F7`）。

## 输出格式

NPZ 文件包含：

| 键 | 类型 | 说明 |
|---|---|---|
| `eeg` | `float32` `[通道, 采样点]` | 去噪后的 EEG，单位伏特 |
| `sfreq` | `float64` | 采样率 (Hz) |
| `ch_names` | `str_` | 标准 10-20 通道名 |
| `channel_types` | `str_` | 通道类型 |
| `unit` | `str_` | 单位 ("V") |
| `source_edf` | `str_` | 源 EDF 路径 |
| `measurement_date` | `str_` | 测量日期 (ISO) |
| `duration_seconds` | `float64` | 记录时长 |
| `ic_labels` | `str_` | 每个成分的 ICLabel 类别标签 |
| `ic_label_confidence` | `float32` | ICLabel 概率矩阵 |
| `excluded_ica_components` | `int64` | 删除的成分索引 |
| `kept_ica_components` | `int64` | 保留的成分索引 |
| `n_ica_components` | `int64` | ICA 成分数 |
| `ica_n_iterations` | `int64` | ICA 收敛迭代次数 |
| `ica_fit_band_hz` | `float64` | ICA 拟合频带 |
| `wavelet_detail_bands_hz` | `float64` | D1–D6 理论频带 |
| `renamed_channels_json` | `str_` | 通道重命名映射 (JSON) |
| `dropped_non_eeg_channels` | `str_` | 删除的非 EEG 通道名 |
| `dropped_bad_channels` | `str_` | 删除的坏通道名 |
| `processing_parameters_json` | `str_` | 完整处理参数 (JSON) |

## 实际效果

- 通过 0.5 Hz 高通降低基线漂移。
- 通过层间递减小波阈值收缩降低高频随机噪声（D1 最强、D6 最弱）。
- 删除 ICLabel 判定为低脑电概率、高噪声概率的 ICA 成分（眼动、肌电、心电、工频、通道噪声）。
- 通过 45 Hz 低通抑制残余高频成分。
- 去噪强度适中：仅删除脑电概率低于 10% 且噪声概率高于 50% 的成分。删除阈值可通过 `DenoiseConfig.iclabel_min_brain_prob` 和 `iclabel_min_noise_prob` 调整。

## 局限性

- 不适用于双极导联、颅内 EEG 或实时处理。
- 不保留 45 Hz 以上活动。
- 小波去噪在 ICA 之前执行；非线性阈值收缩可能对 ICA 源分离产生有限影响。
- 未提供定量评价指标（SNR、MSE）；效果仅通过可视化对比评估。
- 尽管固定了随机种子，ICA 结果可能因 MNE 版本不同而有差异。

## 安装

```bash
pip install -r requirements.txt
```

## 许可证

MIT
