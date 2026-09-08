# EEG Denoising Based on Wavelet Transform and ICA

**English** | [中文](README.zh-CN.md)

## Overview

`eeg_denoise.py` performs offline denoising of a single scalp-EEG EDF recording. The pipeline combines wavelet shrinkage with ICA/ICLabel artifact removal and outputs a compressed NPZ file.

## Technical Pipeline

1. **Read EDF** — Load and preload one EDF file with MNE.
2. **Channel standardization** — Clean EDF channel labels (strip `EEG`/`POLY` prefixes and reference suffixes), map to `standard_1020` montage, apply legacy aliases (`T3→T7`, `T4→T8`, `T5→P7`, `T6→P8`), and remove non-EEG and bad channels.
3. **High-pass filter** — 0.5 Hz FIR zero-phase double-pass high-pass.
4. **Segmented wavelet denoising** — Per-channel, 8-second segments with 50% overlap, `sym8` wavelet, 6-level decomposition, symmetric padding, Hann window overlap-add.
5. **Common average reference** — Applied before ICA to match ICLabel training conditions.
6. **ICA + ICLabel** — Extended Infomax ICA fitted on a 1–100 Hz copy; ICLabel classifies components; components with brain probability < 10% and noise probability > 50% are removed.
7. **Low-pass filter** — 45 Hz FIR zero-phase double-pass low-pass on the reconstructed EEG.
8. **Output** — Save compressed NPZ with denoised EEG and metadata; optionally display a 40-second before/after comparison plot.

## Wavelet Parameters

| Parameter | Value |
|---|---|
| Wavelet | `sym8` |
| Decomposition level | 6 |
| Boundary mode | `symmetric` |
| Noise estimator | D1-level MAD |
| Base threshold | `sigma * sqrt(2 * ln(N))` (universal) |
| Level decay | `lambda_j = lambda_1 * 2^(-(j-1)/2)` |
| Threshold mode | `garrote` |
| Segment length | 8 seconds |
| Overlap | 50% |
| Window | Hann |

Noise sigma: `sigma = median(|D1 - median(D1)|) / 0.6744897501960817`

Level-dependent threshold ratios at 256 Hz:

| Level | Relative threshold | Frequency band |
|---|---|---|
| D1 | 1.000 | 64–128 Hz |
| D2 | 0.707 | 32–64 Hz |
| D3 | 0.500 | 16–32 Hz |
| D4 | 0.354 | 8–16 Hz |
| D5 | 0.250 | 4–8 Hz |
| D6 | 0.177 | 2–4 Hz |

## ICA and ICLabel Parameters

| Parameter | Value |
|---|---|
| ICA method | Extended Infomax |
| n_components | Determined by MNE from data rank |
| max_iter | `auto` (MNE-managed) |
| Random seed | 97 |
| ICA fit band | 1–100 Hz |
| Reference | Common average |
| ICLabel removal rule | Remove components with brain probability < 10% and noise probability > 50% |
| Final low-pass | 45 Hz |

## Usage

```bash
python eeg_denoise.py <input_edf> [-o output.npz] [options]
```

### Arguments

| Argument | Description |
|---|---|
| `input` | Path to a `.edf` file, or a directory (processes the first sorted EDF) |
| `-o, --output` | Output NPZ path (default: `<input_name>_denoised.npz`) |
| `--ica-decim` | ICA fit decimation step (default: 1, use all samples) |
| `--overwrite` | Overwrite existing output file |
| `--no-plot` | Skip the interactive comparison plot |
| `--log-level` | Log level: DEBUG, INFO (default), WARNING, ERROR |

### Example

```bash
python eeg_denoise.py patient.edf -o patient_denoised.npz
```

## Input Requirements

- One `.edf` file or a directory containing EDF files.
- Channel names must map to `standard_1020`. `EEG`/`POLY` prefixes and common reference suffixes (`REF`, `LE`, `RE`, `AVG`, `A1`, `A2`) are cleaned automatically.
- At least 3 valid EEG channels must remain after cleaning and bad-channel exclusion.
- Sampling rate must exceed 120 Hz (45 Hz low-pass requires > 90 Hz; 8-second `sym8 level=6` requires ~960 samples). 256 Hz is recommended.
- The script does not resample. D1–D6 frequency bands shift with sampling rate.
- EEG must not contain NaN or Inf.
- Bipolar derivations (e.g., `Fp1-F7`) are not supported.

## Output Format

NPZ file containing:

| Key | Type | Description |
|---|---|---|
| `eeg` | `float32` `[channels, samples]` | Denoised EEG in volts |
| `sfreq` | `float64` | Sampling rate (Hz) |
| `ch_names` | `str_` | Standard 10-20 channel names |
| `channel_types` | `str_` | Channel types |
| `unit` | `str_` | Unit ("V") |
| `source_edf` | `str_` | Source EDF path |
| `measurement_date` | `str_` | Measurement date (ISO) |
| `duration_seconds` | `float64` | Recording duration |
| `ic_labels` | `str_` | ICLabel class labels per component |
| `ic_label_confidence` | `float32` | ICLabel probability matrix |
| `excluded_ica_components` | `int64` | Removed component indices |
| `kept_ica_components` | `int64` | Retained component indices |
| `n_ica_components` | `int64` | Number of ICA components |
| `ica_n_iterations` | `int64` | ICA convergence iterations |
| `ica_fit_band_hz` | `float64` | ICA fitting frequency band |
| `wavelet_detail_bands_hz` | `float64` | Theoretical D1–D6 frequency bands |
| `renamed_channels_json` | `str_` | Channel rename mapping (JSON) |
| `dropped_non_eeg_channels` | `str_` | Removed non-EEG channel names |
| `dropped_bad_channels` | `str_` | Removed bad channel names |
| `processing_parameters_json` | `str_` | Full processing parameters (JSON) |

## Effects

- Reduces baseline drift via 0.5 Hz high-pass.
- Reduces high-frequency random noise via level-dependent wavelet shrinkage (strongest at D1, weakest at D6).
- Removes ICA components classified by ICLabel as low-brain-probability and high-noise-probability artifacts (eye blinks, muscle, heart, line noise, channel noise).
- Suppresses residual high-frequency content via 45 Hz low-pass.
- Denoising intensity is moderate: only components with brain probability below 10% and noise probability above 50% are removed. The removal threshold can be adjusted via `DenoiseConfig.iclabel_min_brain_prob` and `iclabel_min_noise_prob`.

## Limitations

- Not suitable for bipolar derivations, intracranial EEG, or real-time processing.
- Does not preserve activity above 45 Hz.
- Wavelet denoising is applied before ICA; the non-linear thresholding may affect ICA source separation to a limited degree.
- No quantitative evaluation metrics (SNR, MSE) are provided; quality is assessed via visual comparison only.
- ICA results may vary across MNE versions despite the fixed random seed.

## Installation

```bash
pip install -r requirements.txt
```

## License

MIT
