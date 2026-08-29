"""EDF EEG denoising: wavelet shrinkage, ICA/ICLabel, NPZ, and MNE comparison.

Example
-------
python eeg_denoise.py patient.edf -o patient_denoised.npz

The NPZ stores EEG samples in volts with shape ``(n_channels, n_samples)``.
After saving, an interactive MNE comparison window overlays the first
40 seconds: before denoising in black and after denoising in red.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import threading
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pywt
from mne.preprocessing import ICA
from mne_icalabel import label_components
from tqdm import tqdm


LOGGER = logging.getLogger("eeg_denoise")
MAD_NORMAL_SCALE = 0.6744897501960817
TOTAL_STAGES = 7
DEFAULT_EDF_INPUT = Path(
    r"\\PaiDong\Dataset\Huashan\EEG\20260708\DOCedf"
)


@contextmanager
def _stage(number: int, name: str) -> Iterator[None]:
    """Print start, success/failure, and wall time for one pipeline stage."""

    LOGGER.info("=" * 72)
    LOGGER.info("[%d/%d] 开始：%s", number, TOTAL_STAGES, name)
    started_at = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started_at
        LOGGER.error(
            "[%d/%d] 失败：%s（已运行 %.1f 秒）",
            number,
            TOTAL_STAGES,
            name,
            elapsed,
        )
        raise
    else:
        elapsed = time.perf_counter() - started_at
        LOGGER.info(
            "[%d/%d] 完成：%s（耗时 %.1f 秒）",
            number,
            TOTAL_STAGES,
            name,
            elapsed,
        )


@contextmanager
def _elapsed_progress(description: str) -> Iterator[None]:
    """Show an indeterminate tqdm bar whose counter is elapsed seconds."""

    stop_event = threading.Event()
    progress = tqdm(
        total=None,
        desc=description,
        unit="s",
        dynamic_ncols=True,
        leave=True,
    )

    def update_elapsed_time() -> None:
        while not stop_event.wait(1.0):
            progress.update(1)

    update_thread = threading.Thread(
        target=update_elapsed_time,
        name="eeg-denoise-progress",
        daemon=True,
    )
    update_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        update_thread.join(timeout=2.0)
        progress.close()


@dataclass(frozen=True)
class DenoiseConfig:
    """Fixed processing parameters requested for the denoising pipeline."""

    highpass_hz: float = 0.5
    lowpass_hz: float = 45.0
    wavelet: str = "sym8"
    wavelet_level: int = 6
    wavelet_mode: str = "symmetric"
    threshold_mode: str = "garrote"
    noise_estimator: str = "D1_MAD"
    base_threshold: str = "universal"
    threshold_decay: str = "lambda_j=lambda_1*2**(-(j-1)/2)"
    segment_seconds: float = 8.0
    segment_overlap: float = 0.5
    ica_method: str = "extended_infomax"
    ica_fit_highpass_hz: float = 1.0
    ica_fit_lowpass_hz: float = 100.0
    ica_random_state: int = 97
    iclabel_min_brain_prob: float = 0.10
    iclabel_min_noise_prob: float = 0.50
    plot_duration_seconds: float = 40.0


@dataclass
class ChannelPreparation:
    """Information retained while converting EDF labels to standard EEG names."""

    raw: mne.io.BaseRaw
    rename_map: dict[str, str]
    dropped_non_eeg_channels: list[str]
    dropped_bad_channels: list[str]


@dataclass
class ICAResult:
    """ICA/ICLabel output needed for reconstruction and NPZ metadata."""

    cleaned_raw: mne.io.BaseRaw
    labels: list[str]
    confidence: np.ndarray
    excluded_components: list[int]
    kept_components: list[int]
    n_components: int
    n_iterations: int
    fit_band_hz: tuple[float, float | None]


def _resolve_one_edf(input_path: Path) -> Path:
    """Resolve one EDF path; a directory means its first sorted EDF file."""

    input_path = input_path.expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() != ".edf":
            raise ValueError(f"输入文件不是 EDF：{input_path}")
        return input_path.resolve()

    if input_path.is_dir():
        candidates = sorted(
            (
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() == ".edf"
            ),
            key=lambda path: path.name.lower(),
        )
        if not candidates:
            raise FileNotFoundError(f"目录中没有 EDF 文件：{input_path}")
        selected = candidates[0].resolve()
        LOGGER.info("输入为目录，只处理排序后的第一个 EDF：%s", selected.name)
        return selected

    raise FileNotFoundError(f"找不到输入路径：{input_path}")


def _clean_edf_channel_name(channel_name: str) -> str:
    """Clean common EDF EEG prefixes and reference suffixes."""

    clean_name = re.sub(
        r"^\s*(EEG|POLY)\s*",
        "",
        channel_name,
        flags=re.IGNORECASE,
    )
    clean_name = re.sub(
        r"[\s_-]*(REF|LE|RE|AVG|A1|A2)\s*$",
        "",
        clean_name,
        flags=re.IGNORECASE,
    )
    return clean_name.strip()


def prepare_eeg_channels(raw: mne.io.BaseRaw) -> ChannelPreparation:
    """Match EDF channels to standard_1020 and keep only good scalp EEG."""

    montage = mne.channels.make_standard_montage("standard_1020")
    standard_names = {name.lower(): name for name in montage.ch_names}
    # Legacy 10-20 labels are mapped only when the modern target is not used.
    legacy_aliases = {
        "t3": "T7",
        "t4": "T8",
        "t5": "P7",
        "t6": "P8",
    }

    prepared = raw.copy()
    rename_map: dict[str, str] = {}
    used_names: set[str] = set()

    for old_name in prepared.ch_names:
        clean_name = _clean_edf_channel_name(old_name)
        standard_name = standard_names.get(clean_name.lower())
        if standard_name is None:
            standard_name = legacy_aliases.get(clean_name.lower())
        if standard_name is not None and standard_name not in used_names:
            rename_map[old_name] = standard_name
            used_names.add(standard_name)

    if len(used_names) < 3:
        raise RuntimeError(
            "匹配到的标准 10-20 EEG 通道少于 3 个，ICLabel 无法可靠运行。"
            "如果 EDF 使用 Fp1-F7 等双极导联，必须先转换为有明确电极"
            "位置的单极参考数据，不能直接用于 ICLabel。原始通道为："
            + ", ".join(raw.ch_names)
        )

    prepared.rename_channels(rename_map)
    matched_eeg_names = set(rename_map.values())
    channel_types = {
        name: ("eeg" if name in matched_eeg_names else "misc")
        for name in prepared.ch_names
    }
    prepared.set_channel_types(channel_types, on_unit_change="ignore")

    dropped_non_eeg = [
        name for name in prepared.ch_names if name not in matched_eeg_names
    ]
    dropped_bad = [
        name for name in prepared.info["bads"] if name in matched_eeg_names
    ]

    eeg_picks = mne.pick_types(
        prepared.info,
        meg=False,
        eeg=True,
        exclude="bads",
    )
    if len(eeg_picks) < 3:
        raise RuntimeError(
            "剔除 EDF 中标记的坏通道后，剩余 EEG 通道少于 3 个。"
        )

    prepared.pick(eeg_picks)
    prepared.set_montage(
        montage,
        match_case=False,
        on_missing="raise",
    )

    if dropped_non_eeg:
        LOGGER.info(
            "不参与处理的非标准/非 EEG 通道（%d）：%s",
            len(dropped_non_eeg),
            ", ".join(dropped_non_eeg),
        )
    if dropped_bad:
        LOGGER.warning(
            "剔除 EDF 已标记的坏 EEG 通道（%d）：%s",
            len(dropped_bad),
            ", ".join(dropped_bad),
        )

    return ChannelPreparation(
        raw=prepared,
        rename_map=rename_map,
        dropped_non_eeg_channels=dropped_non_eeg,
        dropped_bad_channels=dropped_bad,
    )


def _validate_sampling_rate(sfreq: float, config: DenoiseConfig) -> int:
    """Validate frequency settings and return the 8-second segment length."""

    nyquist = sfreq / 2.0
    if config.lowpass_hz >= nyquist:
        raise ValueError(
            f"采样率 {sfreq:g} Hz 的 Nyquist 频率仅为 {nyquist:g} Hz，"
            f"无法执行 {config.lowpass_hz:g} Hz 低通。"
        )

    segment_samples = int(round(config.segment_seconds * sfreq))
    wavelet = pywt.Wavelet(config.wavelet)
    max_level = pywt.dwt_max_level(segment_samples, wavelet.dec_len)
    if max_level < config.wavelet_level:
        minimum_samples = (wavelet.dec_len - 1) * (2**config.wavelet_level)
        minimum_sfreq = minimum_samples / config.segment_seconds
        raise ValueError(
            f"8 秒分段只有 {segment_samples} 点，{config.wavelet} 的有效最大"
            f"分解层数为 {max_level}，不足 level={config.wavelet_level}。"
            f"该设置约需采样率 >= {minimum_sfreq:g} Hz。"
        )
    return segment_samples


def _mad_noise_sigma(detail_d1: np.ndarray) -> float:
    """Estimate Gaussian noise sigma from the level-1 detail MAD."""

    center = np.median(detail_d1)
    mad = np.median(np.abs(detail_d1 - center))
    return float(mad / MAD_NORMAL_SCALE)


def _denoise_wavelet_segment(
    segment: np.ndarray,
    config: DenoiseConfig,
) -> np.ndarray:
    """Denoise one fixed-length segment and return the identical length."""

    coeffs = pywt.wavedec(
        segment,
        wavelet=config.wavelet,
        mode=config.wavelet_mode,
        level=config.wavelet_level,
    )
    sigma = _mad_noise_sigma(coeffs[-1])
    universal_threshold = sigma * math.sqrt(2.0 * math.log(segment.size))

    thresholded: list[np.ndarray] = [coeffs[0]]
    # wavedec order is [A6, D6, D5, ..., D1]. D1 gets the full universal
    # threshold; thresholds are progressively weakened toward D6 to protect
    # lower-frequency EEG activity.
    for coefficient_index, detail in enumerate(coeffs[1:], start=1):
        detail_level = config.wavelet_level - coefficient_index + 1
        threshold = universal_threshold * 2.0 ** (
            -(detail_level - 1) / 2.0
        )
        if threshold > 0.0:
            detail = pywt.threshold(
                detail,
                value=threshold,
                mode=config.threshold_mode,
            )
        thresholded.append(detail)

    reconstructed = pywt.waverec(
        thresholded,
        wavelet=config.wavelet,
        mode=config.wavelet_mode,
    )
    return np.asarray(reconstructed[: segment.size], dtype=np.float64)


def wavelet_overlap_denoise(
    signal: np.ndarray,
    sfreq: float,
    config: DenoiseConfig,
) -> np.ndarray:
    """Denoise one EEG channel with symmetric padding and Hann overlap-add."""

    if signal.ndim != 1:
        raise ValueError("wavelet_overlap_denoise 只接受一维单通道信号。")
    if signal.size == 0:
        return signal.copy()
    if not np.all(np.isfinite(signal)):
        raise ValueError("EEG 中存在 NaN 或 Inf，无法进行小波去噪。")

    segment_samples = int(round(config.segment_seconds * sfreq))
    hop_samples = int(
        round(segment_samples * (1.0 - config.segment_overlap))
    )
    if segment_samples < 2 or hop_samples < 1:
        raise ValueError("分段长度或重叠比例无效。")

    left_pad = segment_samples - hop_samples
    right_pad = segment_samples - hop_samples
    padded_length = signal.size + left_pad + right_pad
    remainder = (padded_length - segment_samples) % hop_samples
    if remainder:
        right_pad += hop_samples - remainder

    padded = np.pad(signal, (left_pad, right_pad), mode="symmetric")
    window = np.hanning(segment_samples)
    if not np.any(window > 0):
        raise RuntimeError("无法创建有效的 Hann 重叠窗。")

    accumulated = np.zeros_like(padded, dtype=np.float64)
    weights = np.zeros_like(padded, dtype=np.float64)
    final_start = padded.size - segment_samples

    for start in range(0, final_start + 1, hop_samples):
        stop = start + segment_samples
        cleaned_segment = _denoise_wavelet_segment(
            padded[start:stop],
            config,
        )
        accumulated[start:stop] += cleaned_segment * window
        weights[start:stop] += window

    target = slice(left_pad, left_pad + signal.size)
    target_weights = weights[target]
    if np.any(target_weights <= np.finfo(np.float64).eps):
        raise RuntimeError("重叠相加时出现未覆盖的采样点。")

    output = accumulated[target] / target_weights
    if output.shape != signal.shape:
        raise RuntimeError("小波重构后的长度与输入不一致。")
    return output


def _apply_highpass(
    raw: mne.io.BaseRaw,
    config: DenoiseConfig,
) -> mne.io.BaseRaw:
    """Apply the 0.5-Hz high-pass filter in place."""

    # The untouched signal needed for plotting is copied before this function.
    # Reusing this Raw avoids another full-recording allocation for long EDFs.
    filtered = raw
    filtered.filter(
        l_freq=config.highpass_hz,
        h_freq=None,
        picks=None,
        method="fir",
        phase="zero-double",
        fir_design="firwin",
        verbose="ERROR",
    )
    return filtered


def _apply_wavelet(
    raw: mne.io.BaseRaw,
    config: DenoiseConfig,
) -> mne.io.BaseRaw:
    """Apply channel-wise segmented wavelet denoising in place."""

    sfreq = float(raw.info["sfreq"])
    wavelet_raw = raw

    def denoise_one_channel(channel_data: np.ndarray) -> np.ndarray:
        return wavelet_overlap_denoise(channel_data, sfreq, config)

    LOGGER.info(
        "开始小波去噪：%d 通道，%g 秒分段，%g%% 重叠",
        len(wavelet_raw.ch_names),
        config.segment_seconds,
        config.segment_overlap * 100.0,
    )
    with tqdm(
        total=len(wavelet_raw.ch_names),
        desc="[4/7] 小波去噪",
        unit="通道",
        dynamic_ncols=True,
        leave=True,
    ) as progress:

        def denoise_and_report(channel_data: np.ndarray) -> np.ndarray:
            cleaned_channel = denoise_one_channel(channel_data)
            progress.update(1)
            return cleaned_channel

        wavelet_raw.apply_function(
            denoise_and_report,
            picks="eeg",
            channel_wise=True,
            n_jobs=1,
            verbose="ERROR",
        )
    return wavelet_raw


def _run_ica_iclabel(
    wavelet_raw: mne.io.BaseRaw,
    config: DenoiseConfig,
    ica_decim: int,
) -> ICAResult:
    """Fit extended Infomax; drop components whose brain probability is
    below ``iclabel_min_brain_prob`` and noise probability exceeds
    ``iclabel_min_noise_prob``, keeping all other components."""

    # ICLabel was trained with a common-average reference.
    # The wavelet result is no longer needed separately. Reuse its storage to
    # keep peak RAM bounded for long clinical recordings.
    target_raw = wavelet_raw
    target_raw.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose="ERROR",
    )

    fit_raw = target_raw.copy()
    nyquist = float(fit_raw.info["sfreq"]) / 2.0
    fit_high = config.ica_fit_highpass_hz
    fit_low: float | None
    if config.ica_fit_lowpass_hz < nyquist:
        fit_low = config.ica_fit_lowpass_hz
    else:
        fit_low = None
        warnings.warn(
            f"采样率限制使 ICLabel 拟合支路无法达到 1–100 Hz；"
            f"实际使用 {fit_high:g}–{nyquist:g} Hz。分类性能可能降低。",
            RuntimeWarning,
            stacklevel=2,
        )

    fit_raw.filter(
        l_freq=fit_high,
        h_freq=fit_low,
        picks=None,
        method="fir",
        phase="zero-double",
        fir_design="firwin",
        verbose="ERROR",
    )

    effective_ica_samples = math.ceil(fit_raw.n_times / ica_decim)
    if effective_ica_samples < 5 * len(fit_raw.ch_names):
        raise RuntimeError(
            "应用 --ica-decim 后的有效采样点相对 EEG 通道数过少，"
            "无法稳定拟合 ICA。"
        )

    ica = ICA(
        n_components=None,
        method="infomax",
        fit_params={"extended": True},
        max_iter="auto",
        random_state=config.ica_random_state,
    )
    LOGGER.info(
        "[5/7][ICA 1/4] 拟合数据准备完成：%d 通道，%d 采样点，decim=%d",
        len(fit_raw.ch_names),
        fit_raw.n_times,
        ica_decim,
    )
    LOGGER.info(
        "[5/7][ICA 2/4] 开始 extended Infomax ICA；"
        "算法不提供逐迭代回调，进度条显示已运行秒数"
    )
    with _elapsed_progress("[5/7] ICA 运行中"):
        ica.fit(
            fit_raw,
            picks="eeg",
            decim=ica_decim,
            reject_by_annotation=True,
            verbose="ERROR",
        )
    LOGGER.info(
        "[5/7][ICA 2/4] ICA 拟合完成：%d 个成分，真实迭代 %d 次",
        ica.n_components_,
        ica.n_iter_,
    )

    LOGGER.info("[5/7][ICA 3/4] 使用 ICLabel 自动分类 ICA 成分……")
    with _elapsed_progress("[5/7] ICLabel 推理中"):
        label_result = label_components(fit_raw, ica, method="iclabel")
    labels = [str(label) for label in label_result["labels"]]
    confidence = np.asarray(
        label_result["y_pred_proba"],
        dtype=np.float32,
    )
    if len(labels) != ica.n_components_:
        raise RuntimeError("ICLabel 返回的标签数与 ICA 成分数不一致。")

    class_names = [
        str(name)
        for name in label_result.get("y_pred_proba_class_names", [])
    ]
    brain_index = class_names.index("brain") if class_names else 0
    brain_prob = confidence[:, brain_index]
    # Noise probability = sum of probabilities over every class other than
    # brain (muscle, eye blink, heart, line noise, channel noise, other).
    # ICLabel probability rows sum to one, so this equals 1 - brain_prob,
    # but summing the non-brain columns expresses the rule explicitly.
    noise_prob = confidence.sum(axis=1) - brain_prob

    excluded = [
        index
        for index in range(len(labels))
        if brain_prob[index] < config.iclabel_min_brain_prob
        and noise_prob[index] > config.iclabel_min_noise_prob
    ]
    excluded_set = set(excluded)
    kept = [
        index for index in range(len(labels)) if index not in excluded_set
    ]
    if not kept:
        raise RuntimeError(
            "ICLabel 规则将所有成分判定为噪声并剔除。为避免输出近乎空白"
            "的脑电，程序已安全中止；请检查通道名、蒙太奇、参考方式和"
            "原始数据质量。"
        )

    label_counts = {
        label: labels.count(label) for label in sorted(set(labels))
    }
    LOGGER.info("ICLabel 分类统计：%s", label_counts)
    LOGGER.info("保留 %d 个成分：%s", len(kept), kept)
    LOGGER.info(
        "剔除 %d 个成分（脑电概率 < %g%% 且噪声概率 > %g%%）：%s",
        len(excluded),
        config.iclabel_min_brain_prob * 100.0,
        config.iclabel_min_noise_prob * 100.0,
        excluded,
    )

    LOGGER.info(
        "[5/7][ICA 4/4] 重构 EEG 并执行最终 %g Hz 低通……",
        config.lowpass_hz,
    )
    del fit_raw
    cleaned = target_raw
    ica.apply(cleaned, exclude=excluded, verbose="ERROR")
    cleaned.filter(
        l_freq=None,
        h_freq=config.lowpass_hz,
        picks=None,
        method="fir",
        phase="zero-double",
        fir_design="firwin",
        verbose="ERROR",
    )

    if not np.all(np.isfinite(cleaned.get_data())):
        raise RuntimeError("ICA 重构或最终低通后出现 NaN/Inf。")

    return ICAResult(
        cleaned_raw=cleaned,
        labels=labels,
        confidence=confidence,
        excluded_components=excluded,
        kept_components=kept,
        n_components=int(ica.n_components_),
        n_iterations=int(ica.n_iter_),
        fit_band_hz=(fit_high, fit_low),
    )


def _average_reference_for_plot(
    raw: mne.io.BaseRaw,
    duration_seconds: float,
) -> mne.io.BaseRaw:
    """Copy only the visible window and apply the final reference for plotting."""

    duration_samples = int(round(duration_seconds * raw.info["sfreq"]))
    n_samples = min(duration_samples, raw.n_times)
    plot_raw = mne.io.RawArray(
        raw.get_data(start=0, stop=n_samples),
        raw.info.copy(),
        first_samp=raw.first_samp,
        copy="auto",
        verbose="ERROR",
    )
    plot_raw.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose="ERROR",
    )
    return plot_raw


def show_interactive_comparison(
    before_raw: mne.io.BaseRaw,
    after_raw: mne.io.BaseRaw,
    config: DenoiseConfig,
) -> None:
    """Show a 40-second interactive MNE overlay: before black, after red."""

    if before_raw.ch_names != after_raw.ch_names:
        raise ValueError("绘图前后的 EEG 通道顺序不一致。")

    sfreq = float(after_raw.info["sfreq"])
    duration_samples = int(round(config.plot_duration_seconds * sfreq))
    n_samples = min(
        duration_samples,
        before_raw.n_times,
        after_raw.n_times,
    )
    actual_duration = n_samples / sfreq

    # EvokedArray is used only as an MNE visualization container for the
    # continuous 40-second window. It does not average or alter the data.
    before_evoked = mne.EvokedArray(
        before_raw.get_data(start=0, stop=n_samples),
        before_raw.info.copy(),
        tmin=0.0,
        comment="Before denoising",
        nave=1,
        verbose="ERROR",
    )
    after_evoked = mne.EvokedArray(
        after_raw.get_data(start=0, stop=n_samples),
        after_raw.info.copy(),
        tmin=0.0,
        comment="After denoising",
        nave=1,
        verbose="ERROR",
    )

    LOGGER.info(
        "打开 MNE 交互式对比窗口：前 %.1f 秒，黑色为去噪前，红色为去噪后",
        actual_duration,
    )
    mne.viz.plot_compare_evokeds(
        {
            "Before denoising": before_evoked,
            "After denoising": after_evoked,
        },
        picks=after_raw.ch_names,
        colors={
            "Before denoising": "black",
            "After denoising": "red",
        },
        styles={
            "Before denoising": {"linewidth": 0.7, "alpha": 0.8},
            "After denoising": {"linewidth": 0.8, "alpha": 0.85},
        },
        ci=False,
        truncate_xaxis=False,
        legend=True,
        axes="topo",
        title=(
            f"EEG denoising comparison ({actual_duration:.1f} s): "
            "before=black, after=red"
        ),
        show=True,
        combine=None,
        time_unit="s",
    )

    # MNE uses Matplotlib for this comparison figure. Explicitly block here so
    # a command-line run keeps the interactive zoom/pan window open.
    import matplotlib.pyplot as plt

    plt.show(block=True)


def _wavelet_detail_bands(sfreq: float, level: int) -> np.ndarray:
    """Return idealized dyadic [low, high] frequency bands for D1..Dlevel."""

    return np.asarray(
        [
            (sfreq / (2 ** (detail_level + 1)), sfreq / (2**detail_level))
            for detail_level in range(1, level + 1)
        ],
        dtype=np.float64,
    )


def _measurement_date(raw: mne.io.BaseRaw) -> str:
    """Convert an optional MNE measurement date to an ISO string."""

    measurement_date = raw.info.get("meas_date")
    if measurement_date is None:
        return ""
    if hasattr(measurement_date, "isoformat"):
        return str(measurement_date.isoformat())
    return str(measurement_date)


def save_npz(
    output_path: Path,
    source_edf: Path,
    preparation: ChannelPreparation,
    result: ICAResult,
    config: DenoiseConfig,
    ica_decim: int,
) -> None:
    """Save denoised EEG and reproducibility metadata without object arrays."""

    raw = result.cleaned_raw
    sfreq = float(raw.info["sfreq"])
    processing_parameters: dict[str, Any] = asdict(config)
    processing_parameters.update(
        {
            "ica_decim": ica_decim,
            "iclabel_policy": (
                "remove_components_with_brain_prob_lt_0.10_and_noise_prob_gt_0.50"
            ),
            "output_reference": "common_average",
            "output_unit": "V",
            "pipeline_order": [
                "read_edf",
                "standardize_and_pick_eeg",
                "highpass_0.5Hz",
                "segmented_wavelet_denoise",
                "common_average_reference",
                "fit_extended_infomax_on_1_to_100Hz_branch",
                "remove_ICLabel_components_with_brain_prob_lt_0.10_and_noise_prob_gt_0.50",
                "lowpass_45Hz",
            ],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        eeg=np.asarray(raw.get_data(), dtype=np.float32),
        sfreq=np.asarray(sfreq, dtype=np.float64),
        ch_names=np.asarray(raw.ch_names, dtype=np.str_),
        channel_types=np.asarray(raw.get_channel_types(), dtype=np.str_),
        unit=np.asarray("V", dtype=np.str_),
        source_edf=np.asarray(str(source_edf), dtype=np.str_),
        measurement_date=np.asarray(_measurement_date(raw), dtype=np.str_),
        first_samp=np.asarray(raw.first_samp, dtype=np.int64),
        duration_seconds=np.asarray(
            raw.n_times / sfreq,
            dtype=np.float64,
        ),
        ic_labels=np.asarray(result.labels, dtype=np.str_),
        ic_label_confidence=np.asarray(
            result.confidence,
            dtype=np.float32,
        ),
        excluded_ica_components=np.asarray(
            result.excluded_components,
            dtype=np.int64,
        ),
        kept_ica_components=np.asarray(
            result.kept_components,
            dtype=np.int64,
        ),
        n_ica_components=np.asarray(result.n_components, dtype=np.int64),
        ica_n_iterations=np.asarray(result.n_iterations, dtype=np.int64),
        ica_fit_band_hz=np.asarray(
            [
                result.fit_band_hz[0],
                np.nan
                if result.fit_band_hz[1] is None
                else result.fit_band_hz[1],
            ],
            dtype=np.float64,
        ),
        wavelet_detail_levels=np.arange(
            1,
            config.wavelet_level + 1,
            dtype=np.int64,
        ),
        wavelet_detail_bands_hz=_wavelet_detail_bands(
            sfreq,
            config.wavelet_level,
        ),
        renamed_channels_json=np.asarray(
            json.dumps(
                preparation.rename_map,
                ensure_ascii=False,
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
        dropped_non_eeg_channels=np.asarray(
            preparation.dropped_non_eeg_channels,
            dtype=np.str_,
        ),
        dropped_bad_channels=np.asarray(
            preparation.dropped_bad_channels,
            dtype=np.str_,
        ),
        processing_parameters_json=np.asarray(
            json.dumps(
                processing_parameters,
                ensure_ascii=False,
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
        interactive_comparison=np.asarray(
            "MNE plot_compare_evokeds; first 40 s; before black; after red",
            dtype=np.str_,
        ),
    )


def process_edf(
    input_path: Path,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    ica_decim: int = 1,
    show_plot: bool = True,
    config: DenoiseConfig | None = None,
) -> Path:
    """Run the complete single-EDF denoising pipeline."""

    if config is None:
        config = DenoiseConfig()
    if ica_decim < 1:
        raise ValueError("ica_decim 必须是大于等于 1 的整数。")

    edf_path = _resolve_one_edf(input_path)
    if output_path is None:
        output_path = (
            Path.cwd() / f"{edf_path.stem}_denoised.npz"
        ).resolve()
    else:
        output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"输出已存在：{output_path}。如需覆盖，请添加 --overwrite。"
        )

    with _stage(1, "读取单个患者 EDF"):
        LOGGER.info("EDF 路径：%s", edf_path)
        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose="ERROR",
        )
        LOGGER.info(
            "EDF 原始信息：%d 通道，采样率 %g Hz，%d 采样点",
            len(raw.ch_names),
            float(raw.info["sfreq"]),
            raw.n_times,
        )

    with _stage(2, "通道标准化、10-20 蒙太奇和参数校验"):
        preparation = prepare_eeg_channels(raw)
        prepared_raw = preparation.raw
        sfreq = float(prepared_raw.info["sfreq"])
        _validate_sampling_rate(sfreq, config)
        LOGGER.info(
            "患者记录：%d 个有效 EEG 通道，采样率 %g Hz，时长 %.1f 秒",
            len(prepared_raw.ch_names),
            sfreq,
            prepared_raw.n_times / sfreq,
        )
        bands = _wavelet_detail_bands(sfreq, config.wavelet_level)
        LOGGER.info(
            "当前采样率下 D1–D6 理论频带（Hz）：%s",
            np.array2string(bands, precision=3),
        )
        if not math.isclose(sfreq, 256.0, rel_tol=0.0, abs_tol=1e-6):
            warnings.warn(
                "当前采样率不是 256 Hz，因此 D6/D5/D4 不会恰好对应"
                " 2–4/4–8/8–16 Hz；实际频带已在终端打印并写入 NPZ。",
                RuntimeWarning,
                stacklevel=2,
            )
        before_plot_raw = _average_reference_for_plot(
            prepared_raw,
            config.plot_duration_seconds,
        )
        del raw

    with _stage(3, "0.5 Hz 高通"):
        filtered_raw = _apply_highpass(prepared_raw, config)

    with _stage(4, "8 秒、50% 重叠的 sym8 level-6 小波去噪"):
        wavelet_raw = _apply_wavelet(filtered_raw, config)

    with _stage(5, "extended Infomax ICA、ICLabel 和 45 Hz 最终低通"):
        ica_result = _run_ica_iclabel(
            wavelet_raw,
            config,
            ica_decim=ica_decim,
        )

    with _stage(6, "保存压缩 NPZ 和处理元数据"):
        LOGGER.info("NPZ 路径：%s", output_path)
        save_npz(
            output_path,
            edf_path,
            preparation,
            ica_result,
            config,
            ica_decim,
        )

    if show_plot:
        with _stage(7, "打开 MNE 40 秒黑/红交互式叠加图"):
            show_interactive_comparison(
                before_plot_raw,
                ica_result.cleaned_raw,
                config,
            )
    else:
        LOGGER.info("=" * 72)
        LOGGER.info("[7/7] 已跳过交互式绘图（--no-plot）")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "读取一个 EDF 患者脑电，依次执行 0.5 Hz 高通、"
            "分段小波去噪、extended Infomax ICA + ICLabel、45 Hz 低通，"
            "输出 NPZ，并用 MNE 打开 40 秒去噪前后交互叠加图。"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_EDF_INPUT,
        help=(
            "一个 .edf 文件；也可给目录，此时只处理排序后的第一个 EDF。"
            "省略时使用 read.py 中的 DOCedf 目录"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 NPZ 路径（默认：当前目录/<输入名>_denoised.npz）",
    )
    parser.add_argument(
        "--ica-decim",
        type=int,
        default=1,
        help="ICA 拟合降采样步长；默认 1 表示使用全部采样点",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有 NPZ",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只保存 NPZ，不打开 MNE 交互式对比窗口",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="日志级别（默认 INFO）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        output_path = process_edf(
            args.input,
            args.output,
            overwrite=args.overwrite,
            ica_decim=args.ica_decim,
            show_plot=not args.no_plot,
        )
    except Exception as error:
        LOGGER.error("%s", error)
        if args.log_level == "DEBUG":
            LOGGER.exception("完整异常信息")
        return 1

    print(f"完成。\nNPZ: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
