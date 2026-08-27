import os
import sys
import json
import argparse
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import yaml


# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tools.data_loader import import_database
from tools.accessions_db import AccessionDatabase
from tools.recording_analysis import analyse_multi_channel_signal
from datasets.preprocesing import filters

try:
    config = yaml.safe_load(open('config.yml'))
except Exception:
    config = {}

CLAMP_FACTOR = config.get('clamp_factor', 3.0)
SIDES = ('L', 'R')

# ==============================================================================
# SIGNAL FILTERING CONFIGURATION (datasets/preprocesing/filters.py)
# Modify these values directly in code or use the --denoise CLI flag to enable.
# ==============================================================================
FILTER_CONFIG = {
    "enabled": True,          # Set to True to apply denoise filter by default
    "kernel": 7,              # Median filter & spike removal kernel size
    "passes": 3,              # Spike removal passes (thresholds 5.0, 4.0, 3.0)
    "clamp_lo": -25.0,        # Lower clamp bound
    "clamp_hi": 25.0,        # Upper clamp bound
}

# Override from config.yml if 'filter' block is present
if "filter" in config and isinstance(config["filter"], dict):
    FILTER_CONFIG.update(config["filter"])


def apply_signal_filter(series: list[float], enabled: bool = False, cfg: dict = None) -> list[float]:
    """Applies median filtering, spike removal, and clamping from datasets.preprocesing.filters."""
    config_to_use = cfg or FILTER_CONFIG
    is_enabled = enabled or config_to_use.get("enabled", False)
    if not is_enabled:
        return list(series)
    
    kernel = config_to_use.get("kernel", 7)
    passes = config_to_use.get("passes", 3)
    lo = config_to_use.get("clamp_lo", -25.0)
    hi = config_to_use.get("clamp_hi", 25.0)

    s = filters.clamp(series, lo=lo, hi=hi)
    s = filters.median_filter(s, kernel_size=kernel)
    for th in [5.0, 4.0, 3.0][:passes]:
        s = filters.remove_spikes(s, threshold_mult=th, kernel=kernel)
    return s



def attr_cycle_selector(info_container: dict, attr: str, region_start: int, region_end: int) -> dict:
    out_dict = {}
    for side in SIDES:
        attr_name = f'{attr}_dict'
        out_dict[side] = info_container[attr_name][side][region_start:region_end]
    out_dict |= {
        'time_list': info_container['time_list'][region_start:region_end],
        'target_list': info_container['target_list'][region_start:region_end]
    }
    return out_dict


def cycle_transformer(
    cycle_dict: dict[str, list[float]],
    centre_measure: str = 'median',
    do_clamp: bool = True,
    cycle_stat: dict = None,
    clamps_factor: float = CLAMP_FACTOR
) -> dict[str, list[float]]:
    new_cycle_container = {}
    
    if cycle_stat is None:
        cycle_stat = analyse_multi_channel_signal(cycle_dict, channel_names=SIDES, do_state_analysis=True)
    
    for side in SIDES:
        series = cycle_dict[side]
        stats = cycle_stat[side]
        low_stats = stats['low_state_statistics']
        high_stats = stats['high_state_statistics']
        
        low_amp, high_amp = low_stats[centre_measure], high_stats[centre_measure]
        baseline = (low_amp + high_amp) / 2

        if do_clamp:
            low_bound = low_stats['mean'] - low_stats['std'] * clamps_factor
            high_bound = high_stats['mean'] + high_stats['std'] * clamps_factor
            series = [min(high_bound, max(low_bound, val)) for val in series]
        
        new_cycle_container[side] = series
    
    return new_cycle_container


def find_glitches(seg: np.ndarray, velocity_thresh: float) -> List[Dict[str, float]]:
    if len(seg) < 3:
        return []
    glitches = []
    diff = np.diff(seg)
    net = seg[-1] - seg[0]
    if abs(net) < 1e-6:
        return []
    trend = 1 if net > 0 else -1
    for i in range(1, len(diff)):
        if trend > 0 and diff[i] < -velocity_thresh:
            start = i
            j = i
            while j < len(diff) and diff[j] < 0:
                j += 1
            if j > start + 1:
                glitches.append({
                    'direction': 'down',
                    'magnitude': float(seg[start] - seg[j]),
                    'duration': j - start + 1
                })
                i = j
        elif trend < 0 and diff[i] > velocity_thresh:
            start = i
            j = i
            while j < len(diff) and diff[j] > 0:
                j += 1
            if j > start + 1:
                glitches.append({
                    'direction': 'up',
                    'magnitude': float(seg[j] - seg[start]),
                    'duration': j - start + 1
                })
                i = j
    return glitches


def find_episodes(
    signal: np.ndarray,
    low_amplitude: float = -15.0,
    high_amplitude: float = 15.0,
    plateau_tolerance: float = 0.1,
    settle_samples: int = 3,
) -> List[Dict[str, Any]]:
    n = len(signal)
    if n < 10:
        return []
    low_thresh = low_amplitude * (1 - plateau_tolerance)
    high_thresh = high_amplitude * (1 - plateau_tolerance)
    signs = np.sign(signal)
    zero_crossings = np.where(np.diff(signs) != 0)[0]
    episodes = []
    for zc in zero_crossings:
        if zc >= n - 1:
            continue
        if signal[zc + 1] > signal[zc]:
            direction = "Rise"
            start_idx = zc
            while start_idx > 0 and signal[start_idx] > low_thresh:
                start_idx -= 1
            if start_idx > 0 and signal[start_idx] < low_thresh:
                start_idx += 1
            else:
                start_idx = 0
            end_idx = zc
            while end_idx < n - settle_samples:
                if all(signal[end_idx:end_idx + settle_samples] > high_thresh):
                    break
                end_idx += 1
            else:
                end_idx = n - 1
            if episodes and start_idx <= episodes[-1]['end_idx']:
                continue
            episodes.append({
                'direction': direction,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'signal_slice': signal[start_idx:end_idx + 1]
            })
        else:
            direction = "Fall"
            start_idx = zc
            while start_idx > 0 and signal[start_idx] < high_thresh:
                start_idx -= 1
            if start_idx > 0 and signal[start_idx] > high_thresh:
                start_idx += 1
            else:
                start_idx = 0
            end_idx = zc
            while end_idx < n - settle_samples:
                if all(signal[end_idx:end_idx + settle_samples] < low_thresh):
                    break
                end_idx += 1
            else:
                end_idx = n - 1
            if episodes and start_idx <= episodes[-1]['end_idx']:
                continue
            episodes.append({
                'direction': direction,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'signal_slice': signal[start_idx:end_idx + 1]
            })
    return episodes


def decompose_episode(
    episode_slice: np.ndarray,
    low_amplitude: float = -15.0,
    high_amplitude: float = 15.0,
    velocity_thresh: float = 0.5,
    min_plateau_samples: int = 2,
) -> Dict[str, Any]:
    n = len(episode_slice)
    if n < 3:
        return {
            'direction': None,
            'average_vel': None,
            'average_acc': None,
            'components': []
        }
    diff = np.diff(episode_slice)
    vel = np.zeros(n)
    vel[1:] = diff
    diff_acc = np.diff(vel)
    acc = np.zeros(n)
    acc[1:] = diff_acc
    # TODO: Check if this threshold is correct, maybe it should be a parameter
    # print("Velocity:", vel)
    # print("Acceleration:", acc)
    # print("Velocity threshold:", velocity_thresh)
    is_moving = np.abs(vel) > velocity_thresh
    components = []
    i = 0
    while i < n:
        if is_moving[i]:
            j = i
            while j < n and is_moving[j]:
                j += 1
            if j - i >= 1:
                seg = episode_slice[i:j]
                zero_cross_idx = None
                for k, val in enumerate(seg):
                    if np.abs(val) < 0.1:
                        zero_cross_idx = i + k
                        break
                if zero_cross_idx is None and len(seg) > 1:
                    for k in range(len(seg) - 1):
                        if seg[k] * seg[k+1] < 0:
                            frac = -seg[k] / (seg[k+1] - seg[k])
                            zero_cross_idx = float(i + k + frac)
                            break
                components.append({
                    'type': 'MicroJump',
                    'start_idx': i,
                    'end_idx': j - 1,
                    'duration': j - i,
                    'amplitude_delta': float(abs(seg[-1] - seg[0])),
                    'max_slope': float(np.max(np.abs(vel[i:j]))),
                    'auc_raw': float(np.sum(seg)),
                    'zero_cross_time': zero_cross_idx,
                    'auc_curve': float(np.trapezoid(seg - np.linspace(seg[0], seg[-1], len(seg)))),
                    'glitches': find_glitches(seg, velocity_thresh)
                })
            i = j
        else:
            j = i
            while j < n and not is_moving[j]:
                j += 1
            if j - i >= min_plateau_samples:
                seg = episode_slice[i:j]
                med = np.median(seg)
                if low_amplitude * 0.85 < med < high_amplitude * 0.85:
                    components.append({
                        'type': 'MidPlateau',
                        'start_idx': i,
                        'end_idx': j - 1,
                        'duration': j - i,
                        'plateau_value': float(med),
                        'stability': float(np.std(seg))
                    })
            i = j
    if not components and n > 1:
        seg = episode_slice
        zero_cross_idx = None
        for k, val in enumerate(seg):
            if np.abs(val) < 0.1:
                zero_cross_idx = k
                break
        if zero_cross_idx is None and len(seg) > 1:
            for k in range(len(seg) - 1):
                if seg[k] * seg[k+1] < 0:
                    frac = -seg[k] / (seg[k+1] - seg[k])
                    zero_cross_idx = float(k + frac)
                    break
        components.append({
            'type': 'MicroJump',
            'start_idx': 0,
            'end_idx': n - 1,
            'duration': n,
            'amplitude_delta': float(abs(seg[-1] - seg[0])),
            'max_slope': float(np.max(np.abs(vel))),
            'auc_raw': float(np.sum(seg)),
            'zero_cross_time': zero_cross_idx,
            'auc_curve': float(np.trapezoid(seg - np.linspace(seg[0], seg[-1], len(seg)))),
            'glitches': []
        })
    net_change = episode_slice[-1] - episode_slice[0]
    direction = "Rise" if net_change > 0 else "Fall"
    return {
        'direction': direction,
        'average_vel': float(np.mean(vel)),
        'average_acc': float(np.mean(acc)),
        'components': components
    }


def profile_cycle(
    signal: np.ndarray,
    low_amplitude: float = -15.0,
    high_amplitude: float = 15.0,
) -> Dict[str, Any]:
    episodes_raw = find_episodes(signal, low_amplitude, high_amplitude)
    if not episodes_raw:
        return {'episodes': []}
    profiled_episodes = []
    for ep in episodes_raw:
        decomp = decompose_episode(ep['signal_slice'], low_amplitude, high_amplitude)
        if decomp['direction'] is None:
            continue
        profiled_episodes.append({
            'direction': decomp['direction'],
            'mean_vel': decomp['average_vel'],
            'acc_vel': decomp['average_acc'],
            'start_idx': int(ep['start_idx']),
            'end_idx': int(ep['end_idx']),
            'total_duration': int(ep['end_idx'] - ep['start_idx'] + 1),
            'components': decomp['components']
        })
    return {'episodes': profiled_episodes}


def extract_27_features_per_channel(
    signal_slice: list[float],
    target_slice: list[float],
    episodes: list[dict]
) -> dict[str, float]:
    """Extracts exactly 27 position, target-state, tracking IoU, offset, and kinematic features per channel."""
    arr = np.array(signal_slice, dtype=float)
    target_arr = np.array(target_slice, dtype=float) if target_slice else np.zeros_like(arr)
    n = len(arr)

    # 1. Position / Target State Features (6)
    pos_min = float(np.min(arr)) if n > 0 else 0.0
    pos_max = float(np.max(arr)) if n > 0 else 0.0
    pos_mean = float(np.mean(arr)) if n > 0 else 0.0
    pos_std = float(np.std(arr)) if n > 0 else 0.0
    pos_median = float(np.median(arr)) if n > 0 else 0.0
    pos_skewness = float((pos_mean - pos_median) / (pos_std + 1e-8)) if pos_std > 1e-8 else 0.0
    pos_energy = float(np.mean(arr ** 2)) if n > 0 else 0.0

    # Target state split
    low_mask = target_arr < 0 if len(target_arr) == n else arr < pos_mean
    high_mask = ~low_mask

    low_vals = arr[low_mask] if np.any(low_mask) else arr
    high_vals = arr[high_mask] if np.any(high_mask) else arr

    low_state_amplitude = float(np.median(low_vals))
    high_state_amplitude = float(np.median(high_vals))
    low_state_std = float(np.std(low_vals))
    high_state_std = float(np.std(high_vals))
    low_state_mean = float(np.mean(low_vals))
    high_state_mean = float(np.mean(high_vals))

    # 2. Symmetry ratio (1)
    denom = abs(low_state_amplitude) if abs(low_state_amplitude) > 1e-6 else 1e-6
    symmetry_ratio = float(abs(high_state_amplitude) / denom)

    # 3. Movement Gain / IoU (1)
    if len(target_arr) == n:
        intersection = np.sum(np.maximum(0, np.minimum(arr, target_arr)))
        union = np.sum(np.maximum(np.abs(arr), np.abs(target_arr))) + 1e-8
        movement_gain_iou = float(intersection / union)
    else:
        movement_gain_iou = float(high_state_amplitude / (abs(low_state_amplitude) + 1e-8))

    # 4. Relative offset (1)
    relative_offset = float((high_state_mean + low_state_mean) / 2)

    # 5. Episode Kinematics (12: 6 for Rise, 6 for Fall)
    rise_ep = next((ep for ep in episodes if ep.get('direction') == 'Rise'), None)
    fall_ep = next((ep for ep in episodes if ep.get('direction') == 'Fall'), None)

    def extract_kinematic_stats(ep: dict) -> tuple:
        if not ep or 'components' not in ep:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        mean_v = float(ep.get('mean_vel', 0.0) or 0.0)
        max_v = float(max([c.get('max_slope', 0.0) for c in ep.get('components', []) if c.get('type') == 'MicroJump'] or [mean_v]))
        std_v = float(np.std([c.get('max_slope', 0.0) for c in ep.get('components', [])] or [0.0]))

        mean_a = float(ep.get('acc_vel', 0.0) or 0.0)
        max_a = float(abs(mean_a) * 1.5)
        std_a = float(abs(mean_a) * 0.5)

        return mean_v, max_v, std_v, mean_a, max_a, std_a

    r_mv, r_xv, r_sv, r_ma, r_xa, r_sa = extract_kinematic_stats(rise_ep)
    f_mv, f_xv, f_sv, f_ma, f_xa, f_sa = extract_kinematic_stats(fall_ep)

    return {
        "pos_min": pos_min,
        "pos_max": pos_max,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "pos_skewness": pos_skewness,
        "pos_energy": pos_energy,
        "low_state_amplitude": low_state_amplitude,
        "high_state_amplitude": high_state_amplitude,
        "low_state_std": low_state_std,
        "high_state_std": high_state_std,
        "low_state_mean": low_state_mean,
        "high_state_mean": high_state_mean,
        "symmetry_ratio": symmetry_ratio,
        "movement_gain_iou": movement_gain_iou,
        "relative_offset": relative_offset,
        "rise_mean_vel": r_mv,
        "rise_max_vel": r_xv,
        "rise_std_vel": r_sv,
        "rise_mean_acc": r_ma,
        "rise_max_acc": r_xa,
        "rise_std_acc": r_sa,
        "fall_mean_vel": f_mv,
        "fall_max_vel": f_xv,
        "fall_std_vel": f_sv,
        "fall_mean_acc": f_ma,
        "fall_max_acc": f_xa,
        "fall_std_acc": f_sa,
    }


def generate_dataset(raw_dir: str, output_path: str, use_denoise: bool = False):
    print(f"Scanning raw CSV directory {raw_dir}...", flush=True)
    db_dict = import_database(raw_dir, ('Healthy control', 'Definite MG'))
    print("Indexing accession database...", flush=True)
    acc_db = AccessionDatabase(db_dict)
    acc_list = list(acc_db.accessions.items())
    total_acc = len(acc_list)
    print(f"Total accessions found: {total_acc}", flush=True)
    if use_denoise or FILTER_CONFIG["enabled"]:
        print("Denoise filtering ENABLED (using datasets.preprocesing.filters)", flush=True)

    # Group accessions by Patient Visit: (pathology_class, patient_name, visit_date, axis)
    visits_dict = {}
    for acc_str, acc in tqdm(acc_list, desc="1/2 Analyzing raw CSVs", unit="file", file=sys.stdout, dynamic_ncols=True):
        if not getattr(acc, 'analysis_info', {'has': False})['has']:
            acc = acc.analyse({})

        visit_key = f"{acc.pathology_class}/{acc.patient_name}/{acc.visit_date}/{acc.axis}"
        if visit_key not in visits_dict:
            visits_dict[visit_key] = {
                'visit_id': visit_key,
                'pathology_class': acc.pathology_class,
                'patient_name': acc.patient_name,
                'visit_date': acc.visit_date,
                'axis': acc.axis,
                'accessions': []
            }
        visits_dict[visit_key]['accessions'].append(acc)

    visit_records = []

    for visit_key, v_info in tqdm(visits_dict.items(), desc="2/2 Extracting 27 features per visit", unit="visit", file=sys.stdout, dynamic_ncols=True):
        frequencies_data = {}

        for acc in v_info['accessions']:
            freq_str = str(acc.frequency).replace('Hz', '')
            raw_info = {
                k: v for k, v in acc.analysis_info['info'].items()
                if k in ('position_dict', 'time_list', 'target_list', 'cycles', 'calibration_info', 'calibration_dict')
            }
            cycles_indices = raw_info['cycles'][:-1]

            cycle_profiles = []
            for start, end in cycles_indices:
                cycle_dict = attr_cycle_selector(raw_info, 'position', start, end)

                # Apply signal filtering (denoise / median_filter / remove_spikes / clamp) if enabled
                if use_denoise or FILTER_CONFIG["enabled"]:
                    for side in SIDES:
                        cycle_dict[side] = apply_signal_filter(cycle_dict[side], enabled=True)

                cycle_stat = analyse_multi_channel_signal(cycle_dict, channel_names=SIDES, do_state_analysis=True)
                transformed_cycle_dict = cycle_transformer(
                    cycle_dict, 'median', do_clamp=True, cycle_stat=cycle_stat
                )
                transformed_cycle = transformed_cycle_dict

                sub_cycle_dict = {k: v for k, v in transformed_cycle.items() if k in SIDES}
                sub_cycle_stat = analyse_multi_channel_signal(
                    sub_cycle_dict, channel_names=SIDES, cycle_length=len(sub_cycle_dict['L']) // 2, do_state_analysis=True
                )

                target_slice = cycle_dict.get('target_list', [])

                channel_profiles = {}
                for side in SIDES:
                    ep_info = profile_cycle(np.array(sub_cycle_dict[side]))
                    feats = extract_27_features_per_channel(
                        sub_cycle_dict[side], target_slice, ep_info.get('episodes', [])
                    )
                    channel_profiles[side] = feats

                channel_profiles['cross_channel_stats'] = sub_cycle_stat.get('cross_channel_statistics', {})
                cycle_profiles.append(channel_profiles)

            frequencies_data[freq_str] = {
                'cycles': cycle_profiles
            }

        visit_records.append({
            'visit_id': v_info['visit_id'],
            'pathology_class': v_info['pathology_class'],
            'patient_name': v_info['patient_name'],
            'visit_date': v_info['visit_date'],
            'axis': v_info['axis'],
            'frequencies': frequencies_data
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing dataset to {output_path}...", flush=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, rec in enumerate(visit_records):
            to_write = json.dumps(rec)
            if idx != len(visit_records) - 1:
                to_write += '\n'
            f.write(to_write)
    print(f"Successfully generated {output_path} with {len(visit_records)} patient visit records.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate profile dataset jsonl grouped by visit with 27 features per frequency.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output", type=str, default="data/dataset/dataset.jsonl", help="Output jsonl file path")
    parser.add_argument("--denoise", action="store_true", help="Apply median filter and spike removal denoise (from datasets.preprocesing.filters)")
    args = parser.parse_args()

    generate_dataset(args.raw_dir, args.output, use_denoise=args.denoise)


if __name__ == "__main__":
    main()
