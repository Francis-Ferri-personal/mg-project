import os
import sys
import json
import argparse
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tools.data_loader import import_database
from tools.accessions_db import AccessionDatabase
from tools.recording_analysis import analyse_multi_channel_signal

SIDES = ('L', 'R')

CHANNEL_STATISTICS = (
    'median', 'mean', 'mode',
    'max', 'min', 'var', 'std',
    'nonparametric_skewness',
    'robust_range',
    'low_state_amplitude',
    'high_state_amplitude',
    'state_symmetry_ratio',
    'iqr', 'energy', 'power_su'
)


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
    do_translate: bool = True,
    do_clamp: bool = True,
    cycle_stat: dict = None,
    clamps_mult: float = 2.0
) -> dict[str, list[float]]:
    if not do_clamp and not do_translate:
        return cycle_dict
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
        if do_translate:
            series = [val - baseline for val in series]
        if do_clamp:
            low_bound = low_stats['mean'] - low_stats['std'] * clamps_mult
            high_bound = high_stats['mean'] + high_stats['std'] * clamps_mult
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
    low_std: float = 1.0,
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


def select_exports(og_dict: dict, **kwargs) -> dict:
    out_dict = {}
    for k, values in kwargs.items():
        out_dict[k] = out_dict.setdefault(k, {})
        for val in values:
            out_dict[k][val] = og_dict[k][val]
    return out_dict


def generate_dataset(raw_dir: str, output_path: str):
    print(f"Loading database from {raw_dir}...")
    db_dict = import_database(raw_dir, ('Healthy control', 'Definite MG'))
    acc_db = AccessionDatabase(db_dict)
    acc_list = list(acc_db.accessions.items())
    total_acc = len(acc_list)
    print(f"Total accessions found: {total_acc}")

    recording_profiles = []
    for acc_str, acc in tqdm(acc_list, desc="Generating dataset profiles", unit="acc"):
        if not getattr(acc, 'analysis_info', {'has': False})['has']:
            acc = acc.analyse({})
        
        raw_info = {
            k: v for k, v in acc.analysis_info['info'].items()
            if k in ('position_dict', 'time_list', 'target_list', 'cycles', 'calibration_info', 'calibration_dict')
        }
        cycles_indices = raw_info['cycles'][:-1]
        
        cycle_profiles = []
        for start, end in cycles_indices:
            cycle_dict = attr_cycle_selector(raw_info, 'position', start, end)
            cycle_stat = analyse_multi_channel_signal(cycle_dict, channel_names=SIDES, do_state_analysis=True)
            transformed_cycle_dict = cycle_transformer(
                cycle_dict, 'median', do_translate=True, do_clamp=True, cycle_stat=cycle_stat
            )
            transformed_cycle = cycle_dict | transformed_cycle_dict
            
            sub_cycle_dict = {k: v for k, v in transformed_cycle.items() if k in SIDES}
            sub_cycle_stat = analyse_multi_channel_signal(
                sub_cycle_dict, channel_names=SIDES, cycle_length=len(sub_cycle_dict['L']) // 2, do_state_analysis=True
            )
            
            complete_profile = {}
            for side in SIDES:
                side_info = select_exports(sub_cycle_stat[side], statistics=CHANNEL_STATISTICS)
                complete_profile[side] = side_info | profile_cycle(np.array(sub_cycle_dict[side]))
            
            complete_profile |= {'cross_channel_stats': sub_cycle_stat['cross_channel_statistics']}
            cycle_profiles.append(complete_profile)

        recording_profiles.append({
            'acc_str': acc.accession_str,
            'cycles': cycle_profiles
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, rec in enumerate(recording_profiles):
            to_write = json.dumps(rec)
            if idx != len(recording_profiles) - 1:
                to_write += '\n'
            f.write(to_write)
    print(f"Successfully generated {output_path} with {len(recording_profiles)} recording profiles.")


def main():
    parser = argparse.ArgumentParser(description="Generate profile dataset jsonl from raw CSV accessions.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output", type=str, default="data/dataset/dataset.jsonl", help="Output jsonl file path")
    args = parser.parse_args()

    generate_dataset(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
