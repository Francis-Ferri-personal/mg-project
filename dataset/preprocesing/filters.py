from statistics import median


def median_filter(series: list, kernel_size: int = 3) -> list:
    out = list(series)
    half = kernel_size // 2
    for i in range(len(series)):
        start = max(0, i - half)
        end = min(len(series), i + half + 1)
        out[i] = median(series[start:end])
    return out


def remove_spikes(series: list, threshold_mult: float = 5.0, kernel: int = 5) -> list:
    out = list(series)
    half = kernel // 2
    for i in range(len(series)):
        start = max(0, i - half)
        end = min(len(series), i + half + 1)
        local = series[start:end]
        local_median = median(local)
        deviations = [abs(x - local_median) for x in local]
        mad = median(deviations) if deviations else 0
        diff = abs(series[i] - local_median)
        if mad == 0:
            if diff != 0:
                prev_same = i > 0 and series[i - 1] == series[i]
                next_same = i < len(series) - 1 and series[i + 1] == series[i]
                if not prev_same and not next_same:
                    out[i] = local_median
        elif diff > threshold_mult * mad:
            out[i] = local_median
    return out


def clamp(series: list, lo: float = -25, hi: float = 25) -> list:
    return [max(lo, min(hi, x)) for x in series]


def denoise(series: list, kernel: int = 7, passes: int = 3) -> list:
    s = clamp(series)
    s = median_filter(s, kernel)
    for th in [5.0, 4.0, 3.0][:passes]:
        s = remove_spikes(s, threshold_mult=th, kernel=kernel)
    return s
