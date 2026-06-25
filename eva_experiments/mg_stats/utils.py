import math

def rate_of_change(
        series:list,
        delta:float=1/120) -> list:
    """A more complicated velocity calculation

    Args:
        sample_list (list): A list containing values
        time_between_sample (float, optional): Time in seconds. Defaults to 1/120.

    Returns:
        list: Rate of change
    """
    return [0] \
            + [(series[idx]-series[idx-1])/delta
               for idx in range(1, len(series))]

def rolling_average(data:list|tuple, window_size=0) -> list:
    """Smooths out data

    Args:
        data ( iterable ) : value series
        window_size (int, optional): how many samples to average. Defaults to 0.

    Returns:
        List: smoothed out data
    """
    if window_size < 2:
        return data
    
    if window_size % 2 == 0 :
        window_size += 1
    
    half_window = window_size//2
    data_end = len(data)
    out = []

    for i in range(len(data)):
        start = max(0,i-half_window)
        end = min(data_end,i+half_window)
        out.append(sum(data[start:end])/window_size)
    
    return out

def min_max_normalise(series:list) -> list:
    """Normalise a series by linearly interpolating every point in series
    between series minimum and maximum

    Args:
        series (list): A series of ints or floats

    Returns:
        list: A series of floats from 0 to 1
    """
    s_max, s_min = max(series), min(series)
    dist = s_max - s_min
    return [(entry-s_min)/dist for entry in series]


def percentile_normalise(series:list, thresh=0.95) -> list:
    """Normalise a series by linearly interpolating every point in series
    between points in the middle thresh percentile

    Args:
        series (list): A series of ints or floats
        thresh (float, optional): The ratio of middle region that is valid. Defaults to 0.95 (from 0.025 to 0.0975).

    Returns:
        list: A series of floats from 0 to 1
    """
    actual_thresh = (1-thresh)/2
    p_pos = math.ceil(len(series)*(1-actual_thresh))
    p_neg = math.ceil(len(series)*(actual_thresh))
    s_sorted = sorted(series)
    s_max = s_sorted[p_pos]
    s_min = s_sorted[p_neg]
    dist = s_max-s_min
    return [entry>=s_max and 1 or entry>s_min and (entry-s_min)/dist or 0 for entry in series]

def pointwise_mean(nested_series:list[list]):
    N = len(nested_series)
    return [sum(x)/N for x in zip(*nested_series)]

def pointwise_std(nested_series:list[list]):
    N = len(nested_series)
    pw_mean= pointwise_mean(nested_series)

    # some cursed nesting dawg
    numerator = [[(xi-x_mean)**2 for xi in xs] 
                 for xs,x_mean in zip(zip(*nested_series),pw_mean)]
    return [((sum(num)/N)**0.5) for num in numerator]