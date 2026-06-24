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