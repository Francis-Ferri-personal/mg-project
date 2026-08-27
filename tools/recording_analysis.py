import yaml

import yaml


RECORDING_ANALYSIS_SIDES = ('L','R')
DEFAULT_PERCENTILES = (1,5,10,20,25,30,40,50,60,70,75,80,90,95,99)


def basic_statistics(
    series:list[float], 
    quantiles:tuple = DEFAULT_PERCENTILES,
    expected_range:tuple=(-15,15)
) -> dict[str, ]:
    """Gets basic statistics of a series

    Args:
        series (list): A series of numbers
        quantiles (tuple, optional): Quantiles (in percent) to retreive. Defaults to DEFAULT_PERCENTILES.
        expected_range (tuple, optional): Series boundaries. Defaults to (-15,15).

    Returns:
        dict[str, ]:
         - series: the series itself, sorted
         - percentiles: values of the percentiles, keys are '{quantile}%' (e.g. '50%', '90%')
         - quantiles_indices: indices of the percentiles
         - statistics: actual statistics
    """
    
    series = sorted(series)
    series_len = len(series)

    quant_idx_dict = {
        quantile : int(series_len * (quantile/100)) - (min(quantile,1))
        for quantile in quantiles
    }

    percentiles_dict = {
        f'{k}%':series[v] for k,v in quant_idx_dict.items()
    }

    statistics_dict = {
        'median': (median:=percentiles_dict['50%']),
        'mean': (mean:=sum(series)/series_len),
        'mode': simple_hist(series,40,True)[-1][0],

        'min': (minimum:=series[0]),
        'max': (maximum:=series[-1]),
        'mid-extreme': (minimum+maximum)/2,

        'var': (var:=sum([(x-mean)**2 for x in series])/series_len),
        'std': (std:=var**0.5),

        'mad': sorted([x-median for x in series])[quant_idx_dict[50]],

        'nonparametric_skewness': std and (mean-median)/std or 0,
        'percentile_asymmetry': (percentiles_dict['95%'] - percentiles_dict['50%'])\
                                 - (percentiles_dict['50%'] - percentiles_dict['5%']),

        'raw_range': (raw_range:=maximum - minimum),
        'robust_range': (robust_range:=percentiles_dict['95%'] - percentiles_dict['5%']),

        'outside_ratio': len([x for x in series if x < expected_range[0] or x > expected_range[1]]) / len(series),
        'raw_range_ratio': raw_range / (expected_range[1]-expected_range[0]),
        'robust_range_ratio': robust_range / (expected_range[1]-expected_range[0]),

        'high_extreme_ratio': percentiles_dict['95%'] and maximum/percentiles_dict['95%'] or 0,
        'low_extreme_ratio': percentiles_dict['5%'] and minimum/percentiles_dict['5%'] or 0,
        
        'iqr':percentiles_dict['75%'] - percentiles_dict['25%'],

        'energy' : (energy:=sum([x*x for x in series])),
        'power_su' : energy/len(series),
        
    }

    output_dict = {
        'series':series,
        'percentiles':percentiles_dict,
        'quantiles_indices':quant_idx_dict,
        'statistics':statistics_dict
    }
    
    return output_dict

def simple_hist(series, n_bins, sort_output:bool=False) -> list[tuple[float,int]]:
    series = sorted(series)
    series_len = len(series)

    amt_per_bin = series_len//n_bins
    if amt_per_bin < 1:
        raise IndexError("n_bins cannot be bigger than length of series")

    mid_point = amt_per_bin//2

    output = {}
    counted = 0

    while counted<series_len:
        end_point = min(counted+amt_per_bin, series_len)
        tmp = series[counted:end_point]
        tmp_mid = tmp[min(mid_point, len(tmp)//2 - 1)]

        output[tmp_mid] = output.setdefault(tmp_mid,0) + len(tmp)
        counted += amt_per_bin

    if sort_output:
        output = {k:v for k,v in sorted(output.items(),key = lambda x:x[1])}

    return list(output.items())

def channel_cross_correlation_helper(channel1, channel2, cycle_period) -> tuple[float, int]:
    shifts = list(range(cycle_period))
    current_highest = -1e10
    current_shift = None

    for shift in shifts:
        tmp_channel2 = [channel2[(i+shift)%len(channel2)] for i in range(len(channel2))]

        shift_score = sum([c1*c2 for c1,c2 in zip(channel1,tmp_channel2)])
        
        if shift_score > current_highest:
            current_highest = shift_score
            current_shift = shift

    return current_highest, current_shift

def analyse_multi_channel_signal(
        signal:dict[str, list[float]], 
        channel_names:tuple = RECORDING_ANALYSIS_SIDES,
        cycle_length:int = None, 
        do_state_analysis:bool=False,
        quantiles:tuple=DEFAULT_PERCENTILES
    ) -> dict[str,dict[str]]:
    """Gets basic statistics of a signal for each channel

    if cycle_length is given a number, will return cross channel statistics
    for the first two channels in channel_names

    Args:
        signal (dict[str, list[float]]): Expects a dictionary with channel names as keys where each value is a list of float
        channel_name (tuple, optional): a tuple containing channel keys. Defaults to ('L','R')
        cycle_length (int, optional): length of cycle, for cross-correlation. If None, will not return cross_channel_statistics
        do_state_analysis (bool, optional): For square signals. Defaults to False.

    Returns:
        dict:
        - channel:
            - percentiles
            - statistics
            - cross_channel_statistics (optional)
            
            if do_state_analysis is set to True, the following keys are available as well:
                - low_state_statistics
                - high_state_statistics
    """
    # PER CHANNEL STUFF
    channel_series_dict = {}
    channel_layer_dict = {}
    
    for channel_name in channel_names:
        if channel_name not in signal: #maybe only left and right
            print(f"Channel {channel_name} not found in signal, skipping")
            continue

        channel_series = signal[channel_name]

        channel_series, channel_percentiles, channel_quant_idx, channel_statistics = basic_statistics(channel_series, quantiles).values()
        channel_series_dict[channel_name] = channel_series

        channel_output = {
            'percentiles':channel_percentiles,
            'statistics':channel_statistics,
        }

        if do_state_analysis:
            low_state = basic_statistics(channel_series[:channel_quant_idx[40]], quantiles, (-20.2,-9.8)) 
            high_state = basic_statistics(channel_series[channel_quant_idx[60]:], quantiles, (9.8, 20.2))
            
            channel_statistics |= {
                'low_state_amplitude':(lsm:=low_state['percentiles']['50%']),
                'high_state_amplitude':(hsm:=high_state['percentiles']['50%']),
                'baseline':((lsm+hsm)/2),
                'state_separation': channel_percentiles['60%'] - channel_percentiles['40%'],
                'state_symmetry_ratio':hsm and abs(lsm)/abs(hsm) or 0,
            }

            additional_statistics = {
                'low_state_statistics':low_state['statistics'],
                'high_state_statistics':high_state['statistics'],
            }

            channel_output |= additional_statistics

        channel_layer_dict[channel_name] = channel_output

    output_dict = {
        **channel_layer_dict
    }

    # CROSS CHANNEL STUFF
    if cycle_length is not None:
        channel_1_name, channel_2_name = channel_names[:2]
        channel_1_layer = channel_layer_dict[channel_1_name]
        channel_2_layer = channel_layer_dict[channel_2_name]

        cc_median_diff = channel_1_layer['percentiles']['50%'] - channel_2_layer['percentiles']['50%']
        cc_amplitude_ratio =  channel_2_layer['statistics']['raw_range'] \
            and channel_1_layer['statistics']['raw_range'] / channel_2_layer['statistics']['raw_range']\
            or 0
        
        _, cc_lag = channel_cross_correlation_helper(
            channel_series_dict[channel_1_name],
            channel_series_dict[channel_2_name],
            cycle_length
        )
        
        cross_channel_statistics = {
            'median_difference': cc_median_diff,
            'amplitude_ratio': cc_amplitude_ratio,
            'lag': cc_lag
        }

        output_dict['cross_channel_statistics'] = cross_channel_statistics

    return output_dict

def analyse_record(accession_str:str, 
                   raw_info:dict,  
                   cycle_length:int, 
                   output_dict:dict=None, **kwargs) -> dict:
    
    output_result = analyse_multi_channel_signal(
        raw_info['position_dict'], cycle_length, True
    )
    
    output_result |= {
        'has_calibration':raw_info['calibration_info']['exists'],
        'calibrations':raw_info['calibration_dict'],
    }

    if output_dict is not None:
        output_dict[accession_str] = output_result
    
    return output_result