RECORDING_ANALYSIS_SIDES = ('L','R','AVG')

def basic_statistics(series, expected_range:tuple=(-15,15)) -> dict[list, dict, dict]:
    series = sorted(series)
    series_len = len(series)

    percentiles_dict = {
        0: {'index':0, 'value':series[0]},
        **{p:{'index':(index:=(int(series_len * (p/100))-1)), 'value':series[index]} 
                        for p in (1,5,10,20,25,30,40,50,60,70,75,80,90,95,99)},
        100: {'index':series_len-1, 'value':series[-1]}
    }

    statistics_dict = {
        'median': (median:=percentiles_dict[50]['value']),
        'mean': (mean:=sum(series)/series_len),

        'min': (minimum:=series[0]),
        'max': (maximum:=series[-1]),
        'mid-extreme': (minimum+maximum)/2,

        'var': (var:=sum([(x-mean)**2 for x in series])/series_len),
        'std': (std:=var**0.5),

        'mad': sorted([x-median for x in series])[percentiles_dict[50]['index']],

        'nonparametric_skewness': std and (mean-median)/std or 0,
        'percentile_asymmetry': (percentiles_dict[95]['value'] - percentiles_dict[50]['value'])\
                                 - (percentiles_dict[50]['value'] - percentiles_dict[5]['value']),

        'raw_range': (raw_range:=maximum - minimum),
        'robust_range': (robust_range:=percentiles_dict[95]['value'] - percentiles_dict[5]['value']),

        'outside_ratio': len([x for x in series if x < expected_range[0] or x > expected_range[1]]) / len(series),
        'raw_range_ratio': raw_range / (expected_range[1]-expected_range[0]),
        'robust_range_ratio': robust_range / (expected_range[1]-expected_range[0]),

        'high_extreme_ratio': percentiles_dict[95]['value'] and maximum/percentiles_dict[95]['value'] or 0,
        'low_extreme_ratio': percentiles_dict[5]['value'] and minimum/percentiles_dict[5]['value'] or 0,
        
        'iqr':percentiles_dict[75]['value'] - percentiles_dict[25]['value'],

        'energy' : (energy:=sum([x*x for x in series])),
        'power_su' : energy/len(series),
        
    }

    output_dict = {
        'series':series,
        'percentiles':percentiles_dict,
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

def analyse_multi_channel_signal(attr_info:dict, cycle_length:int, do_state_analysis:bool=False):
    # PER CHANNEL STUFF
    channel_series_dict = {}
    channel_layer_dict = {}
    
    for side_idx, side in enumerate(RECORDING_ANALYSIS_SIDES):
        channel_series = attr_info[side]

        channel_series, channel_percentiles, channel_statistics = basic_statistics(channel_series).values()

        if do_state_analysis:
            low_state = basic_statistics(channel_series[:channel_percentiles[40]['index']], (-20.2,-9.8)) 
            high_state = basic_statistics(channel_series[channel_percentiles[60]['index']:], (9.8, 20.2))

            channel_mode = [simple_hist(state['series'],30,True)[-1][0] for state in (low_state, high_state)]
            
            channel_statistics |= {
                'channel_mode':channel_mode,
                'low_state_amplitude':(lsm:=low_state['percentiles'][50]['value']),
                'high_state_amplitude':(hsm:=high_state['percentiles'][50]['value']),
                'state_separation': channel_percentiles[60]['value'] - channel_percentiles[40]['value'],
                'state_symmetry_ratio':hsm and abs(lsm)/abs(hsm) or 0,
            }

            restructured_percentiles = {
                'low_state':low_state['percentiles'],
                'high_state':high_state['percentiles'],
                'channel':channel_percentiles
            }
        
        else:
            restructured_percentiles = {
                'channel':channel_percentiles
            }

        # RESTRUCTURE PERCENTILES
        for percentile_key, state_percentile_dict in restructured_percentiles.items():
            tmp_percentile_dict = {}

            for k, v in state_percentile_dict.items():
                tmp_percentile_dict[f'{k}%'] = v['value']
            
            restructured_percentiles[percentile_key] = {k:v for k,v in tmp_percentile_dict.items()}

        for k, v in restructured_percentiles.items():
            if k=='low_state':
                low_state['percentiles'] = v
                continue
            if k=='high_state':
                high_state['percentiles'] = v
                continue
            channel_percentiles = v

        # OUTPUT

        channel_series_dict[side] = channel_series

        additional_statistics = {}
        if do_state_analysis:
            additional_statistics = {
                'low_state_statistics':low_state['statistics'],
                'high_state_statistics':high_state['statistics'],
            }

        channel_layer_dict[side] = {
            'percentiles':channel_percentiles,
            'statistics':channel_statistics,
            ** additional_statistics
        }
    
    # CROSS CHANNEL STUFF
    cc_median_diff = channel_layer_dict['L']['percentiles']['50%'] - channel_layer_dict['R']['percentiles']['50%']
    cc_amplitude_ratio =  channel_layer_dict['R']['statistics']['raw_range'] \
                            and channel_layer_dict['L']['statistics']['raw_range'] / channel_layer_dict['R']['statistics']['raw_range']\
                            or 0
    
    _, cc_lag = channel_cross_correlation_helper(channel_series_dict['L'],channel_series_dict['R'],cycle_length)
    
    cross_channel_statistics = {
        'median_difference': cc_median_diff,
        'amplitude_ratio': cc_amplitude_ratio,
        'lag': cc_lag
    }

    output = {
        **channel_layer_dict,
        'cross_channel_statistics':cross_channel_statistics
    }

    return output
    

def analyse_record(acc_str:str, raw_info:dict, 
                    processed_info:dict, cycle_length:int, 
                    output_dict:dict=None) -> dict:
    
    output_result = analyse_multi_channel_signal(
        processed_info['position_dict'], cycle_length, True
    )

    output_result |= {
        'has_calibration':raw_info['calibration_info']['exists'],
        'calibrations':raw_info['calibration_dict'],
    }

    if output_dict is not None:
        output_dict[acc_str] = output_result
    
    return output_result