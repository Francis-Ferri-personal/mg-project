import os
import sys

sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))

from eva_experiments.utils import lerp

# def find_saccades(
#         df:pd.DataFrame,
#         axis:str,
#         smoothing:int=25,
#         threshold = 3.5,
#         debug=-1):
    
    
#     sample_freq = 120
#     time_bet_sample = 1/sample_freq
    
#     positions_of_interest = [f'{x}{axis[0].upper()}' for x in 'LR']

#     position1, position2 = [rolling_average(list(df[x]), smoothing) 
#                             for x 
#                             in positions_of_interest]
    
#     time_list = list(df['Time(sec)'])

#     saccade_positions = []

#     final_idx = len(position1)-1

#     #slide window:
#     idx = 0
#     while idx < final_idx:
#         tmp_idx = idx
#         if debug > 2:
#             print(f'current idx: {idx}; current tmp_idx: {tmp_idx}')

#         start_idx = min(tmp_idx, final_idx)
#         end_idx = start_idx + 12

#         if end_idx >= final_idx:
#             idx = final_idx
#             break

#         start = (position1[start_idx] + position2[start_idx])/2
#         end = (position1[end_idx] + position2[end_idx])/2
        
#         possible_saccade = (q:=abs(end-start)) > threshold
#         if possible_saccade and debug>-1:
#             print(f'! searching possible saccade region at {tmp_idx} ({time_list[tmp_idx]}s) !\n! start: {start} | end: {end} | diff: {q} !')

#         start_idx = end_idx
#         shifter = 4

#         while possible_saccade:
#             start_idx = (start_idx + shifter)
#             end_idx = start_idx + 4
            
#             if debug > 1:
#                 print(f'\t| searching {start_idx}-{end_idx}')

#             if end_idx >= final_idx:
#                 idx = final_idx
#                 break

#             start = (position1[start_idx] + position2[start_idx])/2
#             end = (position1[end_idx] + position2[end_idx])/2

#             found_saccade = (q:=abs(end-start)) > (threshold/2)

#             if not found_saccade:
#                 if debug > 0:
#                     print(f'| no significant change found at {start_idx}-{end_idx}\n\t\t| start: {start} | end: {end} | diff: {q}')
#                 idx = end_idx
#                 break
            
#             if debug > 0:
#                 print(f'\t\t| significant change found at {start_idx}-{end_idx}\n\t\t| start: {start} | end: {end} | diff: {q} !')
#             shifter += 4
        
#         if not possible_saccade:
#             idx += 4
#             continue
        
#         if (idx-tmp_idx)>4:
#             if debug > -1:
#                 print(f'! found possible saccade at {tmp_idx} ({time_list[tmp_idx]}s)!\n')
#             saccade_positions.append((tmp_idx, time_list[tmp_idx]))

#     # return group_label, positions_of_interest, freq_label
#     return saccade_positions

def find_saccades2(speed_pair, 
                   threshold = 32,
                   max_dur = 120,
                   min_dur = 35,
                   time_list = None,
                   debug = -1):
    
    raise NotImplementedError("Finding Saccades has not been properly implemented")

    if debug > -1:
        print('**** FINDING SACCADES ****')
        print(f'\t| Threshold: {threshold}')
    
    spd_bundle = list(speed_pair)

    if len(spd_bundle) == 2:
        spd1, spd2 = spd_bundle
        spd_avg = [(v1+v2)/2 for v1,v2 in zip(spd1,spd2)]
    elif len(spd_bundle) == 3:
        spd1, spd2, spd_avg = spd_bundle
    else:
        raise Exception()
    if debug > 0:
        print('\t\t| average biaxial speed calculated!')

    N = len(spd_avg)

    corrector = min(spd_avg)
    spd_avg = [s-corrector for s in spd_avg]

    if debug > 1:
        print(f'\t\t\t| Corrector = {corrector}')

    current_idx = 0
    last_saccade = 0
    saccade_idx = []
    in_saccade = False

    if debug>-1:
        print('\t| looping to find_saccade')
    while current_idx < (N-1):
        time_to_last_saccade = current_idx - last_saccade

        if time_to_last_saccade < min_dur:
            current_idx += 1
            continue

        v = spd_avg[current_idx]

        if debug > 2:
            print(f'\t\t\t\t| checking index: {current_idx}{f'({time_list[current_idx]})' if time_list is not None else ''}, value: {v}; in saccade? {in_saccade}; time since last: {time_to_last_saccade}')

        if abs(v)>=threshold:
            if debug > 0:
                print(f'\t\t\t| Value ({v}) above threshold at index {current_idx}{f'({time_list[current_idx]})' if time_list is not None else ''}\n\t\t\t| indices since last saccade: {time_to_last_saccade}')

            saccade_idx.append(current_idx) 
            last_saccade = current_idx

        current_idx += 1
    if debug >-1 :
        print(f'\t| found {len(saccade_idx)} saccades, returning')

    return saccade_idx

def find_jumps(target_list:list, direction:str='both', debug=-1) -> list[tuple[str,int]]:
    """Finds jumps in targets

    Args:
        target_list (list): a list of target positions
        direction (str, optional): 'positive', 'negative', or 'both. Defaults to 'both'.
        debug (int, optional): Print out debug information. Defaults to -1.

    Returns:
        list: A list of tuples ([direction],[jump_index])
    """
    if direction is None:
        direction = 'both'

    jumps_ids = [i for i in range(1,len(target_list)-4) if abs(target_list[i])==15]
    if debug>-1:
        print(jumps_ids)

    if direction == 'positive':
        jumps_ids = [(direction,i) for i in jumps_ids if (target_list[i]-target_list[i-1]) >= 15] 
    
    elif direction == 'negative':
        jumps_ids = [(direction,i) for i in jumps_ids if (target_list[i]-target_list[i-1]) <= -15]

    elif direction == 'both':
        jumps_ids = [(q>0 and 'positive' or 'negative',i) for i in jumps_ids if abs(q:=(target_list[i]-target_list[i-1])) >= 15]
    
    else:
        raise Exception("Direction has to be 'positive', 'negative', or 'both'")

    out = []
    if target_list[0]!=0:
        jump_direction = target_list[0]>0 and 'positive' or 'negative'
        if jump_direction != direction and direction!='both':
            out = []
        else:
            out = [(jump_direction,0)]
    
    return out+jumps_ids

def find_gain(position_series:list, target_series:list, do_clamp=True) -> list:
    """Finds the ratio of position from target, normalised in 0 to 1 scale

    Args:
        position_series (list): Position as a list
        target_series (list): Targets as a list
        do_clamp (bool): Clamps output to 0,1 range

    Returns:
        list: A list of gain
    """
    gain_series = []
    
    for p,t in zip(position_series, target_series):
        if t == 0:
            gain_series.append(0)
            continue
        gain_series.append(lerp(p/t,-1.0,1.0,0.0,1.0,do_clamp))

    return gain_series