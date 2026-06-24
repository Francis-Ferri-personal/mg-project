
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

def find_gain(position_series:list, target_series:list) -> list:
    """Finds the ratio of position from target, normalised in 0 to 1 scale

    Args:
        position_series (list): Position as a list
        target_series (list): Targets as a list

    Returns:
        list: A list of gain
    """
    gain_series = []
    
    for p,t in zip(position_series, target_series):
        if t == 0:
            gain_series.append(0)
            continue
        if t < 0:
            gain_series.append(lerp(p/t,1.0,-1.0,0.0,1.0))
            continue
        gain_series.append(lerp(p/t,-1.0,1.0,0.0,1.0))

    return gain_series


def lerp(value:int|float, 
         min_in:int|float, 
         max_in:int|float, 
         min_out:int|float, 
         max_out:int|float,
         do_clamp:bool=True) -> int|float:
    """Linearly maps value from input range to output range 

    Args:
        value ( int|float ): value in input range
        min_in ( int|float ): minimum value in input
        max_in ( int|float ): maximum value in input
        min_out ( int|float ): minimum value in output
        max_out ( int|float ): maximum value in output
        do_clamp ( bool ): clamps output to output range

    Returns:
        int|float: Mapping of input value in output range
    """
    domain = max_in - min_in
    range_ = max_out - min_out
    ratio = (value - min_in) / domain
    
    out_val = min_out + ratio*range_

    if do_clamp:
        return clamp(value, min_out, max_out)
    
    return out_val

def clamp(value:int|float, min_val:int|float, max_val:int|float) -> int|float:
    """Clamps value between `min_val` and `max_val`

    Args:
        value (int | float)
        min_val (int | float)
        max_val (int | float)

    Returns:
        int | float
    """
    return min(max_val,max(min_val,value))