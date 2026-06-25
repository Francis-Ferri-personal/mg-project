def merge_dictionaries(d:list[dict]) -> dict:
    """Combines multiple dictionaries in a list into one dictionary
    ! Overwrites key-value pair !

    Args:
        d (list[dict]): A list of dictionaries

    Raises:
        TypeError: If not all of the inputs are dictionaries, this will be raised

    Returns:
        dict: A combined dictionary
    """


    if not all (isinstance(arg, dict) for arg in d):
        raise TypeError("Items to join must be of type dictionary!")
    
    values = d
        
    out = {}

    for v in values:
        out.update(v)
    
    return out

def merge_sublists(l:list|tuple|set) -> list:
    """Combines multiple lists inside a list to a big list

    Args:
        l (list | tuple | set): A list of lists

    Raises:
        TypeError: If the type of input is not a list, set, or tuple, raises athis exception.

    Returns:
        list: A combined list
    """
    if not isinstance(l,(list,set,tuple)):
        raise TypeError
    
    out = []
    for l_ in l:
        out += list(l_)
    return out

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
        return clamp(out_val, min_out, max_out)
    
    return out_val

def float_range(start:int|float, 
                stop:int|float, 
                step:int|float) -> list:
    """Returns a list from start to stop with steps in between (inclusive)
    Probably slow

    Args:
        start (int | float): value
        stop (int | float): value
        step (int | float): value

    Returns:
        list
    """
    out = [start]
    while start < stop:
        start += step

        if stop-start < 9e-6:
            out.append(stop)
            break

        out.append(start)
    return out

def series_split(series:list, split_ratio:tuple[float]) -> list[list]:
    """Splits a series into multiple sections

    Args:
        series (list): A series
        split_ratio (float|tuple[float]): 
            - If singular float, the series is uniformly split with this ratio
            - If multiple floats but not 1, the series is split by weighting and the last region is 1-total float
            - If multiple floats, the series is split by weighting

    Raises:
        Exception: If split is above 1.0 (100%)

    Returns:
        list[list]: A list of split sections
    """
    if not isinstance(split_ratio,(list,tuple)):
        split_ratio = [split_ratio]

    if sum(split_ratio)>1:
        raise Exception("Split shouldn't amount to be above 1")
    
    if len(split_ratio) == 1:
        split_ratio = float_range(0,1,split_ratio[0])
    
    if split_ratio[0]!=0:
        splits = [0]
        for key in split_ratio:
            x = key+splits[-1]
            splits.append(key+splits[-1])
        if splits[-1] < 1.0:
            splits += [1]
        split_ratio = splits.copy()

    N = len(series)
    
    split_region = [int(N*rat) for rat in split_ratio]
    
    sections = [series[split_region[i-1]:split_region[i]]
                for i in range(1,len(split_region))]
    
    return sections

def transfer_indices(series, target_len, do_lerp:bool=True) -> list:
    N = len(series)
    original_indices_ratio = [i/(N-1) for i in range(N)]
    new_indices_ratio = [i/(target_len-1) for i in range(target_len)]

    out = []
    ratio_pointer = 0

    for index_ratio in new_indices_ratio:
        while True:
            current_ratio = original_indices_ratio[ratio_pointer]
            prev_ratio = original_indices_ratio[max(0,ratio_pointer-1)]

            if current_ratio == index_ratio:
                out.append(series[ratio_pointer])
                break

            if prev_ratio < index_ratio and current_ratio > index_ratio:
                prev_val, current_val = series[ratio_pointer-1:ratio_pointer+1]

                if do_lerp:
                    out_val = lerp(
                        index_ratio,
                        prev_ratio, current_ratio,
                        prev_val,current_val 
                    )
                
                else:
                    out_val = (index_ratio-prev_ratio) < (current_ratio-index_ratio) \
                                and prev_val \
                                or current_val
                    
                out.append(out_val)
                break

            ratio_pointer += 1

    return out  

