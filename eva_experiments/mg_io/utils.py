import os
from collections import deque

def anchored_listdir(dir:str, 
                   ensure_visible:bool=True, 
                   allowed:list[str]=None) -> list[str]:
    """Calls os.listdir and appends current directory to listeed subdirectories
    Creates a long path to target subdirectory

    Args:
        - dir (str): The target directory to call listdir on.
        - ensure_visible (bool): Only returns items in the directory whose name do not start with '.', defaults to True.
        - allowed (list[str]): A list of items in the directory permitted to be returned; if not given, all items are returned.

    Raises:
        TypeError: If allowed is given but not an iterable, raises TypeError

    Returns:
        list[str]: A list of subdirectories
    """
    
    out = os.listdir(dir)
    
    if ensure_visible:
        out = [sub for sub in out if sub[0]!='.']
    
    if allowed is not None:
        if not isinstance(allowed,(list,tuple,set)):
            raise TypeError("Allowlist must be an iterable")
        
        out = [sub for sub in out if sub in allowed]

    return [os.path.join(dir,sub) for sub in out]

def get_all_csv(dir:str) -> dict[str,list]:
    """Gets all CSV under a specific directory with complete paths from the
    given directory to the CSV files. CSV files in a subfolder will be grouped

    Args:
        dir (str): The initial dictionary to scan in

    Returns:
        dict[str,list]: A dictionary with group paths as keys and lists of csvs as values
    """
    out = {}
    seen = deque([dir])
    while seen:

        current = seen.popleft()
        if current is None:
            break

        # current_subdir = map(lambda x: os.path.join(current, x), os.listdir(current))
        current_subdir:list[str] = anchored_listdir(current)

        if current_subdir and all ([subdir[-3:]=='csv' for subdir in current_subdir]):
            dict_key = os.path.dirname(current)
            id_key : str = current.split('/')[-1]
            visit_date, patient_name = id_key.split(' ', maxsplit=1)
            visit_info = {visit_date: current_subdir}

            out[dict_key] = out.setdefault(dict_key, dict())
            out[dict_key][patient_name] = out[dict_key].setdefault(patient_name,dict())|visit_info
            
            continue

        seen.extend(current_subdir)
    return out  