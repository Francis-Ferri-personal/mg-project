import os
import gc

import matplotlib.pyplot as plt
import matplotlib

from pandas import read_csv, DataFrame
from .utils import lerp, clamp, transfer_indices, float_range
from .stats import rate_of_change, rolling_average, pointwise_mean, percentile_normalise
from .recording_analysis import basic_statistics

CSV_ENCODING = 'utf-16-le'
ACCESSION_INFO_KEYS = (
    'accession',
    'accession_str',
    'pathology_class',
    'patho_class_idx',
    'patient_name',
    'patient_idx',
    'visit_date',
    'visit_idx',
    'axis',
    'axis_idx',
    'rel_path',
    'file_idx',
    'frequency',
)
PATHOLOGY_CLASSES_DEFAULT = ('Healthy control', 'Definite MG')
AXIS_KEYS = ('horizontal','vertical')

class Accession:
    KINEMATIC_ATTRIBUTES = ('position','gain','speed','velocity','acceleration')

    def __init__(self, accession_path:str = None, accession_info:dict = None, accession_str = None):
        """Class for singular Accession

        Args:
            accession_path (str): Path to CSV file
            accession_info (dict): General Accession Information

        Accession Info:
        - accession (list) : Accession
        - rel_path (str) : Relative path from the current file to CSVs
        - patient_name (str) : Name of the patient
        - visit_date (str) : Visit date
        - pathology_class (str) : Pathology class value
        - patho_class_idx (int) : Index of pathology class
        - visit_idx (int) : Index of visit
        - axis (str) : Axis of the accession
        - file_idx (int): File index
        - frequency (str) : Frequency of the file 
        """
        if accession_path is None:
            if accession_info is not None:
                raise ValueError('Has to have accession_path if accession_info is given!')
            
            if accession_str is None:
                raise ValueError('accession_str has to be provided if accession_path is not given!')
            
            accession_info = Accession.__translate_accession_str(accession_str)

            return
        self.accession_path = accession_path

        for k,v in accession_info.items():
            setattr(self, k, v)

        self.standard_cycle_length = [480,360,240][self.file_idx]
        self.standard_cycle_duration = int(self.standard_cycle_length/120)

    DEFAULT_ANALYSIS_NORMAL = {
        'gain':{'rawness':'normal','value_smoothing':3,'source_smoothing':0},
        'velocity':{'rawness':'normal','value_smoothing':3,'source_smoothing':0},
        'speed':{'rawness':'normal','value_smoothing':3,'source_smoothing':0},
    }

    DEFAULT_ANALYSIS_RAW = {
        'gain':{'rawness':'raw','value_smoothing':3,'source_smoothing':0},
        'velocity':{'rawness':'raw','value_smoothing':3,'source_smoothing':0},
        'speed':{'rawness':'raw','value_smoothing':3,'source_smoothing':0},
    }

    def _translate_accession_str(accession_str:str) -> dict:
        as_list = [int(x) for x in accession_str]
        accession_info = {k:v for k,v in zip(('pathology_class','patient_index','visit_idx','axis_idx','file_idx'))}

    def _find_jumps(target_list:list, direction:str='both', debug=-1) -> list[tuple[str,int]]:
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
            jumps_ids = [(q>0 and 'positive' or 'negative',i) 
                         for i in jumps_ids if abs(q:=(target_list[i]-target_list[i-1])) >= 15]
        
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
    
    def _find_gain(position_series:list, target_series:list, do_clamp=True) -> list:
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
    
    def _longest_continuous_subsequence(series):
        """Find longest continuous subsequence using start and length."""
        if not series:
            return []
        
        # Best sequence tracking
        best_start = 0
        best_len = 1
        
        # Current sequence tracking
        curr_start = 0
        curr_len = 1
        
        for i in range(1, len(series)):
            if series[i] == series[i-1] + 1:
                curr_len += 1
            else:
                if curr_len > best_len:
                    best_start = curr_start
                    best_len = curr_len
                curr_start = i
                curr_len = 1
        
        # Final check
        if curr_len > best_len:
            best_start = curr_start
            best_len = curr_len
        
        return [best_start, best_start + best_len]
    
    def _find_calibration(target_list) -> dict:
        calibration_info = {
            'exists': False,
            'length': -1,
            'indices': []
        }

        calibration_bundle = [idx for idx, val in enumerate(target_list) if abs(val)<=0.5]

        calibration_indices = Accession.__longest_continuous_subsequence(calibration_bundle)

        if len(calibration_indices) < 2:
            return calibration_info
        
        if calibration_indices[1]-calibration_indices[0] < 70:
            return calibration_info

        calibration_info['exists'] = True
        calibration_info['length'] = calibration_indices[1]-calibration_indices[0]
        calibration_info['indices'] = calibration_indices

        return calibration_info
    
    def _find_energy_power(series:list, max_time:float):
        energy = sum([x*x for x in series])
        power = energy/max_time
        return energy, power

    def _analyse_jumps(jump_ids) -> dict:
        first_cycle_direction = jump_ids[0][0]
        cycles = [idx for direction,idx in jump_ids 
                    if direction == first_cycle_direction]
        cycles = [(cycles[idx], cycles[idx+1]) for idx in range(len(cycles)-1)]

        return {'first_cycle_direction':first_cycle_direction, 'cycles':cycles}

    def _analyse_position(df, sides_of_interest, calibration_info) -> tuple[dict]:
        position_dict = {}
        calibration_dict = {}
        
        for side in sides_of_interest:
            if side == 'AVG':
                series = pointwise_mean(list(position_dict.values()))
                side = side
            else:
                series = list(df[side])
                side = side[0]
            
            calibration_indices = calibration_info['indices'] # beginning and ending index of calibration pattern

            if len(calibration_indices) < 2: # if no pattern is found, don't use calibration
                calibration_offset = 0
            else: # mean of calibration is offset, subtract from position
                calibration_pattern = series[calibration_indices[0]:calibration_indices[1]]
                calibration_offset = sum(calibration_pattern)/len(calibration_pattern)

            position_dict[side] = [v-calibration_offset for v in series]
            calibration_dict[side] = calibration_offset

        return position_dict, calibration_dict
    
    def _get_basic_info_from_df(input_df:DataFrame, axis:str) -> dict:
        sides_of_interest = [f'{x}{axis[0].upper()}' for x in 'LR'] + ['AVG']
        time_list = list(input_df['Time(sec)'])

        target_list = list(input_df[f'Target{axis[0].upper()}'])
        calibration_info = Accession._find_calibration(target_list)

        jump_ids = Accession._find_jumps(target_list, 'both')
        processed_jumps = Accession._analyse_jumps(jump_ids)

        position_dict, calibration_dict = Accession._analyse_position(input_df, 
                                                    sides_of_interest, 
                                                    calibration_info)
        
        output_dict = {
            'sides_of_interest':sides_of_interest,
            'time_list':time_list,
            'target_list':target_list,
            'calibration_info':calibration_info,
            'jump_ids':jump_ids,
            **processed_jumps,
            'position_dict':position_dict,
            'calibration_dict':calibration_dict
        }

        return output_dict
    
    def _get_kinematics(position_dict:dict, time_list:list, target_list:list, analysis_attrs:dict) -> dict:
        output_dict = {}

        kinematic_attrs = [kine_attr 
                           for kine_attr in analysis_attrs 
                           if kine_attr in Accession.KINEMATIC_ATTRIBUTES]

        for kine_attr in Accession.KINEMATIC_ATTRIBUTES:
            if kine_attr == 'position':
                continue 

            if kine_attr not in kinematic_attrs:
                for a_attr in ('dict','source_smoothing','value_smoothing','is_normal'):
                    attr_name = f'{kine_attr}_{a_attr}'
                    output_dict[attr_name] = None
                    # setattr(self,f'{kine_attr}_{a_attr}',getattr(self,f'{kine_attr}_{a_attr}',None))
                continue

            attr_info : dict = analysis_attrs[kine_attr]
            attr_source_smoothing = attr_info.get('source_smoothing',0)
            attr_value_smoothing = attr_info.get('value_smoothing',0)
            attr_rawness = attr_info.get('rawness','raw')

            if kine_attr == 'velocity':
                attr_dict = {k:rate_of_change(rolling_average(v,attr_source_smoothing)) 
                             for k,v in position_dict.items()}
                
            if kine_attr == 'speed':
                attr_dict = {k:[abs(x) for x in rate_of_change(rolling_average(v,attr_source_smoothing))] 
                             for k,v in position_dict.items()}
                
            if kine_attr == 'gain':
                attr_clamp = attr_info.get('do_clamp',True)
                attr_dict = {k:Accession._find_gain(rolling_average(v,attr_source_smoothing), target_list, attr_clamp) 
                             for k,v in position_dict.items()}
                
            if kine_attr == 'acceleration':
                attr_dict = {k:rate_of_change(rolling_average(rate_of_change(v),attr_source_smoothing))
                            for k,v in position_dict.items()}
            
            attr_dict = {k:rolling_average(v,attr_value_smoothing) for k,v in attr_dict.items()}

            if attr_rawness == 'normal':
                attr_dict = {k:percentile_normalise(v) for k,v in attr_dict.items()}

            for a_attr,k_attr in zip(('dict','source_smoothing','value_smoothing','is_normal'),
                                     (attr_dict,attr_source_smoothing,attr_value_smoothing,attr_rawness)):
                # setattr(self,f'{kine_attr}_{a_attr}',k_attr)
                attr_name = f'{kine_attr}_{a_attr}'
                output_dict[attr_name] = k_attr
        
        energy_dict = {}
        power_dict = {}

        #   GET ENERGY AND POWER
        for kine in Accession.KINEMATIC_ATTRIBUTES:
            kine_info = output_dict.get(f'{kine}_dict',None)
            if kine_info is None:
                continue

            tmp_energy_dict = {}
            tmp_power_dict = {}

            for k,v in kine_info.items():
                tmp_energy_dict[k], tmp_power_dict[k] = Accession._find_energy_power(v, time_list[-1])

            energy_dict[kine] = tmp_energy_dict
            power_dict[kine] = tmp_power_dict

        output_dict['energy_dict'] = energy_dict
        output_dict['power_dict'] = power_dict

        return output_dict
    
    def _standardise_df(input_df_path) -> DataFrame:
        df = read_csv(input_df_path, encoding = CSV_ENCODING)
        df.columns = df.columns.str.replace(' ','')
        return df
    
    def analyse_from_df(self):
        return

    def analyse(self,
                analysis_attrs:dict[dict]=None,
                save_analysis = True
                ):
        """Analyse accession for rendering and other usage
        default analyses position, time_list, and target_list
        can use Accession.DEFAULT_ANALYSIS

        Analysis Attributes:
        - gain:
            - rawness: 'raw' OR 'normal', default 'normal'
            - do_clamp: True OR False, default True
            - source_smoothing: int, default 0
            - value_smoothing: int,  default 3

        - velocity:
            - rawness: 'raw' OR 'normal', default 'normal'
            - source_smoothing: int, default 0
            - value_smoothing: int,  default 3

        - speed
            - rawness: 'raw' OR 'normal', default 'normal'
            - source_smoothing: int, default 0
            - value_smoothing: int,  default 3

        - acceleration
            - rawness: 'raw' OR 'normal', default 'normal'
            - source_smoothing: int, default 0, smoothens velocity
            - value_smoothing: int,  default 3

        - saccades (NOT IMPLEMENTED)

        Args:
            analysis_attrs (dict[dict]): Look at analysis attributes for more info on available options
        """
        if analysis_attrs is None:
            analysis_attrs = dict()

        if not hasattr(self,'accession_path'):
            #dont try to read...
            pass

        df = Accession._standardise_df(self.accession_path)

        self.analysis_info = {
            'has':False,
            'info': {}
        }

        basic_info = Accession._get_basic_info_from_df(df, self.axis)
        output_dict = Accession._get_kinematics(
            basic_info['position_dict'], basic_info['time_list'], basic_info['target_list'], analysis_attrs
        )

        output_dict |= basic_info

        del df
        # gc.collect()

        if not save_analysis:
            return output_dict
        
        self.analysis_info = {
            'has':True,
            'info':output_dict
        }
        return self
        
    def __find_peaks(series, window_size=20, sensitivity=2.0, slope_threshold=0.3):
        """
        Hybrid approach: uses both baseline deviation AND slope changes.
        Detects transient peaks in oscillating signals.
        """
        n = len(series)
        if n < window_size * 2:
            return []
        
        visited = [False] * n
        peaks = []
        
        # Calculate slopes
        slopes = [series[i+1] - series[i] for i in range(n-1)]
        slopes.append(0)  # Pad to match length
        
        for i in range(int(window_size), int(n - window_size)):
            if visited[i]:
                continue
            
            # Calculate local median baseline
            window_size = int(window_size)
            window = series[i-window_size:i+window_size]
            sorted_window = sorted(window)
            median = sorted_window[len(sorted_window) // 2]
            
            # MAD for local spread
            deviations = [abs(x - median) for x in window]
            sorted_deviations = sorted(deviations)
            mad = sorted_deviations[len(sorted_deviations) // 2]
            
            threshold = median + sensitivity * mad
            
            # Check local max
            if series[i] > threshold and series[i] > series[i-1] and series[i] > series[i+1]:
                # Check slope characteristics
                rise_slopes = [slopes[j] for j in range(i-int(window_size//2), i)]
                fall_slopes = [slopes[j] for j in range(i, i+int(window_size//2))]
                
                avg_rise = sum(rise_slopes) / len(rise_slopes) if rise_slopes else 0
                avg_fall = sum(fall_slopes) / len(fall_slopes) if fall_slopes else 0
                
                # Must have a significant rise and fall
                if avg_rise > slope_threshold and avg_fall < -slope_threshold:
                    # Find exact boundaries
                    start_idx = i
                    end_idx = i
                    
                    # Find where it started rising
                    for j in range(i-1, max(0, i-window_size), -1):
                        if series[j] < median + 0.5 * mad:
                            start_idx = j
                            break
                    
                    # Find where it returned
                    for j in range(i+1, min(n, i+window_size)):
                        if series[j] < median + 0.5 * mad:
                            end_idx = j
                            break
                    
                    # Calculate rise and fall magnitudes
                    rise = series[i] - series[start_idx] if start_idx < i else 0
                    fall = series[i] - series[end_idx] if end_idx > i else 0
                    
                    # Only accept if both rise and fall are significant
                    if rise > sensitivity * mad and fall > sensitivity * mad:
                        peak_indices = list(range(start_idx, end_idx + 1))
                        for idx in peak_indices:
                            visited[idx] = True
                        
                        peaks.append({
                            'index': i,
                            'value': series[i],
                            'start': start_idx,
                            'end': end_idx,
                            'rise': rise,
                            'fall': fall,
                            'rise_slope': avg_rise,
                            'fall_slope': avg_fall,
                            'width': end_idx - start_idx + 1,
                            'indices': peak_indices,
                            'values': [series[idx] for idx in peak_indices],
                            'baseline': median,
                            'threshold': threshold
                        })
        
        return peaks

    def __do_process_analysis(use_analysis:dict, 
                              standard_cycle_length:int, 
                              standard_cycle_duration:float) -> tuple[dict]:

        valid_cycles = [(start,end) for start,end in use_analysis.get('cycles')
                           if (end - start - 1) >= standard_cycle_length]
        
        standard_cycles = [(start,start+standard_cycle_length)
                            for start in range(0,
                                                len(valid_cycles) * standard_cycle_length, 
                                                standard_cycle_length)]

        standard_dict = {
            'standard_cycles':standard_cycles
        }

        standard_analysis = {}

        for cycle_idx, (start, end) in enumerate(valid_cycles):
            standard_dict['actual_time_list'] = standard_dict.setdefault('actual_time_list',[])\
                                            + transfer_indices(use_analysis.get('time_list')[start:end], standard_cycle_length)
            standard_dict['target_list'] = standard_dict.setdefault('target_list',[])\
                                            + transfer_indices(use_analysis.get('target_list')[start:end], standard_cycle_length)
            
            for kine_attr in Accession.KINEMATIC_ATTRIBUTES:
                kine_name = f'{kine_attr}_dict'
                kine_info = use_analysis.get(kine_name)

                if kine_info is None:
                    continue
                
                standard_dict[kine_name] = standard_dict.setdefault(kine_name,{})
                data_kine:dict = standard_dict[kine_name]
                
                standard_analysis[kine_attr] = standard_analysis.setdefault(kine_attr,{})
                analysis_kine:dict = standard_analysis[kine_attr]

                for attr_k, attr_v in kine_info.items():
                    cycle =  transfer_indices(attr_v[start:end], standard_cycle_length)
                    # print(f'data_kine {data_kine}')
                    data_kine[attr_k] = data_kine.setdefault(attr_k,[])\
                                        + cycle
                    
                    analysis_mean = sum(cycle)/standard_cycle_length
                    analysis_var = sum([(x-analysis_mean)**2 for x in cycle])/standard_cycle_length
                    analysis_sorted = sorted(cycle)
                    analysis_percentiles = {int(k): analysis_sorted[int(standard_cycle_length*(k/100))]
                                            for k in (5,10,20,25,30,40,50,60,70,80,90,95)}
                    analysis_median = analysis_percentiles[50]
                    analysis_energy, analysis_power = Accession.__find_energy_power(cycle, standard_cycle_duration)
                    analysis_region = cycle_idx * standard_cycle_duration
                    analysis_region = (analysis_region, analysis_region + standard_cycle_duration)

                    # analysis_peaks = Accession.__find_peaks(cycle)

                    analysis = {
                        'cycle_idx':cycle_idx,
                        'cycle_region':analysis_region,
                        'max':max(cycle),
                        'min':min(cycle),
                        'median':analysis_median,
                        'percentiles':analysis_percentiles,
                        'mean':analysis_mean,
                        'variance':analysis_var,
                        'std':analysis_var**0.5,
                        'energy':analysis_energy,
                        'power':analysis_power,
                        # 'peaks':analysis_peaks
                    }
                    analysis_kine[attr_k] = analysis_kine.setdefault(attr_k,[]) + [analysis]

        standard_dict['time_list'] = float_range(0,1/120*(len(standard_dict['actual_time_list'])-1),1/120)

        #   GET ENERGY AND POWER
        standard_energy_dict = {}
        standard_power_dict = {}
        
        for kine in Accession.KINEMATIC_ATTRIBUTES:
            kine_info = standard_dict.get(f'{kine}_dict',None)
            if kine_info is None:
                continue

            energy_dict = {}
            power_dict = {}

            for k,v in kine_info.items():
                energy_dict[k], power_dict[k] = Accession.__find_energy_power(v, standard_dict['time_list'][-1])

            standard_energy_dict[kine] = energy_dict
            standard_power_dict[kine] = power_dict 

        standard_dict['power_dict'] = standard_power_dict
        standard_dict['energy_dict'] = standard_energy_dict

        return standard_dict, standard_analysis

    def process_analysis(self, save_processed:bool = True):
        """Stitches cycles in a recording to be equal in length to the standard cycle length
        for a type of recording
        """
        if not getattr(self,'analysis_info',{'has':False})['has']:
            print('DO ANALYSIS FIRST!')
            return
        
        self.processed_info = {
            'has':False,
            'info':{},
            'analysis':{}
        }

        use_analysis : dict = self.analysis_info['info']

        standard_dict, standard_analysis = Accession.__do_process_analysis(use_analysis, 
                                                                           self.standard_cycle_length,
                                                                           self.standard_cycle_duration)
        
        if not save_processed:
            return standard_dict, standard_analysis
        
        self.processed_info['has'] = True
        self.processed_info['info'] = standard_dict
        self.processed_info['analysis'] = standard_analysis
        
        return self
    
    def draw(self,
             attributes_to_draw:list=[], 
             region:tuple[float|int]=None,
             tick_dist:float=None,
             use_standard:bool=False,
             figsize = None,
             dpi = 72,
             save_name = '',
             save_dir = '',
             do_draw = True):
        """_summary_

            Available Attributes:
            - position
            - gain
            - velocity
            - speed
            - jump

            Args:
                attributes_to_draw (list): Draw Attributes, if left blank only draws target
                region (tuple[float | int], optional): _description_. Defaults to None.
                tick_dist (float, optional): _description_. Defaults to 1.0.

            Returns:
                _type_: _description_
        """

        if tick_dist is None:
            tick_dist = self.standard_cycle_duration
        
        if region is None:
            region = (None, None)

        default_backend = matplotlib.get_backend()

        plt.ion()
        if not do_draw:
            plt.ioff()
            matplotlib.use('agg')

        info_dictionary = self.processed_info['info'] if use_standard else self.analysis_info['info']

        time_list = info_dictionary['time_list']
        target_list = info_dictionary['target_list']

        def determine_region(region_tuple):
            max_end = time_list[-1]
            start = region_tuple[0]
            end = region_tuple[1]

            start = 0.0 if start is None else clamp(start,0,max_end)
            end = max_end if end is None else clamp(end,0,max_end)

            return (start, clamp(start+end,0,max_end))

        region = determine_region(region)

        idp_attr = [x for x in attributes_to_draw 
                        if x in self.KINEMATIC_ATTRIBUTES]
        attr_amt = len(idp_attr)
        if not attr_amt:
            print('NEED ATTRIBUTES TO DRAW!')
            return
        
        if figsize is None:
            figsize = (24,attr_amt*6)

        fig, axs = plt.subplots(nrows = attr_amt,
                                figsize = figsize, 
                                dpi=dpi, 
                                sharex=True, 
                                constrained_layout=True)

        try:
            axs[0]
        except TypeError:
            axs = [axs]
        axs : list[plt.Axes]

        figure_title = f'{self.accession[0]:02d}-{self.accession[1]:02d} | ' 
        figure_title += f'Axis: {self.axis} | Freq: {self.frequency} | '
        figure_title += f'Time: {region[0]:03} - {region[1]:03} seconds'
        fig.suptitle(figure_title)

        # if 'jump' in attributes_to_draw:
        #     ax0_title += ' + Jumps'
        #     for jump_dir, jump_colour in zip(('positive','negative'),('blue',(0.8,0.52,0.01))):
        #         jump_dirs, _ = zip(*self.jump_ids)
        #         if jump_dir not in jump_dirs:
        #             print(f'skipped {jump_dir}')
        #             continue
        #         jump_time, jump_y = zip(*[(time_list[idx],target_list[idx]) 
        #                                   for _,idx 
        #                                   in self.jump_ids if _ == jump_dir])
        #         axs[0].scatter(jump_time, jump_y, color = jump_colour, zorder=3)
            
        # axs[0].set_title(ax0_title)

        lr_colour = [
            (0.303,0.322,0.865), # LEFT
            (0.263,0.845,0.321), # RIGHT
            (0.945,0.121,0.221), # AVG
        ]

        y_data = {
                'position':((-20.2,20.2),'Degree'),
                'speed':((0,600),'Deg/Sec'),
                'velocity':((-600,600),'Deg/Sec'),
                'gain':((-0.25,1.25),''),
                'acceleration':((-1000,1000),'Deg Sec^-2')
        }

        for attr_idx, attribute in enumerate(idp_attr, start=0):
            ax = axs[attr_idx]

            # DRAW ATTRIBUTE
            data = info_dictionary.get(f'{attribute}_dict')
            
            y_lim, y_unit = y_data[attribute]

            if self.analysis_info['info'].get(f'{attribute}_is_normal','raw') == 'raw':
                ax.set_ylim(*y_lim)
            else:
                ax.set_ylim(-0.05,1.05)
            
            for colour, (k, v) in zip(lr_colour, data.items()):
                ax.plot(time_list, v, label=f"{attribute}_{k}", color=colour)

            # DRAW TARGET
            ax2 = ax.twinx()
            ax2.plot(time_list,target_list,linestyle=":",color=(0.666,0.646,0.965),linewidth=3.5,label='Target',zorder=30)
            ax2.set_ylim(-20.2,20.2)
            ax2.set_ylabel('Target (Deg)')
            ax2.legend(loc='lower right')

            ax.set_ylabel(y_unit)
            ax.grid(True),
            ax.set_title(attribute)
            ax.legend(loc='lower left')

            ax.set_xticks(float_range(*region, tick_dist))
            ax.set_xlim(*region)

        if len(save_name) and len(save_dir):
            plt.savefig(os.path.join(save_dir,f'{save_name}.jpg'),bbox_inches='tight')
        
        if not do_draw:
            plt.close('all')
            del fig, axs
            gc.collect(generation=2)
            matplotlib.use(default_backend)

            return
        
        plt.show()

    def __repr__(self):
        big_divider = "="*70 + "\n"
        small_divider = "-"*70 + "\n"

        # BASIC INFO

        rep = big_divider\
                + f"accession: {self.accession}\n"\
                + f"rel_path: {self.rel_path}\n"
        
        rep += small_divider\
                + f"patient_name: {self.patient_name}\n"\
                + f"pathology_class: {self.pathology_class}\n"\
                + f"visit_idx: {self.visit_idx}\n"\
                + f"visit_date: {self.visit_date}\n"\
                + f"axis: {self.axis}\n"\
                + f"frequency: {self.frequency}\n"\
                + f"standard_cycle_length: {self.standard_cycle_length}\n"
        
        # ANALYSIS STUFF
        
        has_analysis = getattr(self,'analysis_info',{'has':False})['has']
        rep += big_divider + f'has_analysis: {has_analysis}\n'
        if not has_analysis:
            rep += 'run Accession.analyse for more info and standardisation\n' + big_divider
            return rep
        
        rep += "use Accession.analysis_info['info'] (dict) with the following keys to access!\n"\
                + small_divider
        
        use_analysis:dict = self.analysis_info['info']

        rep += f"sides_of_interest: {use_analysis.get('sides_of_interest')}\n"\
                + f"calibration_info: \n\t{'\n\t'.join([f'{k}:{v}' for k,v in use_analysis.get('calibration_info').items()])}\n"
        
        rep += small_divider\
                + f"position_dict: {[(k,len(v)) for k,v in use_analysis.get('position_dict').items()]}\n"\
                + f"calibration_dict: {[(k,v) for k,v in use_analysis.get('calibration_dict').items()]}\n"\
                + f"time_list: {(use_analysis.get('time_list')[0],use_analysis.get('time_list')[-1])}\n"\
                + f"target_list (number of samples): {len(use_analysis.get('target_list'))}\n"
        
        rep += small_divider\
                + f"jump_ids: {len(use_analysis.get('jump_ids'))}\n"\
                + f"first_cycle_direction: {use_analysis.get('first_cycle_direction')}\n"\
                + f"cycles: {len(use_analysis.get('cycles'))}\n"
        
        for kinematic in self.KINEMATIC_ATTRIBUTES:
            if kinematic == 'position':
                continue
            kinematic_msg = use_analysis.get(f'{kinematic}_dict')
            kinematic_msg = None if kinematic_msg is None else [(k,len(v)) for k,v in kinematic_msg.items()]
            rep += small_divider\
                    + f"{kinematic}_dict: {kinematic_msg}\n"\
                    + f"{kinematic}_is_normal: {use_analysis.get(f'{kinematic}_is_normal')}\n"\
                    + f"{kinematic}_value_smoothing: {use_analysis.get(f'{kinematic}_value_smoothing')}\n"\
                    + f"{kinematic}_source_smoothing: {use_analysis.get(f'{kinematic}_source_smoothing')}\n"\
                    
        rep += small_divider\
                + f"energy_dict: \n\t{'\n\t'.join([f'{k}:{list(v.items())}' for k,v in use_analysis.get('energy_dict').items()])}\n"\
                + f"power_dict: \n\t{'\n\t'.join([f'{k}:{list(v.items())} (second^-1)' for k,v in use_analysis.get('power_dict').items()])}\n"\
                
        # PROCESSED STUFF

        has_standard = getattr(self,'processed_info', {'has':False})['has']
        rep += big_divider + f'has_standard: {has_standard}\n'
        if not has_standard:
            rep += 'run Accession.process_analysis to standardise recording\n' + big_divider
            return rep
        
        rep += "use Accession.processed_info['info'] (dict) with the following keys for more info\n"\
            + small_divider
        
        use_processed = self.processed_info['info']
        
        rep += f'standard_cycles: {len(use_processed.get('standard_cycles'))}\n'\
                + f"position_dict: {[(k,len(v)) for k,v in use_processed.get('position_dict').items()]}\n"\
                + f"time_list: {(use_processed.get('time_list')[0],use_processed.get('time_list')[-1])}\n"\
                + f"target_list (number of samples): {len(use_processed.get('target_list'))}\n"+small_divider

        for kinematic in self.KINEMATIC_ATTRIBUTES:
            kinematic_msg = use_processed.get(f'{kinematic}_dict', None)
            kinematic_msg = None if kinematic_msg is None else [(k,len(v)) for k,v in kinematic_msg.items()]
            rep += f'{kinematic}_dict: {kinematic_msg}\n'\
            
        rep += small_divider\
                + f"energy_dict: \n\t{'\n\t'.join([f'{k}:{list(v.items())}' for k,v in use_processed.get('energy_dict').items()])}\n"\
                + f"power_dict: \n\t{'\n\t'.join([f'{k}:{list(v.items())} (second^-1)' for k,v in use_processed.get('power_dict').items()])}\n"\
        
        use_processed_analysis = self.processed_info['analysis']
        rep += small_divider\
                + f'standard_analysis: \n\t{'\n\t'.join([f'{k}: {list(v.keys())} ({len(v['AVG'])}) -> {list(v['AVG'][0].keys())}' 
                                                         for k,v in use_processed_analysis.items()])}\n'
                    
        rep += big_divider
        return rep
    
class PreProcessedAccession(Accession):
    INFO_TEMPLATE = {k: None for k in ACCESSION_INFO_KEYS}
    def __init__(self, acc):
        return
    