import matplotlib.pyplot as plt

from pandas import read_csv

from eva_experiments.utils import clamp, float_range, transfer_indices
from eva_experiments.mg_analysis import kinematics
from eva_experiments.mg_stats.utils import rate_of_change, \
                                            rolling_average,\
                                            percentile_normalise,\
                                            pointwise_mean

CSV_ENCODING = 'utf-16-le'

class Accession:
    KINEMATIC_ATTRIBUTES = ('gain','speed','velocity')

    def __init__(self, accession_path:str, accession_info:dict):
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
        self.accession_path = accession_path

        for k,v in accession_info.items():
            setattr(self, k, v)

        self.standard_cycle_length = [480,360,240][self.file_idx]

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

    def _process_jumps(self):
        self.first_cycle_direction = self.jump_ids[0][0]
        cycles = [idx for direction,idx in self.jump_ids 
                    if direction == self.first_cycle_direction]
        self.cycles = [(cycles[idx], cycles[idx+1]) for idx in range(len(cycles)-1)]
        return

    def _process_position(self, df):
        self.position_dict = {}
        for aoi in self.axes_of_interest:
            if aoi == 'AVG':
                self.position_dict[aoi] = pointwise_mean(list(self.position_dict.values()))
                continue
            self.position_dict[aoi] = list(df[aoi])
        return

    def analyse(self,
                analysis_attrs:dict[dict]=None
                ):
        """Analyse accession for rendering and other usage
        default analyses position, time_list, and target_list
        can use Accession.DEFAULT_ANALYSIS

        Analysis Attributes:
        - gain:
            - rawness: 'raw' OR 'normal', default 'normal'
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

        - saccades (NOT IMPLEMENTED)

        Args:
            analysis_attrs (dict[dict]): Look at analysis attributes for more info on available options
        """
        if getattr(self, 'accession_data', None) is None:
            accession_data = read_csv(self.accession_path, encoding = CSV_ENCODING)
            accession_data.columns = accession_data.columns.str.replace(' ','')
            self.accession_data = accession_data

        if analysis_attrs is None:
            analysis_attrs = dict()

        df = self.accession_data

        self.axes_of_interest = [f'{x}{self.axis[0].upper()}' for x in 'LR'] + ['AVG']

        self.time_list = list(df['Time(sec)'])
        self.target_list = list(df[f'Target{self.axis[0].upper()}'])
        self.jump_ids = kinematics.find_jumps(self.target_list, 'both')
        self._process_jumps()
        self._process_position(df)

        kinematic_attrs = [kine_attr 
                           for kine_attr in analysis_attrs 
                           if kine_attr in self.KINEMATIC_ATTRIBUTES]

        for kine_attr in self.KINEMATIC_ATTRIBUTES:
            if kine_attr not in kinematic_attrs:
                for a_attr in ('dict','source_smoothing','value_smoothing','is_normal'):
                    setattr(self,f'{kine_attr}_{a_attr}',getattr(self,f'{kine_attr}_{a_attr}',None))
                continue

            attr_info : dict = analysis_attrs[kine_attr]
            attr_source_smoothing = attr_info.get('source_smoothing',0)
            attr_value_smoothing = attr_info.get('value_smoothing',0)
            attr_rawness = attr_info.get('rawness','raw')

            if kine_attr == 'velocity':
                attr_dict = {k:rate_of_change(rolling_average(v,attr_source_smoothing)) 
                             for k,v in self.position_dict.items()}
                
            if kine_attr == 'speed':
                attr_dict = {k:[abs(x) for x in rate_of_change(rolling_average(v,attr_source_smoothing))] 
                             for k,v in self.position_dict.items()}
                
            if kine_attr == 'gain':
                attr_dict = {k:kinematics.find_gain(rolling_average(v,attr_source_smoothing), self.target_list) 
                             for k,v in self.position_dict.items()}
            
            attr_dict = {k:rolling_average(v,attr_value_smoothing) for k,v in attr_dict.items()}

            if attr_rawness == 'normal':
                attr_dict = {k:percentile_normalise(v) for k,v in attr_dict.items()}

            for a_attr,k_attr in zip(('dict','source_smoothing','value_smoothing','is_normal'),
                                     (attr_dict,attr_source_smoothing,attr_value_smoothing,attr_rawness)):
                setattr(self,f'{kine_attr}_{a_attr}',k_attr)


        self.has_analysis = True

        # saccade_idx = None
        # if id_saccades:
        #     try:
        #         saccade_idx = kinematics.find_saccades2(speed_dict.values(), saccade_threshold)
        #     except AttributeError: #No speed idx
        #         pass
        #     except Exception as e:
        #         print(f'!!! SACCADE FINDING ERROR {e}:\n\taccession: {accession}\n ')
        #         # raise Exception()
        return self

    DEFAULT_DRAWING_ATTRIBUTES = ('position','gain','velocity','speed')

    def draw(self,
             attributes_to_draw:list=[], 
             region:tuple[float|int]=None,
             tick_dist:float=1.0,
             use_standard:bool=False):
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
        
        if region is None:
            region = (None, None)

        def determine_region(region_tuple):
            max_end = self.time_list[-1]
            start = region_tuple[0]
            end = region_tuple[1]

            start = 0.0 if start is None else clamp(start,0,max_end)
            end = max_end if end is None else clamp(end,0,max_end)

            return (start, clamp(start+end,0,max_end))

        region = determine_region(region)

        idp_attr = [x for x in attributes_to_draw 
                        if x in self.DEFAULT_DRAWING_ATTRIBUTES]
        attr_amt = len(idp_attr)+1 # keep 1 for target

        fig, axs = plt.subplots(nrows = attr_amt,
                                figsize=(24,attr_amt*6), 
                                dpi=150, 
                                sharex=True, 
                                constrained_layout=True)

        try:
            axs[0]
        except TypeError:
            axs = [axs]
        axs : list[plt.Axes]

        figure_title = f'{self.accession}' 
        fig.suptitle(figure_title)

        ax0_title = 'Target'
        
        axs[0].set_ylim(-20.2,20.2)
        axs[0].set_xlim(*region)
        axs[0].set_xticks(float_range(*region, tick_dist))
        axs[0].set_ylabel('Degrees')
        axs[0].grid(True)

        axs[0].plot(self.time_list, self.target_list, label='Target List')

        if 'jump' in attributes_to_draw:
            ax0_title += ' + Jumps'
            for jump_dir, jump_colour in zip(('positive','negative'),('blue',(0.8,0.52,0.01))):
                jump_dirs, _ = zip(*self.jump_ids)
                if jump_dir not in jump_dirs:
                    print(f'skipped {jump_dir}')
                    continue
                jump_time, jump_y = zip(*[(self.time_list[idx],self.target_list[idx]) 
                                          for _,idx 
                                          in self.jump_ids if _ == jump_dir])
                axs[0].scatter(jump_time, jump_y, color = jump_colour, zorder=3)
            
        axs[0].set_title(ax0_title)

        lr_colour = [
            (0.303,0.322,0.765),
            (0.263,0.845,0.321),
            (0.745,0.686,0.221)
        ]

        for attr_idx, attribute in enumerate(idp_attr, start=1):
            ax = axs[attr_idx]

            if attribute == 'position':
                data = self.position_dict
                ylim = (-20.2,20.2)
            else:
                ylim = (-0.05,1.05)
            if attribute == 'velocity':
                data = self.velocity_dict
            if attribute == 'speed':
                data = self.speed_dict
            if attribute == 'gain':
                data = self.gain_dict
            
            for colour, (k, v) in zip(lr_colour, data.items()):
                ax.plot(self.time_list, v, label=f"{attribute}_{k}", color=colour)
            
            ax.grid(True),
            ax.set_ylim(*ylim)
            ax.set_title(attribute)
            ax.legend()

        plt.show()

    def standardise_cycles(self):
        """Stitches cycles in a recording to be equal in length to the standard cycle length
        for a type of recording
        """
        if not hasattr(self, 'has_analysis'):
            print('DO ANALYSIS FIRST!')
            return
        
        valid_cycles = [(start,end) for start,end in self.cycles
                           if (end - start - 1) >= self.standard_cycle_length]
        
        self.standard_cycles = [(start,start+self.standard_cycle_length+1)
                                for start in range(0,
                                                   len(valid_cycles)*self.standard_cycle_length + 1, 
                                                   self.standard_cycle_length+1)]

        standard_dict = {}
        self.standard_analysis = {}

        for start, end in valid_cycles:
            standard_dict['time_list'] = standard_dict.setdefault('time_list',[])\
                                            + transfer_indices(self.time_list[start:end], self.standard_cycle_length)
            standard_dict['target_list'] = standard_dict.setdefault('target_list',[])\
                                            + transfer_indices(self.target_list[start:end], self.standard_cycle_length)
            
            for kine_attr in self.KINEMATIC_ATTRIBUTES + ('position',):
                kine_name = f'{kine_attr}_dict'
                kine_info = getattr(self,kine_name)

                if kine_info is None:
                    continue
                
                standard_dict[kine_name] = standard_dict.setdefault(kine_name,{})
                data_kine:dict = standard_dict[kine_name]
                
                self.standard_analysis[kine_attr] = self.standard_analysis.setdefault(kine_attr,{})
                analysis_kine:dict = self.standard_analysis[kine_attr]

                for attr_k, attr_v in kine_info.items():
                    cycle =  transfer_indices(attr_v[start:end], self.standard_cycle_length)
                    # print(f'data_kine {data_kine}')
                    data_kine[attr_k] = data_kine.setdefault(attr_k,[])\
                                        + cycle
                    
                    analysis_mean = sum(cycle)/self.standard_cycle_length
                    analysis_var = sum([(x-analysis_mean)**2 for x in cycle])/self.standard_cycle_length
                    analysis_sorted = sorted(cycle)
                    analysis_percentiles = {int(k): analysis_sorted[int(self.standard_cycle_length*(k/100))]
                                            for k in (5,10,20,25,30,40,50,60,70,80,90,95)}
                    analysis_median = analysis_percentiles[50]
                    analysis = {
                        'max':max(cycle),
                        'min':min(cycle),
                        'median':analysis_median,
                        'percentiles':analysis_percentiles,
                        'mean':analysis_mean,
                        'variance':analysis_var,
                        'std':analysis_var**0.5
                    }
                    analysis_kine[attr_k] = analysis_kine.setdefault(attr_k,[]) + [analysis]

        for k, v in standard_dict.items():
            setattr(self, f'standard_{k}', v)

        self.has_standard = True
        return self

    def __repr__(self):
        big_divider = "="*70 + "\n"
        small_divider = "-"*70 + "\n"

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
        
        has_analysis = getattr(self,'has_analysis',False)
        rep += big_divider + f'has_analysis: {has_analysis}\n'
        if not has_analysis:
            rep += 'run Accession.analyse for more info and standardisation\n' + big_divider
            return rep
        
        rep += f"axes_of_interest: {self.axes_of_interest}\n"
        
        rep += small_divider\
                + f"position_dict: {[(k,len(v)) for k,v in self.position_dict.items()]}\n"\
                + f"time_list: {(self.time_list[0],self.time_list[-1])}\n"\
                + f"target_list (number of samples): {len(self.target_list)}\n"
        
        rep += small_divider\
                + f"jump_ids: {len(self.jump_ids)}\n"\
                + f"first_cycle_direction: {self.first_cycle_direction}\n"\
                + f"cycles: {len(self.cycles)}\n"
        
        for kinematic in self.KINEMATIC_ATTRIBUTES:
            kinematic_msg = getattr(self, f'{kinematic}_dict')
            kinematic_msg = None if kinematic_msg is None else [(k,len(v)) for k,v in kinematic_msg.items()]
            rep += small_divider\
                    + f"{kinematic}_dict: {kinematic_msg}\n"\
                    + f"{kinematic}_is_normal: {getattr(self,f'{kinematic}_is_normal')}\n"\
                    + f"{kinematic}_value_smoothing: {getattr(self,f'{kinematic}_value_smoothing')}\n"\
                    + f"{kinematic}_source_smoothing: {getattr(self,f'{kinematic}_source_smoothing')}\n"

        has_standard = getattr(self,'has_standard',False)
        rep += big_divider + f'has_standard: {has_standard}\n'
        if not has_standard:
            rep += 'run Accession.standardise_cycles to standardise recording\n' + big_divider
            return rep
        
        rep += f'standard_cycles: {len(self.standard_cycles)}\n'\
                + f"standard_position_dict: {[(k,len(v)) for k,v in self.standard_position_dict.items()]}\n"\
                + f"standard_time_list: {(self.standard_time_list[0],self.standard_time_list[-1])}\n"\
                + f"standard_target_list (number of samples): {len(self.standard_target_list)}\n"+small_divider

        for kinematic in self.KINEMATIC_ATTRIBUTES:
            kinematic_msg = getattr(self, f'standard_{kinematic}_dict', None)
            kinematic_msg = None if kinematic_msg is None else [(k,len(v)) for k,v in kinematic_msg.items()]
            rep += f'standard_{kinematic}_dict: {kinematic_msg}\n'

        rep += small_divider\
                + f'standard_analysis: \n\t{'\n\t'.join([f'{k}: {list(v.keys())} ({len(v['AVG'])}) -> {list(v['AVG'][0].keys())}' for k,v in self.standard_analysis.items()])}\n'
                    
        rep += big_divider
        return rep

    # DEFAULT_SPLIT_PARAMS = {
    #     'split_ratio':1/11,
    #     'absolute':5,
    #     'acc_attributes':['gain','speed','velocity'],
    # }
        # def _split_recording(self,
    #                 split_ratio:list=[0.25,0.50],
    #                 absolute:list=None,
    #                 acc_attributes = [
    #                     'position',
    #                     'velocity',
    #                     'speed',
    #                     'target'
    #                 ], 
    #                 do_calc:bool=True):
    #     """Splits a processed accession into sections by jump indices

    #     Args:
    #         accession (list): Accession list
    #         ratio (list, optional): Split ratio. Defaults to [0.25,0.50].
    #         absolute (list, optional): Force specific regions. Defaults to None.
    #         do_calc (bool, optional): do actual averages, for testing set to False. Defaults to True.

    #     Raises:
    #         Exception: if ratio is too weighted

    #     Returns:
    #         if do_calc is false, returns  sections, length of regions, and total section amount
    #         if normal returns averaged section position and averaged target.
    #     """
    #     attribute_access_dict = {
    #         'position', 'velocity', 'speed', 'gain'
    #     }
        
    #     _, raw_jump_ids = zip(*self.jump_ids)

    #     cycle_start_point = self.jump_ids[0][0]
    #     cycle_half_point = cycle_start_point == 'positive' and 'negative' or 'positive'

    #     jumps_indices = [idx 
    #                     for direction,idx 
    #                     in self.jump_ids 
    #                     if direction == cycle_start_point]

    #     attribute_dict = {
    #         k:getattr(self,f'{k}_dict')['AVG']
    #         for k in acc_attributes
    #         if k in attribute_access_dict
    #     }

    #     if 'target' in acc_attributes:
    #         attribute_dict['target'] = self.target_list
        
    #     sections = series_split(jumps_indices,split_ratio)

    #     if absolute is not None:
    #         if isinstance(absolute, int):
    #             absolute = [absolute for i in range(len(sections))]
    #         sections = [section[:absolute[section_idx]] for section_idx, section in enumerate(sections)]
    #         # N = sum(absolute)

    #     output = {k:[] for k in attribute_dict} | {'meta':[len(x) for x in sections],'sections':sections}

    #     if not do_calc:
    #         return output
        
    #     for section in sections:
    #         section_len = len(section)
    #         for attr_key, attr_val in attribute_dict.items():
    #             cycle_starts = []
    #             cycle_ends = []
    #             cycle_wholes = []

    #             # gets the actual region from each attribute list
    #             section_cycles = [
    #                 attr_val[section[sect_idx-1]:section[sect_idx]]
    #                 for sect_idx
    #                 in range(1,section_len)
    #             ]

    #             # for each cycle, split into first half (start)
    #             # and second half (end), and also a complete cycle
    #             for cycle_idx, cycle in enumerate(section_cycles):
    #                 cycle_start = section[cycle_idx]
    #                 cycle_halfway = raw_jump_ids.index(cycle_start) + 1 
    #                 cycle_halfway = raw_jump_ids[cycle_halfway]

    #                 cycle_starts += [[]]
    #                 cycle_ends += [[]]
    #                 cycle_wholes += [[]]

    #                 for point_idx, point in enumerate(cycle, start=cycle_start):
    #                     if point_idx <= cycle_halfway:
    #                         cycle_starts[cycle_idx].append(point)
    #                     if point_idx >= cycle_halfway:
    #                         cycle_ends[cycle_idx].append(point)
    #                     cycle_wholes[cycle_idx].append(point)

    #             for cyc_reg_idx, cycle_region in enumerate((cycle_starts, cycle_ends, cycle_wholes)):
    #                 use_cycle_length = int(self.cycle_length/2) if cyc_reg_idx<2 else self.cycle_length
    #                 valid_cycles = [cycle for cycle in cycle_region if len(cycle)>=use_cycle_length]
    #                 if not len(valid_cycles):
    #                     print([len(cycle) for cycle in cycle_region]) 
    #                     raise ZeroDivisionError()
    #                 valid_cycles_start_avg = sum([cycle[0] for cycle in valid_cycles])/len(valid_cycles)
    #                 valid_cycles_end_avg = sum([cycle[-1] for cycle in valid_cycles])/len(valid_cycles)
    #                 for cycle in cycle_region:
    #                     if len(cycle) >= use_cycle_length:
    #                         continue
    #                     at_start = random.randint(0,1)
    #                     if at_start:
    #                         cycle.insert(0,valid_cycles_start_avg)
    #                         continue
    #                     cycle.append(valid_cycles_end_avg)

    #             data = {
    #                 cycle_start_point : [sum(x)/(section_len-1) for x in zip(*cycle_starts)],
    #                 cycle_half_point : [sum(x)/(section_len-1) for x in zip(*cycle_ends)],
    #                 'whole' : [sum(x)/(section_len-1) for x in zip(*cycle_wholes)]
    #             }
                
    #             output[attr_key].append(data)

    #             # normaliser = output[k]['raw'][0]
    #             # # print(normaliser)
    #             # normalised = {
    #             #     k:[raw/norm for raw, norm in zip(v,normaliser[k])]
    #             #     for k,v in data.items()
    #             # }

    #             # output[k]['normalised'].append(normalised)
    #     return output