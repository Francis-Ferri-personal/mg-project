from . import kinematics
from pandas import read_csv
from .stats import rate_of_change, rolling_average

class Accession:
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

        self.cycle_length = [480,360,240][self.file_idx]

    DEFAULT_ANALYSIS = {
        'gain':{'smoothing':3},
        'velocity':{'smoothing':3},
        'speed':{'smoothing':3},
        'jumps':{'direction':'both'}
    }

    def analyse(self,
                analysis_attrs:dict[dict]=None
                ):
        """Analyse accession for rendering and other usage
        default analyses position, time_list, and target_list
        can use Accession.DEFAULT_ANALYSIS


        Analysis Attributes:
        - gain:
            - rawness: 'raw' OR 'normal', default 'normal'
            - smoothing: int, default 3

        - velocity:
            - rawness: 'raw' OR 'normal', default 'normal'
            - smoothing: int, default 3

        - speed
            - rawness: 'raw' OR 'normal', default 'normal'
            - smoothing: int, default 3

        - jumps:
            - direction: 'positive', 'negative', or 'both'

        - saccades (NOT IMPLEMENTED)

        Args:
            analysis_attrs (dict[dict]): Look at analysis attributes for more info on available options

        Raises:
            Exception: If finding jumps fail

        Returns:
            dict : dict_keys are
            - accession
            - position_dict
            - velocity_dict
            - speed_dict
            - target_list
            - time_list
            - jump_ids
            - saccade_ids
            - axes_of_interest
        """
        if getattr(self, 'accession_data', None) is None:
            accession_data = read_csv(self.accession_path, encoding = 'utf-16-le')
            accession_data.columns = accession_data.columns.str.replace(' ','')
            self.accession_data = accession_data

        if analysis_attrs is None:
            analysis_attrs = dict()

        df = self.accession_data

        axis_label = self.axis
        axes_of_interest = [f'{x}{axis_label[0].upper()}' for x in 'LR']

        self.time_list = list(df['Time(sec)'])
        self.target_list = list(df[f'Target{axis_label[0].upper()}'])

        position_dict = {}
        for aoi in axes_of_interest:
            position_dict[aoi] = list(df[aoi])
        
        position_dict['AVG'] = [(l+r)/2 for l,r in zip(*list(position_dict.values()))]
        axes_of_interest = axes_of_interest + ['AVG']
        
        self.position_dict = position_dict
        self.axes_of_interest = axes_of_interest

        attribute_info: dict
        KINEMATIC_ATTRIBUTES = ('gain','speed','velocity')
        kinematic_attrs = [kine_attr 
                           for kine_attr in analysis_attrs 
                           if kine_attr in KINEMATIC_ATTRIBUTES]

        for kine_attr in KINEMATIC_ATTRIBUTES:
            if kine_attr not in kinematic_attrs:
                for a_attr in ('dict','smoothing','is_normal'):
                    setattr(self,f'{kine_attr}_{a_attr}',None)
                continue

            attr_info : dict = analysis_attrs[kine_attr]
            attr_smoothing = attr_info.get('smoothing',0)
            attr_rawness = attr_info.get('rawness','raw')

            if kine_attr == 'velocity':
                attr_dict = {k:rate_of_change(v) for k,v in position_dict.items()}
            if kine_attr == 'speed':
                attr_dict = {k:[abs(x) for x in rate_of_change(v)] for k,v in position_dict.items()}
            if kine_attr == 'gain':
                attr_dict = {k:kinematics.find_gain(v, self.target_list) for k,v in position_dict.items()}
            
            attr_dict = {k:rolling_average(v,attr_smoothing) for k,v in attr_dict.items()}

            # if attr_rawness == 'normal':
            #     attr_dict = {k:percentile_normalise(v) for k,v in attr_dict.items()}

            for a_attr,k_attr in zip(('dict','smoothing','is_normal'),
                                     (attr_dict,attr_smoothing,attr_rawness)):
                setattr(self,f'{kine_attr}_{a_attr}',k_attr)

        # GET JUMPS
        if 'jumps' in analysis_attrs:
            attribute_info = analysis_attrs['jumps']
            jump_direction = attribute_info.get('direction','both')

            try:
                jump_ids = kinematics.find_jumps(self.target_list,jump_direction)
            except Exception as e:
                print(f'!!! JUMP FINDING ERROR {e}:\n\taccession: {self.accession}\n ')
                raise Exception()
        self.jump_ids = jump_ids
        self.jump_direction = jump_direction

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
    