import random
from eva_experiments.mg_io.accession import Accession

class AccessionDatabase:
    PATHOLOGY_CLASSES_DEFAULT = (
        'Healthy control', 'Definite MG', 'Probable MG', 
        'Non-MG diplopia (CNP, etc)/3rd', 'Non-MG diplopia (CNP, etc)/4th', 
        'Non-MG diplopia (CNP, etc)/6th', 'Non-MG diplopia (CNP, etc)/TAO'
    )
    AXIS_KEYS = ('horizontal','vertical')

    def __init__(self, Database:dict):
        """Container for Accessions, links to path

        Args:
            Database (dict): A hierarchical dictionary, use `import_database` from `eva_rewrite.io.data_loader`
        """
        PATHOLOGY_CLASSES_KEYS = sorted(
            [patho_class 
             for patho_class in Database.keys()],
            key=lambda x:self.PATHOLOGY_CLASSES_DEFAULT.index(x)
        )
        self.PATHOLOGY_CLASSES_KEYS = tuple(PATHOLOGY_CLASSES_KEYS)
        self.DATABASE = {k:Database[k] for k in self.PATHOLOGY_CLASSES_KEYS}
        self._prepare_accessions_database()

    def _prepare_accessions_database(self):
        ACCESSIONS = dict()
        RESTRUCT = dict()

        for patho_idx, (patho_class, patients) in enumerate(self.DATABASE.items()):

            for patient_idx, patient in enumerate(patients):
                patient_name = patient['patient_name']
                visits = sorted(patient['visits'],key=lambda x:x['date'])
                for visit_idx, visit in enumerate(visits):
                    visit_date = visit['date']

                    for axis_idx, (axis_key, axis) in enumerate(visit['files'].items()):

                        for file_idx, file in enumerate(axis):
                            frequency =  ('0.5Hz','0.75Hz','1.0Hz')[file_idx]

                            accession = [patho_idx, patient_idx, visit_idx, axis_idx, file_idx]
                            accession_str = '-'.join([f'{a:02d}' for a in accession])
                            accession_info = {
                                'accession': accession,
                                'accession_str':accession_str,
                                'pathology_class':patho_class,
                                'patho_class_idx':patho_idx,
                                'patient_name':patient_name,
                                'patient_idx':patient_idx,
                                'visit_date':visit_date,
                                'visit_idx':visit_idx,
                                'axis':axis_key,
                                'axis_idx':axis_idx,
                                'rel_path':file,
                                'file_idx':file_idx,
                                'frequency':frequency,
                            }
                            accession_obj = Accession(file,accession_info)
                            ACCESSIONS[accession_str] = accession_obj

        self._indices = list(ACCESSIONS.keys())
        self._len = len(self._indices)
        self.accessions = ACCESSIONS

    def __getitem__(self, key) -> Accession:
        if not isinstance(key, (list, tuple)):
            return self.accessions[self._indices[key]]
        if len(key)==5:
            return self.accessions['-'.join(f'{x:02d}' for x in key)]
        else:
            raise ValueError("accessing AccessionDatabase requires 1 or 5 indices")

    def get_random_accession(self) -> Accession:
        return self.__getitem__(random.randint(0,self._len))


    DEFAULT_PATIENT_STRUCTURE_FILTER = {
        'frequency':('0.75Hz',),
        'pathology_class':('Healthy control','Definite MG'),
        'axis':('horizontal','vertical'),
        'standard_cycles':50
    }
    DEFAULT_WANTED_DATA = ['accession_str']
    def get_patient_structure(self,filters:dict,wanted_data:list) -> dict:
        """Gets patient dictionary from pathology class and patient index

        Args:
            filters:
            wanted_data

        Returns:
            dict: _description_
        """
        restructure = dict()
        for k,v in self.accessions.items():
            restruct_key = v.accession_str[:5]
            if v.pathology_class not in filters.get('pathology_class', self.PATHOLOGY_CLASSES_DEFAULT):
                continue

            restruct_top_layer = {
                'patient_name':v.patient_name,
                'pathology_class':v.pathology_class,
                'visits':{}
            }
            restruct_top_layer = restructure.setdefault(restruct_key,restruct_top_layer)
            restruct_top_layer:dict = restruct_top_layer['visits']

            restruct_mid_layer = {
                'visit_date':v.visit_date,
                'frequencies':{}
            }
            restruct_mid_layer:dict = restruct_top_layer.setdefault(f'{v.visit_idx}', restruct_mid_layer)
            restruct_mid_layer = restruct_mid_layer['frequencies']
            if v.frequency not in filters.get('frequency',('0.5Hz','0.75Hz','1.0Hz')):
                continue

            restruct_freq = f'freq_{v.frequency[:-2]}'
            restruct_bottom_layer = {
                'horizontal':{},
                'vertical':{}
            }
            restruct_final_layer = restruct_mid_layer.setdefault(restruct_freq,restruct_bottom_layer)
            restruct_final_layer = restruct_final_layer[v.axis]
            if len(v.standard_cycles) < filters.get('standard_cycles',10):
                continue
 
            for data in wanted_data:
                restruct_final_layer[data] = getattr(v, data)

        stripped = {}
        for patient, patient_stuff in restructure.items():
            for visit in patient_stuff['visits'].values():
                for freq in visit['frequencies'].values():
                    if all([i in freq for i in filters.get('axis',list(freq.keys()))]) :
                        stripped[patient] = patient_stuff
                        continue

        return stripped
    
    def __repr__(self):
        big_divider = "+"*70+'\n'
        small_divider = "-"*70+'\n'
        rep = big_divider + 'ACCESSION DATABASE\n' + big_divider

        patient_total = 0
        visits_total = 0
        files_total = 0
        
        for patho_idx, (patho_class, patients) in enumerate(self.DATABASE.items()):
            visits = sum([len(patient['visits']) for patient in patients])
            files = sum([len(patient['visits'][visit_idx]['files'][axis])
                         for patient in patients
                         for visit_idx in range(len(patient['visits']))
                         for axis in patient['visits'][visit_idx]['files']])
            rep += f' · PATHOLOGY_CLASS ({patho_idx}) : {patho_class}\n'\
                    +f' ├─ Patient Count: {len(patients)}\n'\
                    +f' ├─ Visits Count: {visits}\n'\
                    +f' └─ File Count: {files}\n'\
                    +small_divider
            patient_total += len(patients)
            visits_total += visits
            files_total += files

        rep += f'TOTAL\n'\
                +f' ├─ Patient Count: {patient_total}\n'\
                +f' ├─ Visits Count: {visits_total}\n'\
                +f' └─ File Count: {files_total}\n'\
                +big_divider
        
        return rep