import json
import random
from .accession import Accession, PATHOLOGY_CLASSES_DEFAULT, AXIS_KEYS

class AccessionDatabase:
    def __init__(self, Database:dict, accepted_pathologies=PATHOLOGY_CLASSES_DEFAULT):
        """Container for Accessions, links to path

        Args:
            Database (dict): A hierarchical dictionary, use `import_database` from `tools.data_loader`
            accepted_pathologies (tuple[str], optional): A tuple of allowed pathologies, if not given uses:
                - Healthy control
                - Definite MG
        """
        self.DATABASE = {k:Database[k] for k in  tuple(accepted_pathologies)}
        self._process_database()
        # json.dump(self.accessions, open('accessions.json','w'), indent=4)
    
    def _process_database(self):
        accessions=dict()

        for patho_idx, (patho_class, patients) in enumerate(self.DATABASE.items()):
            for patient_idx, patient in enumerate(patients):
                visits = sorted(patient['visits'],key=lambda x:x['date'])
                for visit_idx, visit in enumerate(visits):
                    for axis_idx, (axis_key, axis) in enumerate(visit['files'].items()):
                        for file_idx, file in enumerate(axis):
                            accession = [patho_idx, patient_idx, visit_idx, axis_idx, file_idx]
                            accession_str = '-'.join([f'{a:02d}' for a in accession])
                            accession_info = {
                                'accession': accession,
                                'accession_str':accession_str,
                                'pathology_class':patho_class,
                                'patho_class_idx':patho_idx,
                                'patient_name':patient['patient_name'],
                                'patient_idx':patient_idx,
                                'visit_date':visit['date'],
                                'visit_idx':visit_idx,
                                'axis':axis_key,
                                'axis_idx':axis_idx,
                                'rel_path':file,
                                'file_idx':file_idx,
                                'frequency':('0.5Hz','0.75Hz','1.0Hz')[file_idx]
                            }
                            accession_obj = Accession(file,accession_info)
                            accessions[accession_str] = accession_obj

        self._indices = list(accessions.keys())
        self._len = len(self._indices)
        self.accessions = accessions

    def __getitem__(self, key) -> Accession:
            if not isinstance(key, (list, tuple)):
                return self.accessions[self._indices[key]]
            if len(key)==5:
                return self.accessions['-'.join(f'{x:02d}' for x in key)]
            else:
                raise ValueError("accessing AccessionDatabase requires 1 or 5 indices")

    def get_random_accession(self) -> Accession:
        return self.__getitem__(random.randint(0,self._len))
    
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