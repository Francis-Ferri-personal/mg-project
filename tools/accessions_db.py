import json
from .accession import Accession

class AccessionDatabase:

    PATHOLOGY_CLASSES_DEFAULT = ('Healthy control', 'Definite MG')
    AXIS_KEYS = ('horizontal','vertical')

    def __init__(self, Database:dict, accepted_pathologies=PATHOLOGY_CLASSES_DEFAULT):
        """Container for Accessions, links to path

        Args:
            Database (dict): A hierarchical dictionary, use `import_database` from `eva_rewrite.io.data_loader`
            accepted_pathologies (tuple[str], optional): A tuple of allowed pathologies, if not given uses:
                - Healthy control
                - Definite MG
        """
        self.DATABASE = {k:Database[k] for k in  tuple(accepted_pathologies)}
        self._process_database()
        json.dump(self.accessions, open('accessions.json','w'), indent=4)
    
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
                            accessions[accession_str] = accession_info

        self.accessions = accessions

    def get_accession(self,
                      accession:list[int], 
                      **kwargs) -> Accession:
        """Access CSV with accession list, can also randomly selects CSVs

        Accession Order:
        - Pathology class
        - Patient index
        - Visit index
        - Axis
        - Frequency

        Args:
            accession (list): Follow Accession Oorder, if any is None, that value will be randomised

        Raises:
            IndexError: If accession list is not 5
            Exception: Common error with acessing csv
        
        Accession Info Contents:
        - accession (list)
        - accession_str (str)
        - pathology_class (str)
        - patho_class_idx (int)
        - patient_name (str)
        - patient_idx (int)
        - visit_date (str)
        - visit_idx (int)
        - axis (str)
        - axis_idx (int)
        - rel_path (str)
        - file_idx (int)
        - frequency (str)
        """
        if accession is None:
            random_idx = random.randint(0,len(self.accessions))
            accession = list(self.accessions.values())[random_idx]

        if isinstance(accession, list):
            if len(accession) != 5:
                raise IndexError("Accession takes 5 positions!")
            accession = self.accessions['-'.join(f'{a:02d}' for a in accession)]
        elif isinstance(accession, str):
            accession = self.accessions[accession]

        return Accession(accession['rel_path'], accession)