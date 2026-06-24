import os
import sys
from .utils import get_all_csv
import json

def import_database(database_directory:str,
                    valid_directories:list[str]) -> dict:
    """Import Database from a directory

    Args:
        database_directory (str): The path to the database directory
        valid_directories (list[str]): List of subdirectories that will be accessed

    Returns:
        dict: Database as a hierarchical dictionary in the following structure, lowest value is a path string
    
    Structure:
     Database (dict)
      └─ [pathology_class_key] ('Healthy control', 'Definite MG', Probable MG, etc.) (str)
          └─ [patient_index] (0, 1, 2, etc.) (int)
              ├─ patient_name (str)
              └─ visits (str)
                  └─ [visit_index] (0, 1, 2, etc.) (int)
                      ├─ date (str)
                      └─ files (str)
                          └─ [axis_key] ('horizontal' OR 'vertical') (str)
                               └─ [file_index] (0, 1, or 2 for 0.5Hz, 0.75Hz, and 1.0Hz respectively) (int)
    
    Accessing Example:
     `Database['Healthy control'][0]['visits'][0]['files']['horizontal'][0]`
     Retrieves the Horizontal, 0.5Hz recording from the 1st visit of the 1st patient in 'Healthy control'
    """
    
    subdirs_key = map(lambda x: os.path.join(database_directory, x), valid_directories)

    all_csv = [get_all_csv(subdir) for subdir in subdirs_key]
    json.dump(all_csv, open('all_csv.json','w'), indent=4)


    csv_access = dict()
    _, tail = os.path.split(database_directory)

    def handle_files(l_files:list):
        return {k:sorted([x for x in l_files if k in x.lower()]) 
                for k in ['horizontal','vertical']}

    for dir in all_csv:
        for subdir, content in dir.items():

            subdir_key = subdir.split('/')
            subdir_index = -1 if subdir_key[-2]==tail else -2
            subdir_key = '/'.join(subdir_key[subdir_index:])

            reformat_structure = []
            for patient, visits in content.items():
                new_visit_struct = sorted([{'date':date,'files':handle_files(files)} 
                                    for date,files 
                                    in visits.items()], key=lambda x:x['date'])
                new_patient_struct = {'patient_name':patient,'visits':new_visit_struct}
                reformat_structure.append(new_patient_struct)

            csv_access[subdir_key] = sorted(reformat_structure, key=lambda x:x['visits'][0]['date'])

    return csv_access