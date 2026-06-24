class AccessionDatabase:
    PATHOLOGY_CLASSES_DEFAULT = (
        'Healthy control', 'Definite MG'
    )
    AXIS_KEYS = ('horizontal','vertical')

    if accepted_pathologies is None:
        accepted_pathologies = self.PATHOLOGY_CLASSES_DEFAULT[:-1]
