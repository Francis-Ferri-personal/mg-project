import os
import pandas as pd

FREQ_MAP = {
    "0.5": "freq_0.5",
    "0.75": "freq_0.75",
    "1.0": "freq_1.0",
    "1": "freq_1.0",
}

CSV_ENCODING = "utf-16-le"

class MGDataset:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir
        self._files: list[dict] = []

        # dtctionary groups -> patients -> visits 
        self._by_group: dict[str, dict[str, dict[str, list[dict]]]] = {}
        self._index()

    def _index(self):
        groups = sorted(
            d for d in os.listdir(self.raw_dir)
            if os.path.isdir(os.path.join(self.raw_dir, d))
        )

        for group in groups:
            group_dir = os.path.join(self.raw_dir, group)
            for visit_folder in sorted(os.listdir(group_dir)):
                visit_path = os.path.join(group_dir, visit_folder)

                if not os.path.isdir(visit_path):
                    continue

                parts = visit_folder.split(" ", maxsplit=1)
                visit_date = parts[0]
                patient_name = parts[1] if len(parts) > 1 else ""

                for filename in sorted(os.listdir(visit_path)):
                    if not filename.endswith(".csv"):
                        continue

                    fname_lower = filename.lower()
                    if "horizontal" in fname_lower:
                        axis = "horizontal"
                    elif "vertical" in fname_lower:
                        axis = "vertical"
                    else:
                        continue

                    freq = None
                    for key, val in FREQ_MAP.items():
                        if f"({key}hz)" in fname_lower:
                            freq = val
                            break
                    if freq is None:
                        continue

                    file_info = {
                        "path": os.path.join(visit_path, filename),
                        "group": group,
                        "patient": patient_name,
                        "date": visit_date,
                        "axis": axis,
                        "frequency": freq,
                    }

                    self._files.append(file_info)
                    self._by_group.setdefault(group, {})\
                        .setdefault(patient_name, {})\
                        .setdefault(visit_date, [])\
                        .append(file_info)

    @property
    def files(self):
        return self._files

    @property
    def by_group(self):
        return self._by_group

    @property
    def groups(self) -> list[str]:
        return list(self._by_group)

    def patients(self, group: str = None) -> list[str]:
        if group:
            return list(self._by_group.get(group, {}))
        return list({p for g in self._by_group.values() for p in g})

    def visits(self, group: str, patient: str) -> list[str]:
        return list(self._by_group.get(group, {}).get(patient, {}))

    def _from_accession(self, 
        group : int, 
        patient : int = None, 
        date : int = None, 
        axis : int = None, 
        frequency : int = None,
        remove_nones: bool = False,
    ) -> dict[str, str]:

        group_name = list(self.by_group.keys())[group]\
            if isinstance(group, int)\
            else group

        axis_name = ('horizontal','vertical')[axis]\
            if axis in (0,1)\
            else axis

        frequency_name = 'freq_'+('0.5','0.75','1.0')[frequency]\
            if frequency in (0,1,2)\
            else frequency

        as_dict = {
            'group':group_name, 
            'patient':None,
            'date':None, 
            'axis':axis_name, 
            'frequency':frequency_name
        }
        
        access_level = 0
        for level in ('patient', 'date'):
            if isinstance(locals()[level], int):
                access_level += 1
                continue
            break

        if access_level > 0:
            this_group = self.patients(group_name)
            max_patients = len(this_group)
            if patient >= max_patients:
                raise IndexError(f"{group_name} has maximum patient index {max_patients-1}")
            as_dict['patient'] = (patient_name := this_group[patient])

        if access_level > 1:
            this_patient = self.visits(group_name, patient_name)
            max_dates = len(this_patient)
            if date >= max_dates:
                raise IndexError(f"{group_name} | {patient_name} has maximum visit index {max_dates-1}")
            as_dict['date'] = this_patient[date]

        return {k:v for k,v in as_dict.items()
                if not remove_nones and True or v is not None}

    def get_files(self, 
                  group: str | int, 
                  patient: str | int, 
                  date: str |int = None,
                  axis: str |int = None,
                  frequency: str = None) -> list[dict] | dict:
        """Fetches file info(s).
        Exact string and integer indices are interchangable. 
        Fetching is ordered:
            group -> patient -> date -> axis -> frequency
        if a lower order is given but not a higher order, it will return
        down to the lowest consecutive ordering.

        Examples:
            get_files('Healthy control', 0) # returns all files in all visits of the patient
            get_files('Healthy control', 0, 0) # returns all files in the first visit of the patient
            get_files('Healthy control', date=0) # returns empty list because patient is not given
            get_files(0, patient=0, date=0, frequency=0) # returns all files in the same visit date
            get_files(0,0,0,0,0) # returns a single dictionary info

        Args:
            group (str | int): The pathology group name or index, check self.by_group for integer index.
            patient (str | int): The patient name or index.
            date (str | int, optional): Which visit date to fetch. Defaults to None.
            axis (str | int, optional):' 'horizontal' or 'vertical'. Defaults to None.
            frequency (str, optional): 'freq_' + 0.5, 0.75, or 1.0. Defaults to None.

        Returns:
            list[dict] | dict: List of file infos or just one file info
        """

        as_dict = {k:v for k,v in locals().items() 
                   if k in ('group','patient','date','axis','frequency')}
        as_accession = self._from_accession(**as_dict,remove_nones=True)
        as_str = as_dict | as_accession

        group = as_str['group']
        patient = as_str['patient']
        date = as_str['date']
        axis = as_str['axis']
        frequency = as_str['frequency']
        
        patient_visits = self._by_group.get(group, {}).get(patient, {})

        order = (date, axis, frequency)

        deepest_level = 0
        for level in order:
            if level is not None:
                deepest_level += 1
            else:
                break

        if deepest_level == 0:
            return [f for files in patient_visits.values() for f in files]

        patient_visit = patient_visits.get(date, [])
        if deepest_level == 1:
            return patient_visit

        patient_axis = [file for file in patient_visit if file['axis']==axis]
        if deepest_level == 2:
            return patient_axis

        patient_file = [file for file in patient_axis if file['frequency']==frequency]
        if deepest_level == 3:
            return patient_file[0]

    @staticmethod
    def load_csv_of(file_info: dict) -> dict:
        df = pd.read_csv(file_info["path"], encoding=CSV_ENCODING)
        df.columns = df.columns.str.strip()

        axis = file_info["axis"]
        if axis == "horizontal":
            return {
                "time": df["Time(sec)"].tolist(),
                "L": df["LH"].tolist(),
                "R": df["RH"].tolist(),
                "Target": df["TargetH"].tolist(),
                "meta": file_info,
            }
        else:
            return {
                "time": df["Time(sec)"].tolist(),
                "L": df["LV"].tolist(),
                "R": df["RV"].tolist(),
                "Target": df["TargetV"].tolist(),
                "meta": file_info,
            }

    def load_csv(self, key):
        if isinstance(key, slice):
            raise TypeError("key cannot be slice!")
        
        file_info = self[key]
        if not isinstance(file_info, dict):
            raise TypeError("csv key must fetch a single file_info! see self.get_files")
        
        return self.load_csv_of(file_info)

    def __get_by_index(self, key):
        return self._files[key]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.__get_by_index(key)

        if isinstance(key, (tuple, list)):
            order = ('group','patient','date','axis','frequency')
            if len(key) < 5:
                raise IndexError("tuple key must be 5 items long for accession!")
            key = {k:v for k,v in zip(order,key)}

        if isinstance(key, dict):
            try:
                return self.get_files(**key)
            except:
                pass

        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if isinstance(start, int) and isinstance(stop, int):
                return self.files[start:stop]
            if isinstance(start, dict) and isinstance(stop, dict):
                return 

        if isinstance(key, (tuple, list)):
            order = ('group','patient,')

    def __len__(self):
        return len(self._files)

if __name__ == "__main__":
    import argparse
    import sys
    import os

    
    # Add parent directory to path to import from root-level modules
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from tools.visualize import visualize_raw, visualize_cycles, plot_cycles, plot_cycles_comparison
    from datasets.preprocesing.filters import denoise
    from datasets.preprocesing.cycles import get_cycles

    parser = argparse.ArgumentParser(description='Explore MG dataset patient')
    parser.add_argument('--patient', type=int, default=0, help='Patient index (default: 0)')
    parser.add_argument('--group', type=str, default='Definite MG', help='Group name (default: Definite MG)')
    parser.add_argument('--visit', type=int, default=0, help='Visit index (default: 0)')
    args = parser.parse_args()

    ds = MGDataset('data/raw')
    print(f"Groups: {ds.groups}")
    print(f"Patients in '{args.group}': {len(ds.patients(args.group))}")

    patient = ds.patients(args.group)[args.patient]
    print(f"Patient: {patient}")

    visits = ds.visits(args.group, patient)
    print(f"Visits: {visits}")

    files = ds.get_files(args.group, args.patient, visits[args.visit])
    print(f"Files in visit: {len(files)}")
    
    for f in files:
        print(f"  {f['axis']} {f['frequency']}: {os.path.basename(f['path'])}")

    sample = ds.load_csv((args.group, args.patient, visits[args.visit], 0, 0))
    cycles = get_cycles(sample["Target"])

    # ---- test distintos kernels ----
    kernels = [3, 5, 7, 9, 11]
    s, e = cycles[0]
    t_seg = sample["time"][s:e]
    tgt_seg = sample["Target"][s:e]

    fig, axes = plt.subplots(len(kernels) + 1, 1, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"Denoise kernel comparison | Patient {args.patient} | Cycle 0")

    axes[0].plot(t_seg, sample["L"][s:e], label="L")
    axes[0].plot(t_seg, sample["R"][s:e], label="R")
    axes[0].plot(t_seg, tgt_seg, label="Target", color="black", linestyle=":", alpha=0.6)
    axes[0].set_title("Original")
    axes[0].set_ylim(-30, 30)
    axes[0].legend(ncols=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for idx, k in enumerate(kernels):
        L_f = denoise(sample["L"], kernel=k)
        R_f = denoise(sample["R"], kernel=k)
        axes[idx + 1].plot(t_seg, L_f[s:e], label="L")
        axes[idx + 1].plot(t_seg, R_f[s:e], label="R")
        axes[idx + 1].plot(t_seg, tgt_seg, label="Target", color="black", linestyle=":", alpha=0.6)
        axes[idx + 1].set_title(f"kernel={k}")
        axes[idx + 1].set_ylim(-30, 30)
        axes[idx + 1].legend(ncols=3, fontsize=8)
        axes[idx + 1].grid(True, alpha=0.3)

    os.makedirs('out/explore', exist_ok=True)
    figure_path = 'out/explore'

    plt.savefig(os.path.join(figure_path, f"kernel_comparison_{args.patient}.png"))
    plt.close(fig)

    # ---- guardar con kernel=7 (default) ----
    filtered = {
        "time": sample["time"],
        "L": denoise(sample["L"], kernel=7),
        "R": denoise(sample["R"], kernel=7),
        "Target": sample["Target"],
        "meta": sample["meta"],
    }

    figs = plot_cycles_comparison(
        sample, filtered, cycles, str(args.patient), figure_path
    )
    for fig in figs:
        plt.close(fig)

    print(f'Figures saved in {figure_path}/')
    