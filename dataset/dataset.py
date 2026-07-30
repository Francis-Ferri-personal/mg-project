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
                    self._by_group.setdefault(group, {}).setdefault(patient_name, {}).setdefault(visit_date, []).append(file_info)

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

    def get_files(self, group: str, patient: str, date: str = None) -> list[dict]:
        patient_visits = self._by_group.get(group, {}).get(patient, {})
        if date:
            return patient_visits.get(date, [])
        return [f for files in patient_visits.values() for f in files]

    def load_csv(self, file_info: dict) -> dict:
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

    def __len__(self):
        return len(self._files)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.dirname(__file__)

    import matplotlib.pyplot as plt
    from tools.visualize import visualize_raw, visualize_cycles, plot_cycles, plot_cycles_comparison
    from preprocesing.filters import denoise
    from preprocesing.cycles import get_cycles

    id_num = 15

    ds = MGDataset()
    print(f"Groups: {ds.groups}")
    print(f"Patients in 'Definite MG': {len(ds.patients('Definite MG'))}")

    patient = ds.patients('Definite MG')[id_num]
    print(f"First patient: {patient}")

    visits = ds.visits('Definite MG', patient)
    print(f"Visits: {visits}")

    files = ds.get_files('Definite MG', patient, visits[0])
    print(f"Files in first visit: {len(files)}")
    
    for f in files:
        print(f"  {f['axis']} {f['frequency']}: {os.path.basename(f['path'])}")

    sample = ds.load_csv(files[0])
    cycles = get_cycles(sample["Target"])

    # ---- test distintos kernels ----
    kernels = [3, 5, 7, 9, 11]
    s, e = cycles[0]
    t_seg = sample["time"][s:e]
    tgt_seg = sample["Target"][s:e]

    fig, axes = plt.subplots(len(kernels) + 1, 1, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"Denoise kernel comparison | {sample['meta']['patient']} | Cycle 0")

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

    plt.savefig(os.path.join(dataset_dir, f"kernel_comparison_{id_num}.png"))

    # ---- guardar con kernel=7 (default) ----
    filtered = {
        "time": sample["time"],
        "L": denoise(sample["L"], kernel=7),
        "R": denoise(sample["R"], kernel=7),
        "Target": sample["Target"],
        "meta": sample["meta"],
    }

    figs = plot_cycles_comparison(sample, filtered, cycles, str(id_num))
    for fig in figs:
        plt.close(fig)
    
