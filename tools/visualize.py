import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def visualize_raw(data: dict, title: str = None) -> plt.Figure:
    meta = data.get("meta", {})
    if title is None:
        title = f"{meta.get('date','?')} | {meta.get('patient','?')} | {meta.get('group','?')} | {meta.get('axis','?')} {meta.get('frequency','?')}"

    fig, ax = plt.subplots(1, 1, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(title)

    t = data["time"]
    ax.plot(t, data["L"], label="L", linewidth=0.6)
    ax.plot(t, data["R"], label="R", linewidth=0.6)

    ax2 = ax.twinx()
    ax2.plot(t, data["Target"], label="Target", linewidth=0.5, alpha=0.7, color="black", linestyle=":")

    ax.set_ylabel("Degrees")
    ax.set_xlabel("Time (sec)")
    ax.legend(ncols=4, fontsize=8, loc='lower left')
    ax2.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.AutoLocator())

    return fig


def visualize_cycles(t: list, target: list, cycles: list[tuple[int, int]], title: str = "") -> plt.Figure:

    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(title)

    ax.plot(t, target, color="gray", linewidth=0.5, alpha=0.4)

    for i, (s, e) in enumerate(cycles):
        c = colors[i % len(colors)]
        ax.plot(t[s:e], target[s:e], color=c, linewidth=1.5, label=f"Cycle {i+1}" if i < len(colors) else "")

    ax.set_ylabel("Degrees")
    ax.set_xlabel("Time (sec)")
    ax.legend(ncols=5, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.AutoLocator())

    return fig


def plot_cycles(data: dict, cycles: list[tuple[int, int]]) -> list[plt.Figure]:
    t = data["time"]
    meta = data.get("meta", {})
    base_title = f"{meta.get('date','?')} | {meta.get('patient','?')} | {meta.get('group','?')} | {meta.get('axis','?')} {meta.get('frequency','?')}"

    figs = []
    for i, (s, e) in enumerate(cycles):
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
        fig.suptitle(f"{base_title} | Cycle {i+1}")

        ax.axvline(x=t[s], color="green", linewidth=1, alpha=0.6, linestyle="--")
        ax.axvline(x=t[e-1], color="red", linewidth=1, alpha=0.6, linestyle="--")

        ax.plot(t[s:e], data["L"][s:e], label="L", linewidth=0.8)
        ax.plot(t[s:e], data["R"][s:e], label="R", linewidth=0.8)
        ax.plot(t[s:e], data["Target"][s:e], label="Target", linewidth=0.6, alpha=0.7, color="black", linestyle=":")

        ax.set_ylabel("Degrees")
        ax.set_xlabel("Time (sec)")
        ax.legend(ncols=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.AutoLocator())

        figs.append(fig)

    return figs


def plot_cycles_comparison(data: dict, filtered_data: dict, cycles: list[tuple[int, int]], patient_id: str, output_dir: str = "dataset") -> list[plt.Figure]:
    t = data["time"]
    meta = data.get("meta", {})
    base_title = f"{meta.get('date','?')} | {meta.get('patient','?')} | {meta.get('group','?')} | {meta.get('axis','?')} {meta.get('frequency','?')}"

    out_path = os.path.join(output_dir, patient_id)
    os.makedirs(out_path, exist_ok=True)

    figs = []
    for i, (s, e) in enumerate(cycles):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
        fig.suptitle(f"{base_title} | Cycle {i+1}")

        for ax, src, label in [(ax1, data, "Original"), (ax2, filtered_data, "Filtered")]:
            ax.plot(t[s:e], src["L"][s:e], label="L", linewidth=0.8)
            ax.plot(t[s:e], src["R"][s:e], label="R", linewidth=0.8)
            ax.plot(t[s:e], src["Target"][s:e], label="Target", linewidth=0.6, alpha=0.7, color="black", linestyle=":")
            ax.set_title(label)
            ax.set_ylabel("Degrees")
            ax.set_xlabel("Time (sec)")
            ax.set_ylim(-30, 30)
            ax.legend(ncols=3, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(ticker.AutoLocator())

        fig.savefig(os.path.join(out_path, f"cycle_{i}.png"))
        figs.append(fig)

    return figs
