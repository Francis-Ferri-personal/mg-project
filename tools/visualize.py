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
