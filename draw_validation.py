import json
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from mg_dataset.dataset import MGDataset
    from mg_dataset.preprocesing import filters
    from mg_dataset.preprocesing import cycles

except ModuleNotFoundError: # probably fixes imports if venv is set up incorrectly
    import sys
    sys.path.append(os.path.join('mg_scripts','mg_dataset'))

import mg_tools.stats as stats

from tqdm import tqdm

DEFAULT_DATABASE = 'Database/mg-data'
VALIDATION_PATH = 'exports/cv_config.json'

def draw_separate_channels(
    cycle_pos:dict, 
    cycle_vel:dict, 
    channel_names = ('L','R'), 
    highlights:dict = None,
    title = 'separated channels',
    pos_y_boundaries = (-25, 25),
    pos_y_steps = 5,
    vel_y_boundaries = (-500, 500),
    vel_y_steps = 50,
    **canvas_kwargs
) -> tuple[plt.Figure, plt.Axes]:

    default_fig_args = {
        'nrows':2,
        'ncols':2,
        'figsize':(24,14),
        'dpi' : 72,
    }
    
    canvas_kwargs = {} if canvas_kwargs is None else canvas_kwargs
    fig_args = default_fig_args | canvas_kwargs

    fig, axs = plt.subplots(**fig_args, constrained_layout=True)

    fig.suptitle(title)

    for attr_idx, attr in enumerate((cycle_pos, cycle_vel)):
        ax_row = axs[attr_idx]

        attr_name = ['pos','vel'][attr_idx]
        y_bounds = (pos_y_boundaries, vel_y_boundaries)[attr_idx]
        y_steps = (pos_y_steps, vel_y_steps)[attr_idx]

        for channel_idx, channel_name in enumerate(channel_names):
            ax : plt.Axes = ax_row[channel_idx]

            ax.set_ybound(*y_bounds)

            channel_data = attr[channel_name]
            target_list = cycle_pos['target_list']
            ax2 = ax.twinx()

            ax.plot(channel_data, label = f'{attr_name} {channel_name}')
            ax2.plot(target_list, ':', color='red', label = 'target')

            if highlights is not None:
               try:
                for highlight_idx, (state_start, state_end) in enumerate(highlights[channel_name]['states']):
                    ax.fill_betweenx(
                        list(range(*map(int, ax.get_ybound()))), 
                        state_start, state_end,
                        alpha=0.4,
                        color=(0.85,0.33,0.33)
                    )
               except ValueError:
                    print(
                        'Unpacking error', 
                        f' channel: {channel_name} | idx: {highlight_idx}', 
                        f' {highlights[channel_name]['states'][highlight_idx]}',
                        sep = '\n'
                    )
                    pass

            ax.legend(loc='lower left')
            ax2.legend(loc='upper right')
            
            ax.set_title(f'{attr_name} {channel_name}')

            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

            y_bounds = ax.get_ybound()
            while (y_bounds[1]-y_bounds[0])/y_steps > 30:
                y_steps *= 2

            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_steps))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_steps/5))

            ax.tick_params(axis='x', labelsize=8, rotation=45)

            ax.grid(True)
            ax.grid(True, 'minor', alpha=0.3)

    return fig,axs

def process_sample(sample_info: str, db_path: str) -> bool:
    """
    Load one sample, process all its cycles, and save raw/filtered figures.
    Returns True if all cycles were processed (or skipped), False on fatal error.
    """
    # Each worker sets its own matplotlib backend
    import matplotlib
    matplotlib.use('agg')

    import matplotlib.pyplot as plt

    import sys
    sys.stdout.reconfigure(line_buffering=True)

    dataset = MGDataset(db_path)
    get_vel = lambda pos_dict, filter_size: {
        k: stats.median_rolling_average(stats.rate_of_change(pos_dict[k]), filter_size)
        for k in ('L', 'R')
    }

    try:
        sample = dataset.load_csv_of(sample_info)
    except Exception as e:
        print(f"Error loading {sample_info}: {e}")
        return False

    sample_cycles = cycles.get_cycles(sample['Target'])
    if not sample_cycles:
        return True   # nothing to do

    sample_meta = sample['meta']
    meta_order = [sample_meta[k] for k in ('group', 'date', 'frequency', 'axis')]
    sample_path = os.path.join('exports', 'figures', 'validation', *meta_order)
    os.makedirs(sample_path, exist_ok=True)

    worker_id = multiprocessing.current_process().ident

    # can't get this to work
    cycle_iter = tqdm(
        enumerate(sample_cycles),
        total=len(sample_cycles),
        desc=f"Sample {sample_info}",
        position=worker_id,          # Unique line per worker
        leave=False                  # Remove bar when the sample finishes
    )

    for cycle_idx, (start, end) in cycle_iter:
        # Build paths and check existing files
        raw_path = os.path.join(sample_path, 'raw')
        filtered_path = os.path.join(sample_path, 'filtered')
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(filtered_path, exist_ok=True)

        file_name_template = f'cycle_{cycle_idx}'
        raw_name = file_name_template + '-raw.jpg'
        filtered_name = file_name_template + '-filtered.jpg'

        raw_full = os.path.join(raw_path, raw_name)
        filtered_full = os.path.join(filtered_path, filtered_name)

        has_raw = os.path.exists(raw_full)
        has_filtered = os.path.exists(filtered_full)
        if has_raw and has_filtered:
            continue   # already done

        # Extract cycle data
        raw_cycle = {
            'L': sample['L'][start:end],
            'R': sample['R'][start:end],
            'time_list': sample['time'][start:end],
            'target_list': sample['Target'][start:end]
        }
        vel_raw_cycle = raw_cycle | get_vel(raw_cycle, 0)

        median_filtered_cycle = raw_cycle | {
            'L': filters.denoise(sample['L'][start:end]),
            'R': filters.denoise(sample['R'][start:end])
        }
        vel_median_filtered_cycle = raw_cycle | get_vel(median_filtered_cycle, 0)

        sample_name = ' | '.join([f'cycle_idx #{cycle_idx}'] + meta_order)

        # Render raw
        if not has_raw:
            try:
                fig, _ = draw_separate_channels(
                    cycle_pos=raw_cycle,
                    cycle_vel=vel_raw_cycle,
                    highlights=None,
                    vel_y_boundaries=(None, None),
                    title=sample_name + ' | raw'
                )
                fig.savefig(raw_full, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Raw render error for {sample_info}, cycle {cycle_idx}: {e}")

        # Render filtered
        if not has_filtered:
            try:
                fig, _ = draw_separate_channels(
                    cycle_pos=median_filtered_cycle,
                    cycle_vel=vel_median_filtered_cycle,
                    highlights=None,
                    vel_y_boundaries=(None, None),
                    title=sample_name + ' | filtered'
                )
                fig.savefig(filtered_full, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Filtered render error for {sample_info}, cycle {cycle_idx}: {e}")

        cycle_iter.set_postfix(status="rendering")

    return True

if __name__ == '__main__':
    DATASET = MGDataset(DEFAULT_DATABASE)
    kernel_size = 7

    get_vel = lambda pos_dict, filter_size : {
        k:stats.median_rolling_average(stats.rate_of_change(pos_dict[k]), filter_size) for k in ('L','R')
    }

    # read validation json
    with open(VALIDATION_PATH) as validation_json:
        validations = json.load(validation_json)
    
    val_samples = []

    for vold in validations['folds']:
        for file in vold['train']['healthy'] + vold['train']['mg']:
            sample_str = file[:-5]

            if sample_str in val_samples:
                continue

            val_samples += [file[:-5]]
        break # otherwise we'll just draw the entire thing lol

    val_samples_2 = []

    for sample_idx, (patient_str) in enumerate(val_samples):
        group, patient_idx = list(map(int, patient_str.split('-')))
        visit_idx = 0

        while True:
            try:
                visit = DATASET.get_files(group, patient_idx, visit_idx)
            except IndexError:
                break

            for file in visit:
                val_samples_2.append(file)

            visit_idx += 1

    val_samples = val_samples_2

    # parallelised thx to deepseek
    with ProcessPoolExecutor(max_workers=None) as executor:
        # Submit all tasks
        futures = {executor.submit(process_sample, sample, DEFAULT_DATABASE): sample
                    for sample in val_samples}

        # Use tqdm to show overall progress
        with tqdm(total=len(futures), desc="Samples") as pbar:
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    success = future.result()
                except Exception as e:
                    print(f"Sample {sample} raised exception: {e}")
                pbar.update(1)

    print("All samples processed.")