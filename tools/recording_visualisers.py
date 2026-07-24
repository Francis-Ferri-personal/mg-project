import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tools.recording_analysis import RECORDING_ANALYSIS_SIDES

STATS_DISPLAY_CENTRALITIES = {
    'mean':{'marker':'X','color':(0.65,0.2,0.2),'label':'mean'},
    'median':{'marker':'*','color':(0.2,0.62,0.65),'label':'median'},
    'mode':{'marker':'H','color':(0.65, 0.67, 0.2), 'label':'mode'},
    'mid-extreme':{'marker':'o','color':(0.2,0.65,0.2),'label':'mid-extreme'},
}

STATS_DISPLAY_EXTREMES = {
    'min':{'marker':'D','color':(0.56,0.56,0.56),'label':'min'},
    'max':{'marker':'D','color':(0.56,0.56,0.26),'label':'max'},
    '95%':{'marker':'.','color':(0.3,0.73,0.54),'label':'95th percentile'},
    '5%':{'marker':'.','color':(0.68,0.33,0.72),'label':'5th percentile'}
}

STATS_DISPLAY_DEFAULT = {
    'mean':{'marker':'X','color':(0.65,0.2,0.2),'label':'mean'},
    'median':{'marker':'*','color':(0.2,0.62,0.65),'label':'median'},
    '95%':{'marker':'.','color':(0.3,0.73,0.54),'label':'95th percentile'},
    '5%':{'marker':'.','color':(0.68,0.33,0.72),'label':'5th percentile'}
}

class RecordingVisualiser:
    accepted_kwargs = (
        'representative_cycle',
        'list_of_cycles_stats'
    )

    def __init__(self, 
                 list_of_cycles:list[dict[str,list[float]]],
                 accession_str:str,
                 accession_data:dict[str,],
                 **kwargs):
        
        self.list_of_cycles = list_of_cycles
        self.accession_str = accession_str
        self.accession_data = accession_data

        self.canvas_title = f'{' | '.join(f'{k}: {v}' for k,v in accession_data.items())}'

        for k, v in kwargs.items():
            if k not in RecordingVisualiser.accepted_kwargs:
                continue
            setattr(self, k, v)

    @staticmethod
    def draw_whole_recording_of(
            list_of_cycles:list[dict[str,list[float]]],
            title : str = 'Whole Recording',
            canvas: tuple[plt.Figure, plt.Axes] = None,
            **canvas_kwargs):
        """Draws an entire recording's cycles. Pass in the recordings cycle as a list

        Expected list_of_cycles structure:

        cycles:
            - cycle:
                - {L:[series], 
                - R:[series], 
                - AVG:[series], 
                - time_list:[series], 
                - target_list:[series]}

        Args:
            list_of_cycles (list[dict[str,list[float]]]): A list of cycles with time_list and target_list
        """
        default_fig_args = {
            'figsize' : (24,14),
            'dpi' : 100,
        }
        canvas_kwargs = {} if canvas_kwargs is None else canvas_kwargs
        fig_args = default_fig_args | canvas_kwargs

        if canvas is None:
            fig, axs = plt.subplots(
                nrows=3, constrained_layout = True, **fig_args
            )
        else:
            fig, axs = canvas
            if axs.get_gridspec().nrows != 3:
                raise IndexError("canvas must have an axes with 3 rows")

        fig.suptitle(title)
        
        for side_idx, side in enumerate(RECORDING_ANALYSIS_SIDES):
            ax: plt.Axes = axs[side_idx]
            ax2 = ax.twinx()

            for cycle_idx, cycle in enumerate(list_of_cycles):
                ax.plot(cycle['time_list'], cycle[side])
                ax2.plot(cycle['time_list'],cycle['target_list'], 
                            linewidth=1, alpha=0.6, linestyle=':', color=(0.4,0.04,0.54))

            ax.set_xbound(list_of_cycles[0]['time_list'][0],
                            list_of_cycles[-1]['time_list'][-1])
            ax.grid(True)
            ax.set_xlabel('time (s)')
            ax.set_ylabel('object')
            ax2.set_ylabel('target')

            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

            ax.set_title(side)

        plt.show()

    def draw_whole_recording(self, title:str=None, **canvas_kwargs):
        if title is None:
            title = self.canvas_title

        self.draw_whole_recording_of(
            self.list_of_cycles, title=title, canvas=None, **canvas_kwargs
        )

    @staticmethod
    def draw_statistics_over_cycles_of(
        list_of_cycles_stats:list[dict[str,]],
        stats_display_options:dict[str,dict[str,]] = STATS_DISPLAY_DEFAULT,
        title : str = 'Stats Over Cycles',
        canvas : tuple[plt.Figure, plt.Axes] = None,
        **canvas_kwargs):
        """Draws the statistics of each cycle, pass in the list of cycle statistics and the options to display
        by default it draws the 5th and 95th percentile, mean, and median.

        Expected list_of_cycles_stats:
        cycles:
            - cycle:
                - L:
                    - percentiles
                        - 0%, 1%, ..., 100%
                    - statistics
                        - mean, median, mode, etc...
                - R
                - AVG

        Args:
            list_of_cycles_stats (list[dict[str,]]): See above
            stats_display_options (dict[str,dict[str,]], optional): Defaults to STATS_DISPLAY_DEFAULT. Also accepts:
                - STATS_DISPLAY_CENTRALITIES
                - STATS_DISPLAY_EXTREMES
                - custom stats display dicts
        """
        default_fig_args = {
            'figsize' : (24,14),
            'dpi' : 100,
        }
        canvas_kwargs = {} if canvas_kwargs is None else canvas_kwargs
        fig_args = default_fig_args | canvas_kwargs

        if canvas is None:
            fig, axs = plt.subplots(
                nrows=3, constrained_layout = True, **fig_args
            )
        else:
            fig, axs = canvas
            nrows = axs.get_gridspec().nrows
            if not nrows == 3:
                raise IndexError("canvas must have an axes with 3 rows")

        fig.suptitle(title)
        
        stats_display_dict = {}

        for cycle in list_of_cycles_stats:
            for side in RECORDING_ANALYSIS_SIDES:
                cycle_side : dict[str,dict] = cycle[side]

                for key in stats_display_options:
                    data = cycle_side['statistics']\
                            .get(key, cycle_side['percentiles'].get(key)
                    )
                    
                    statistic_dict = stats_display_dict[key] = stats_display_dict.setdefault(key, {})
                    statistic_dict[side] = statistic_dict.setdefault(side,[]) + [data]

        for stat_attr_k, stat_attr in stats_display_dict.items():
            for side_idx, (side_name, side) in enumerate(stat_attr.items()):
                ax : plt.Axes = axs[side_idx]
                display_options = stats_display_options[stat_attr_k]
                marker, color = display_options['marker'], display_options['color']

                ax.plot(side, marker=marker, color=color, linewidth=0, label=stat_attr_k)
                ax.plot(side, color=color, linewidth=1, alpha=0.5)

                ax.grid(True)
                ax.grid(True, 'minor', linestyle='-', alpha=0.2)

                ax.set_title(f'{side_name} stats')

                ax.set_xbound(-1, len(side))
                ax.set_ybound(-28.2,28.2)

                ax.set_ylabel('Degrees')
                ax.legend(ncols=2)

                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

                ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))

                if side_idx == 2:
                    ax.set_xlabel('Cycle Number')
            
        plt.show()

    def draw_statistics_over_cycles(
            self, 
            stats_display_options: dict[str,dict[str,]] = STATS_DISPLAY_DEFAULT,
            list_of_cycles_stats : list[dict[str,]] = None, 
            title:str = None, **canvas_kwargs):
        
        if title is None:
            title = self.canvas_title

        list_of_cycles_stats = getattr(self, 'list_of_cycles_stats', list_of_cycles_stats)

        if list_of_cycles_stats is None:
            raise KeyError("draw_statistics_over_cycles needs instance to have list_of_cycles_stats in instance or as an argument")

        self.draw_statistics_over_cycles_of(
            list_of_cycles_stats, stats_display_options, title=title, canvas=None, **canvas_kwargs
        )

    @staticmethod
    def draw_single_cycle_of(
            cycle_dict: dict[str,list[float]],
            representative_cycle: dict[str,list[float]]=None,
            title : str = 'Stats Over Cycles',
            canvas : tuple[plt.Figure, plt.Axes] = None,
            **canvas_kwargs):
        """Draws a single cycle's signal, if given a representative cycle, will draw it in twinx

        Args:
            cycle_dict (dict[str,list[float]]): A signal with three channels: L, R, AVG
            representative_cycle (dict[str,list[float]], optional): Like cycle_dict. Defaults to None.
        """
        default_fig_args = {
            'figsize' : (24,14),
            'dpi' : 100,
        }
        canvas_kwargs = {} if canvas_kwargs is None else canvas_kwargs
        fig_args = default_fig_args | canvas_kwargs

        if canvas is None:
            fig, ax = plt.subplots(
                nrows=1, constrained_layout = True, **fig_args
            )

        else:
            fig, ax = canvas
            nrows = ax.get_gridspec().nrows
            if not nrows == 1:
                raise IndexError("canvas must have an axes with 1 rows")

        fig.suptitle(title)

        has_representative = representative_cycle is not None
        has_target = 'target_list' in cycle_dict

        if has_representative or has_target:
            ax2 = ax.twinx()
            ax2.set_ybound(-24.2,24.2)
            ax2.set_ylabel('Degrees (Ideal)')

        for side in RECORDING_ANALYSIS_SIDES:
            ax.plot(cycle_dict[side], label=side)

            if representative_cycle is not None:
                ax2.plot(
                    representative_cycle[side], 
                    linestyle=":",alpha=0.65, label=f'averaged_{side}'
                )

        if 'target_list' in cycle_dict:
            ax2.plot(
                cycle_dict['target_list'], 
                linestyle=":", linewidth = 2, alpha = 0.65, label='target'
            )
            ax2.set_ybound(-24.2,24.2)
            ax2.legend(loc="upper right")
            

        ax.grid(True)
        ax.grid(True, 'minor', alpha=0.3)

        ax.set_ybound(-24.2,24.2)
        ax.set_xbound(-1,len(cycle_dict['AVG']))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.set_xlabel('Sample Count')
        ax.set_ylabel('Degrees (Cycle)')

        ax.legend(loc="upper left")

        plt.show()

    def draw_single_cycle(
            self, cycle_idx:int, 
            representative_cycle:dict[str,list[float]] = None, 
            title:str=None, **canvas_kwargs):
        
        if title is None:
            title = self.canvas_title

        representative_cycle = getattr(self, 'representative_cycle', representative_cycle)

        cycle = self.list_of_cycles[cycle_idx]

        self.draw_single_cycle_of(
            cycle, representative_cycle, title=title, canvas=None, **canvas_kwargs
        )