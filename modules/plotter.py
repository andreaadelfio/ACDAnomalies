'''
Plotter module for plotting data points and curves.
'''
import os
import operator
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
try:
    from modules.utils import Logger, logger_decorator, Time
    from modules.config import BACKGROUND_PREDICTION_FOLDER_NAME, PLOT_TRIGGER_FOLDER_NAME
except:
    from utils import Logger, logger_decorator, Time
    from config import BACKGROUND_PREDICTION_FOLDER_NAME, PLOT_TRIGGER_FOLDER_NAME


class Plotter:
    '''
    This class provides methods for plotting data points and curves.
    '''
    logger = Logger('Plotter').get_logger()

    @logger_decorator(logger)
    def __init__(self, x = None, y = None, df: pd.DataFrame = None, xy: dict = None, label = ''):
        '''
        Initialize the Plotter object.

        Parameters:
        ----------
            x (list): The x-coordinates of the data points (default: None).
            y (list): The y-coordinates of the data points (default: None).
            df (pd.DataFrame): The y-coordinates of the data points (default: None).
            xy (dict): A dictionary of x, y, and smooth y values for multiple curves (default: None).
            label (str): The label for the plot (default: '').
        '''
        self.x = x
        self.y = y
        self.df = df
        self.xy = xy
        self.label = label

    @logger_decorator(logger)
    def plot(self, marker = '-', lw = 0, show = True):
        '''
        Plot a single curve.

        Parameters:
        ----------
            marker (str): The marker style for the plot (default: '-').
        '''
        plt.figure()
        plt.tight_layout(pad = 0.4)
        plt.plot(self.x, self.y, marker = marker, lw = lw, label = self.label)
        plt.legend()
        plt.title(self.label)
        plt.xlim(self.x[0], self.x[len(self.x) - 1])
        if show:
            plt.show()

    @logger_decorator(logger)
    def plot_tiles(self, marker = '-', lw = 0.2, with_smooth = False, show = True):
        '''
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
        i = 0
        _, axs = plt.subplots(len(self.xy), 1, sharex=True)
        plt.tight_layout(pad = 0.4)
        axs[0].set_title(self.label)
        for label, xy in self.xy.items():
            axs[i].plot(xy[0], xy[1], marker = marker, lw = lw, label=label)
            if with_smooth:
                axs[i].plot(xy[0], xy[2], marker = marker, label=f'{label} smooth')
            axs[i].legend()
            axs[i].grid()
            axs[i].set_xlim(xy[0][0], xy[0][-1])
            i += 1
        if show:
            plt.show()

    @logger_decorator(logger)
    def multiplot(self, marker = '-', lw = 0.1, with_smooth = False, show = True):
        '''
        Plots multiple curves on the same figure. Uses the `xy` variable passed to the Plotter.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
        plt.figure()
        plt.tight_layout(pad = 0.4)
        plt.title(self.label)
        for label, xy in self.xy.items():
            plt.plot(xy[0], xy[1], marker = marker, lw = lw, label = label)
            if with_smooth:
                plt.plot(xy[0], xy[2], marker = marker, label = f'{label} smooth')
            plt.legend()
        plt.xlim(xy[0][0], xy[0][-1])
        if show:
            plt.show()

    @logger_decorator(logger)
    def df_plot_corr_tiles(self, x_col, excluded_cols = None, marker = '-', ms = 1, lw = 0.1, smoothing_key = 'smooth', show = True):
        '''
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and column not in excluded_cols and column != 'datetime']
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=True, figsize=(17, 10), num=self.label)
        plt.tight_layout(pad = 0.4)
        fig.subplots_adjust(bottom = 0.08, hspace = 0.13, wspace = 0.08, left = 0.04)
        if n_plots > 1:
            axs = axs.flatten()
            for i, column in enumerate(df_columns):
                if column == x_col:
                    x = self.df['datetime']
                    axs[i].plot(x, self.df[column], marker = marker, ms = ms, lw = lw, label=column)
                else:
                    x = self.df[x_col]
                    axs[i].plot(self.df[column], x, marker = marker, ms = ms, lw = lw, label=column)
                if smoothing_key != '' and f'{column}_{smoothing_key}' in self.df.columns:
                    axs[i].plot(x, self.df[f'{column}_{smoothing_key}'], marker = '.', ms = 0.2, lw = '0.1', label=f'{column} {smoothing_key}')
                axs[i].legend(loc='upper right')
                axs[i].set_xlabel(column)
                axs[i].set_ylabel(x_col)
                axs[i].grid()
                axs[i].tick_params(axis="x", labelrotation=30)
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
        else:
            column = df_columns[0]
            axs.plot(self.df[column], x, marker = marker, lw = lw, label=column)
            axs.set_xlabel(column)
            axs.set_ylabel(x_col)
            axs.legend(loc='upper right')
            axs.grid()
            axs.tick_params(axis="x", labelrotation=45)
        if show:
            plt.show()

    @logger_decorator(logger)
    def df_plot_tiles(self, x_col, top_x_col = None, excluded_cols = None, marker = '-', lw = 0.1, smoothing_key = 'smooth', show = True):
        '''
        Plot multiple curves as tiles.

        Parameters:
        ----------
            lw (float): Line width of the curves (default: 0.1).
            with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''

        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and column not in excluded_cols and column != x_col]
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig = plt.figure(figsize=(16, 10), num=self.label)
        fig.subplots_adjust(bottom=0.076, hspace=0, left=0.052, right=0.98, top=0.9, wspace=0.076)
        x = self.df[x_col]
        axs = []

        if n_plots > 1:
            for i in range(n_plots):
                if i == 0:
                    ax = fig.add_subplot(n_rows, n_cols, i + 1)
                else:
                    ax = fig.add_subplot(n_rows, n_cols, i + 1, sharex=axs[0])
                axs.append(ax)

            for i, column in enumerate(df_columns):
                axs[i].plot(x.values, self.df[column], marker=marker, lw=lw, label=column)
                if smoothing_key and f'{column}_{smoothing_key}' in self.df.columns:
                    axs[i].plot(x.values, self.df[f'{column}_{smoothing_key}'], marker='.', ms=0.2, lw=0.1, label=f'{column} {smoothing_key}')
                
                if i >= n_cols * (n_rows - 1) - (n_cols * n_rows - n_plots):
                    axs[i].tick_params(axis="x", labelrotation=30)
                    plt.setp(axs[i].get_xticklabels(), visible=True)
                    offset_text = axs[i].get_xaxis().get_offset_text().get_text()
                    xlabel = f'{x_col} ({offset_text})' if offset_text else f'{x_col} ({x.iloc[0]})'
                    axs[i].set_xlabel(xlabel)
                    axs[i].get_xaxis().get_offset_text().set_visible(False)
                else:
                    plt.setp(axs[i].get_xticklabels(), visible=False)
                
                axs[i].legend(loc='upper right')
                axs[i].grid()
                axs[i].set_xlim(x.iloc[0], x.iloc[-1])
                axs[i].tick_params(axis="y", labelrotation=30)
                
                if top_x_col and top_x_col in self.df.columns and i < n_cols:
                    secax = axs[i].secondary_xaxis('top', functions=(Time.from_met_to_datetime, lambda x: x))
                    secax.set_xlabel(f'{top_x_col} ({self.df[top_x_col].iloc[0]})')
                    secax.tick_params(axis="x", labelrotation=30)
                # if i % n_cols == 0:
                #     axs[i].set_ylabel('Count Rate')

            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
        else:
            column = df_columns[0]
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x, self.df[column], marker = marker, lw = lw, label=column)
            if smoothing_key != '' and f'{column}_{smoothing_key}' in self.df.columns:
                ax.plot(x, self.df[f'{column}_{smoothing_key}'], marker = '.', ms = 0.2, lw = '0.1', label=f'{column} {smoothing_key}')
            ax.legend(loc='upper right')
            ax.grid()
            ax.set_xlim(x.iloc[0], x.iloc[-1])
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_xlabel(f'{x_col} {x.iloc[0]}')

        if show:
            plt.tight_layout(pad=0.4)
            plt.show()

    @logger_decorator(logger)
    def df_plot_tiles_for_pres(self, x_col, excluded_cols = None, marker = '-', lw = 0.1, smoothing_key = 'smooth', show = True):
        '''
        Plot multiple curves as tiles.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
        if not excluded_cols:
            excluded_cols = []
        df_columns = [column for column in self.df.columns if f'_{smoothing_key}' not in column and column not in excluded_cols and column != x_col]
        n_plots = len(df_columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, squeeze=True, figsize=(17, 10), num=self.label)
        plt.tight_layout(pad = 0.4)
        fig.subplots_adjust(bottom = 0.06, hspace = 0)
        x = self.df[x_col]
        if n_plots > 1:
            axs = axs.flatten()
            for i, column in enumerate(df_columns):
                axs[i].plot(x, self.df[column], marker = marker, lw = lw, label=column)
                if smoothing_key != '' and f'{column}_{smoothing_key}' in self.df.columns:
                    axs[i].plot(x, self.df[f'{column}_{smoothing_key}'], marker = '.', ms = 0.2, lw = '0.1', label=f'{column} {smoothing_key}')
                axs[i].legend(loc='upper right')
                axs[i].grid()
                # axs[i].set_xlim(x[0], x[len(x) - 1])
                axs[i].tick_params(axis="x", labelrotation=30)
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
        else:
            column = df_columns[0]
            axs.plot(x, self.df[column], marker = marker, lw = lw, label=column)
            axs.legend(loc='upper right')
            axs.grid()
            axs.set_xlim(x[0], x[len(x) - 1])
            axs.tick_params(axis="x", labelrotation=45)
        if show:
            plt.show()

    @logger_decorator(logger)
    def df_multiplot(self, x_col, marker = '-', lw = 0.1, with_smooth = False, show = True):
        '''
        Plots multiple curves on the same figure.

        Parameters:
        - lw (float): Line width of the curves (default: 0.1).
        - with_smooth (bool): Whether to plot smoothed curves as well (default: False).
        '''
        plt.figure(self.label)
        plt.tight_layout(pad = 0.4)
        plt.title(self.label)
        x = self.df[x_col]
        for column in self.df.columns:
            if column != x_col:
                plt.plot(x, self.df[column], marker = marker, lw = lw, label = column)
                if with_smooth:
                    plt.plot(x, self.df[column + '_smooth'], marker = marker, label = f'{column} smooth')
                plt.legend()
        plt.xlim(x[0], x[len(x) - 1])
        if show:
            plt.show()


    @logger_decorator(logger)
    def plot_tile(self, tiles_df, face='top', col='datetime', smoothing_key = 'smooth'):
        with sns.plotting_context("talk"):
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 12), num=face, tight_layout=True)
            fig.subplots_adjust(hspace=0)

            axs[0].plot(pd.to_datetime(tiles_df[col]), tiles_df[face], 'k-.')
            axs[0].plot(pd.to_datetime(tiles_df[col]), tiles_df[f'{face}_{smoothing_key}'], 'r-')
            axs[0].set_ylabel(f'{face} count rate')

            axs[1].plot(pd.to_datetime(tiles_df[col]), tiles_df[face] - tiles_df[f'{face}_{smoothing_key}'], 'k-.')
            axs[1].axhline(0, color='red')
            # axs[1].plot(pd.to_datetime(tiles_df[col]).ffill(), [0 for _ in tiles_df[col].ffill()], 'k-')
            axs[1].set_xlabel(f"{col} ({tiles_df[col].iloc[0]})")
            axs[1].set_xlim(tiles_df[col].iloc[0], tiles_df[col].iloc[-1])
            plt.xticks(rotation=0)
            axs[1].set_ylabel('Residuals')

    @logger_decorator(logger)
    def plot_tile_knn(self, inputs_outputs_df, y_pred, face='top'):
        '''Function to plot the tile for the KNN model.'''
        with sns.plotting_context("talk"):
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 12), num=face, tight_layout=True)
            fig.subplots_adjust(hspace=0)

            axs[0].plot(pd.to_datetime(inputs_outputs_df['datetime']), inputs_outputs_df[face], 'k-.')
            axs[0].plot(pd.to_datetime(inputs_outputs_df['datetime']), y_pred[face], 'r-')
            axs[0].set_title('foreground and background')
            axs[0].set_ylabel('Count Rate')

            axs[1].plot(pd.to_datetime(inputs_outputs_df['datetime']), inputs_outputs_df[face] - y_pred[face], 'k-.')
            axs[1].plot(pd.to_datetime(inputs_outputs_df['datetime']).ffill(), [0 for _ in inputs_outputs_df['datetime'].ffill()], 'k-')
            axs[1].set_xlabel('time (YYYY-MM-DD hh:mm:ss)')
            plt.xticks(rotation=0)
            axs[1].set_ylabel('Residuals')

    @logger_decorator(logger)
    def plot_pred_true(self, tiles_df, col_pred=['top_pred', 'Xpos_pred', 'Xneg_pred', 'Ypos_pred', 'Yneg_pred'], y_cols_raw=['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']):
        with sns.plotting_context("talk"):
            fig = plt.figure("pred_vs_true", layout="tight")
            fig.set_size_inches(24, 12)
            plt.axis('equal')
            plt.plot(tiles_df[y_cols_raw], tiles_df[col_pred], '.', alpha=0.2)
            min_y, max_y = min(tiles_df[col_pred].min()), max(tiles_df[col_pred].max())
            plt.plot([min_y, max_y], [min_y, max_y], '-')
            plt.xlabel('True signal')
            plt.ylabel('Predicted signal')
        plt.legend(y_cols_raw)

    @logger_decorator(logger)
    def plot_history(self, history, feature):
        plt.figure(f"history_{feature}", layout="tight")
        plt.plot(history.history[feature][4:])
        plt.plot(history.history[f'val_{feature}'][4:])
        plt.ylabel(feature)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

    @logger_decorator(logger)
    def show_models_params(self, folder_name, title, features_dict = {'loss': {'top': 0.001, 'Xpos': 0.001, 'Xneg': 0.001, 'Ypos': 0.001, 'Yneg': 0.001}}, show=True):
        df_ori = pd.read_csv(f'{folder_name}/models_params.csv', sep="\t").reset_index()
        df_ori = df_ori.assign(**{df_ori.columns.tolist()[i+1]: df_ori[df_ori.columns.tolist()[i]] for i in range(len(df_ori.columns.tolist()) - 1)}).drop(columns = 'index')
        cols = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
        n_plots = len(cols)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        _, axs = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(20, 12), num=title)
        axs = axs.flatten()
        for i, col in enumerate(cols):
            data = []
            labels = []
            for label, features in features_dict.items():
                df = df_ori
                for feature, conditions in features.items():
                    df = df[conditions["op"](df[feature], conditions["value"])] # pylint: disable=E1136
                print(df)
                data.append(df[col])
                labels.append(label)
            axs[i].hist(data, bins=int(len(df_ori)/4), alpha=0.5, label=labels, stacked=False, density=True)
            axs[i].set_xlabel('loss')
            axs[i].set_ylabel('count')
            axs[i].set_title(col)
            axs[i].legend()
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')
        if show:
            plt.show()

    @logger_decorator(logger)
    def plot_correlation_matrix(self, inputs_outputs_df: pd.DataFrame, show = True, save = False):
        '''Function to plot the correlation matrix.'''
        correlations = inputs_outputs_df.corr()
        plt.figure(figsize=(18, 18), num='correlations_matrix')
        sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME)

    @logger_decorator(logger)
    def plot_confusion_matrix(self, y_true, y_pred, show = True):
        '''Function to plot the confusion matrix.'''
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8), num='confusion_matrix')
        sns.heatmap(cm, annot=True, cmap='coolwarm', fmt=".2f")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if show:
            plt.show()

    @logger_decorator(logger)
    def plot_anomalies(self, support_vars, tiles_df, y_cols, y_pred_cols, save = True, show = False):
        '''Plots the anomalies passed as `df` in Plotter.'''
        if not os.path.exists(PLOT_TRIGGER_FOLDER_NAME):
            os.makedirs(PLOT_TRIGGER_FOLDER_NAME)

        for anomaly_start, anomalies in tqdm(self.df.items(), desc='Plotting anomalies'):
            faces = list(anomalies.keys())
            figs, axs = plt.subplots(6 + len(support_vars), 1, sharex=True, figsize=(13, 12), num=f"burst_{'_'.join(faces)}")

            anomaly_end = -1
            for anomaly in anomalies.values():
                if anomaly['stopping_time'] > anomaly_end:
                    anomaly_end = anomaly['stopping_time']
            anomaly_delta = (anomaly_end - int(anomaly_start))//5
            start = int(anomaly_start) - anomaly_delta
            end = anomaly_end + anomaly_delta

            for i, (face, face_pred) in enumerate(zip(y_cols, y_pred_cols)):
                face_color = "black"
                if face in faces:
                    face_color = None
                    changepoint = anomalies[face]['changepoint']
                    stopping_time = anomalies[face]['stopping_time']
                    max_significance = anomalies[face]['max_significance']
                    max_point = anomalies[face]['max_point']
                    sigma_val = anomalies[face]['sigma_val']
                    axs[i].axvline(tiles_df['datetime'][changepoint], color='red', lw=0.8)
                    axs[i].axvline(tiles_df['datetime'][stopping_time], color='red', lw=0.8)
                    # axs[i].text(tiles_df['datetime'][max_point], 1.01 * tiles_df[face_pred][max_point], f'{r"$s_{MAX}="}{round(max_significance//sigma_val)}\sigma$', color='black', fontsize=11)
                    # axs[i].fill((tiles_df['datetime'][start], tiles_df['datetime'][end], tiles_df['datetime'][end], tiles_df['datetime'][start]), (-5, -5, 15, 15), color="yellow", alpha=0.1)
                axs[i].plot(tiles_df[start:end]['datetime'], tiles_df[start:end][face], label=face, color=face_color)
                axs[i].plot(tiles_df[start:end]['datetime'], tiles_df[start:end][face_pred], label='background', color="red")
                axs[i].legend(loc="upper left")
                axs[i].set_ylim(min(tiles_df[start:end][face]), 1.01 * max(tiles_df[start:end][face]))

            max_val = -1
            min_val = 100
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            count = 0
            axs[1+i].axhline(0, color='grey', lw=0.8)
            for face in faces:
                significance = anomalies[face]['significance']
                axs[1+i].axhline(significance, color=colors[count], label=f"{face} $\\sigma$")
                diff = tiles_df[start:end][face] - tiles_df[start:end][f'{face}_pred']
                axs[1+i].plot(tiles_df[start:end]['datetime'], diff, color=colors[count], label=f"{face} residual", lw=0.7)
                min_val = min(min(diff), min_val)
                max_val = max(max(max(diff), significance), max_val)
                count+=1
            axs[1+i].fill((tiles_df['datetime'][int(anomaly_start)], tiles_df['datetime'][anomaly_end], tiles_df['datetime'][anomaly_end], tiles_df['datetime'][int(anomaly_start)]), (-5, -5, 15, 15), color="red", alpha=0.1)
            axs[1+i].axvline(tiles_df['datetime'][anomaly_end], color='red', lw=0.7)
            axs[1+i].axvline(tiles_df['datetime'][int(anomaly_start)], color='red', lw=0.7)
            axs[1+i].set_ylim(min_val, 1.01 * max_val)
            axs[1+i].legend(loc="upper left")

            for j, var in enumerate(support_vars):
                axs[2+i+j].plot(tiles_df[start:end]['datetime'], tiles_df[start:end][var], color="green", label=var)
                axs[2+i+j].legend(loc="upper left")

            start_xlim = tiles_df['datetime'][int(anomaly_start)] - timedelta(seconds=anomaly_delta)
            end_xlim = tiles_df['datetime'][anomaly_end] + timedelta(seconds=anomaly_delta)
            axs[-1].set_xlim(start_xlim, end_xlim)
            start_datetime = tiles_df['datetime'][int(anomaly_start)]
            stop_datetime = tiles_df['datetime'][anomaly_end]
            axs[-1].set_xlabel(f"datetime {start_datetime - timedelta(seconds=anomaly_delta)}")
            axs[0].set_title(f"Anomaly in ${', '.join(faces)}$ between {start_datetime} and {stop_datetime}")
            plt.tight_layout()
            figs.subplots_adjust(hspace=0)

            if save:
                figs.savefig(os.path.join(PLOT_TRIGGER_FOLDER_NAME, f"{'_'.join(faces)}_{tiles_df['datetime'][changepoint]}.png"), dpi=200)
            if show:
                plt.show()
            plt.close(figs)
            # if model:
            #     x_cols = [col for col in tiles_df.columns if col not in y_cols + y_pred_cols + ['datetime', 'TIME_FROM_SAA', 'SUN_IS_OCCULTED', 'LIVETIME', 'MET', 'START', 'STOP', 'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'SAA_EXIT', 'IN_SAA']]
            #     get_feature_importance(os.path.join(PLOT_TRIGGER_FOLDER_NAME, f"{face}_{tiles_df['datetime'][changepoint]}_lime.png"), inputs_outputs_df = tiles_df[changepoint:stopping_time], y_cols = y_cols, x_cols = x_cols, model = model, show=False, num_sample=10)
            #     get_feature_importance(os.path.join(PLOT_TRIGGER_FOLDER_NAME, f"{face}_{tiles_df['datetime'][stopping_time+50]}_lime.png"), inputs_outputs_df = tiles_df[changepoint+50:stopping_time+50], y_cols = y_cols, x_cols = x_cols, model = model, show=False, num_sample=10)


    @logger_decorator(logger)
    @staticmethod
    def show():
        '''Shows the plots'''
        plt.show()

    @logger_decorator(logger)
    @staticmethod
    def save(folder_name = '.', params = None, indexes = None):
        '''Saves the plots'''
        folder_name = f'{folder_name}/{params["model_id"]}' if params else folder_name
        for i in plt.get_fignums():
            title = plt.figure(i, figsize=(20, 12)).get_label()
            name = f'{title}.png' if not indexes else f'{title}_{indexes[0]}_{indexes[1]}.png'
            plt.savefig(f'{folder_name}/{name}' if not folder_name.endswith('png') else folder_name)
        plt.close('all')

if __name__ == '__main__':
    Plotter().show_models_params('data/model_nn_1', title = 'layers', features_dict={'1 layer': {'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.eq},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                     '2 layers': {'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                     '3 layers': {'units_1': {'value': 0, 'op': operator.gt},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}}}, show = False)
    Plotter().show_models_params('data/model_nn_1', title = '1 layer', features_dict={
                                                                                            'norm+dropout':
                                                                                                {'drop': {'value': 1, 'op': operator.eq},
                                                                                                 'norm': {'value': 1, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.eq},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'norm':
                                                                                                {'drop': {'value': 0, 'op': operator.eq},
                                                                                                 'norm': {'value': 1, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.eq},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'dropout':
                                                                                                {'drop': {'value': 1, 'op': operator.eq},
                                                                                                 'norm': {'value': 0, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.eq},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'no norm no drop':
                                                                                                {'drop': {'value': 0, 'op': operator.eq},
                                                                                                 'norm': {'value': 0, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.eq},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}}
                                                                                                                }, show = False)
    Plotter().show_models_params('data/model_nn_1', title = '2 layer', features_dict={
                                                                                            'norm+dropout':
                                                                                                {'drop': {'value': 1, 'op': operator.eq},
                                                                                                 'norm': {'value': 1, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'norm':
                                                                                                {'drop': {'value': 0, 'op': operator.eq},
                                                                                                 'norm': {'value': 1, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'dropout':
                                                                                                {'drop': {'value': 1, 'op': operator.eq},
                                                                                                 'norm': {'value': 0, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'no norm no drop':
                                                                                                {'drop': {'value': 0, 'op': operator.eq},
                                                                                                 'norm': {'value': 0, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.eq},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}}
                                                                                                                }, show = False)
    Plotter().show_models_params('data/model_nn_1', title = '3 layer', features_dict={
                                                                                            'norm+dropout':
                                                                                                {'drop': {'value': 1, 'op': operator.eq},
                                                                                                 'norm': {'value': 1, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.gt},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'norm':
                                                                                                {'drop': {'value': 0, 'op': operator.eq},
                                                                                                 'norm': {'value': 1, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.gt},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'dropout':
                                                                                                {'drop': {'value': 1, 'op': operator.eq},
                                                                                                 'norm': {'value': 0, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.gt},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}},
                                                                                            'no norm no drop':
                                                                                                {'drop': {'value': 0, 'op': operator.eq},
                                                                                                 'norm': {'value': 0, 'op': operator.eq},
                                                                                                 'units_1': {'value': 0, 'op': operator.gt},
                                                                                                 'units_2': {'value': 0, 'op': operator.gt},
                                                                                                 'units_3': {'value': 0, 'op': operator.gt}}
                                                                                                                }, show = False)


    # # Plotter.save('data/model_nn_1')
    Plotter().show()
    Plotter().show_models_params('data/model_nn_1', title='loss', features_dict={'loss':
                                                                                 {'units_1': {'value': 60, 'op': operator.lt},
                                                                                 'units_2': {'value': 60, 'op': operator.lt},
                                                                                 'units_3': {'value': 60, 'op': operator.lt}}})