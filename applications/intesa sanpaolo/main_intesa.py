import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing
from math import log

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.plotter import Plotter
from modules.utils import Data

class Curve:
    '''
    From the original python implementation of
    FOCuS Poisson by Kester Ward (2021). All rights reserved.
    '''

    def __init__(self, k_T, lambda_1, t=0):
        self.a = k_T
        self.b = -lambda_1
        self.t = t

    def __repr__(self):
        return "({:d}, {:.2f}, {:d})".format(self.a, self.b, self.t)

    def evaluate(self, mu):
        return max(self.a * log(mu) + self.b * (mu - 1), 0)

    def update(self, k_T, lambda_1):
        return Curve(self.a + k_T, -self.b + lambda_1, self.t - 1)

    def ymax(self):
        return self.evaluate(self.xmax())

    def xmax(self):
        return -self.a / self.b

    def is_negative(self):
        # returns true if slope at mu=1 is negative (i.e. no evidence for positive change)
        return (self.a + self.b) <= 0

    def dominates(self, other_curve):
        return (self.a + self.b >= other_curve.a + other_curve.b) and (self.a * other_curve.b <= other_curve.a * self.b)

class Trigger:
    def __init__(self, tiles_df, y_cols, y_cols_pred, y_cols_raw, units, latex_y_cols):
        self.tiles_df = tiles_df
        self.y_cols = y_cols
        self.y_cols_raw = y_cols_raw
        self.y_cols_pred = y_cols_pred
        self.units = units
        self.latex_y_cols = latex_y_cols

    def focus_step_curve(self, curve_list, k_T, lambda_1):
        '''
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        '''
        if not curve_list:  # list is empty
            if k_T <= lambda_1:
                return [], 0., 0
            else:
                updated_c = Curve(k_T, lambda_1, t=-1)
                return [updated_c], updated_c.ymax(), updated_c.t

        else:  # list not empty: go through and prune

            updated_c = curve_list[0].update(k_T, lambda_1)  # check leftmost quadratic separately
            if updated_c.is_negative():  # our leftmost quadratic is negative i.e. we have no quadratics
                return [], 0., 0,
            else:
                new_curve_list = [updated_c]
                global_max = updated_c.ymax()
                time_offset = updated_c.t

                for c in curve_list[1:] + [Curve(0, 0)]:  # add on new quadratic to end of list
                    updated_c = c.update(k_T, lambda_1)
                    if new_curve_list[-1].dominates(updated_c):
                        break
                    else:
                        new_curve_list.append(updated_c)
                        ymax = updated_c.ymax()
                        if ymax > global_max:  # we have a new candidate for global maximum
                            global_max = ymax
                            time_offset = updated_c.t

        return new_curve_list, global_max, time_offset

    def trigger_gauss_focus(self, signal, face, diff):
        '''
        From the original python implementation of
        FOCuS Poisson by Kester Ward (2021). All rights reserved.
        '''
        result = {f'{face}_offset': [], f'{face}_significance': []}
        curve_list = []
        for T in tqdm(signal.index, desc=face):
            x_t = signal[T]
            if diff[T].value > 2:
                curve_list = []
            curve_list, global_max, offset = self.focus_step_curve(curve_list, x_t, 1)
            result[f'{face}_offset'].append(offset)
            result[f'{face}_significance'].append(np.sqrt(2 * global_max))
        return result

    def run(self, thresholds: dict, type='z_score', save_anomalies_plots=True, support_vars=[], file=''):
        '''Run the trigger algorithm on the dataset.

        Args:
            `tiles_df` (pd.DataFrame): dataframe containing the data
            `y_cols` (list): list of columns to be used for the trigger
            `y_pred_cols` (list): list of columns containing the predictions
            `thresholds` (dict): thresholds dictionary for each signal

        Returns:
            dict: dict containing the anomalies
        '''
        triggs_dict = {}
        diff = self.tiles_df['DAT_DEFF'].diff()
        pool = multiprocessing.Pool(8)
        results = []
        
        triggerer = self.trigger_gauss_focus
        print(f'Running {type} trigger algorithm...', end=' ')
        for face in self.y_cols:
            result = pool.apply_async(triggerer, (self.tiles_df[face], face, diff))
            results.append(result)

        for result in results:
            triggs_dict.update(result.get())
        pool.close()
        pool.join()

        triggs_df = pd.DataFrame(triggs_dict)
        triggs_df['DAT_DEFF'] = self.tiles_df['DAT_DEFF']
        return_df = triggs_df.copy()
        mask = False
        for face in self.y_cols:
            mask |= triggs_df[f'{face}_significance'] > thresholds[face]

        triggs_df = triggs_df[mask]

        Plotter(df = triggs_df).df_plot_tiles(x_col = 'DAT_DEFF',
                                                y_cols=triggs_df.columns,
                                                excluded_cols = ['DAT_DEFF'],
                                                smoothing_key='smooth',
                                                show = True,
                                                save = False)
        count = 0
        anomalies_faces = {face: [] for face in self.y_cols}
        old_stopping_time = {face: -1 for face in self.y_cols}

        for index, row in tqdm(triggs_df.iterrows(), total=len(triggs_df), desc='Identifying triggers'):
            for face in self.y_cols:
                if row[f'{face}_significance'] > thresholds[face]:
                    new_changepoint = row[f'{face}_offset'] + index
                    new_stopping_time = index + 1
                    new_datetime = str(row['DAT_DEFF'])

                    if (index == old_stopping_time[face] + 1 or new_changepoint <= old_stopping_time[face] + 1) and anomalies_faces[face]:
                        last_anomaly = anomalies_faces[face].pop()
                        old_changepoint = last_anomaly[1]
                        old_datetime = last_anomaly[3]
                        new_anomaly = (face, old_changepoint, new_stopping_time, old_datetime, new_datetime)
                    else:
                        count += 1
                        new_anomaly = (face, new_changepoint, new_stopping_time, new_datetime, new_datetime)

                    anomalies_faces[face].append(new_anomaly)
                    old_stopping_time[face] = new_stopping_time
        
        anomalies_list = []
        for face in self.y_cols:
            anomalies_list += anomalies_faces[face]

        print('Merging triggers...', end=' ')
        merged_anomalies = {}
        for face, start, stopping_time, start_datetime, stop_datetime in anomalies_list:
            if start in merged_anomalies:
                merged_anomalies[start][face] = {'start_index': start, 'stop_index': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime}
            else:
                merged_anomalies[start] = {face: {'start_index': start, 'stop_index': stopping_time, 'start_datetime': start_datetime, 'stop_datetime': stop_datetime}}
        print(f'{len(merged_anomalies)} anomalies in total.')
        print(merged_anomalies)
        detections_file_path = os.path.join('intesa sanpaolo', f'anomalies_{file}.csv')
        with open(detections_file_path, 'w') as f:
            f.write("start_datetime,stop_datetime,start_met,stop_met,triggered_faces\n")
            for anomaly_start, anomaly in sorted(merged_anomalies.items(), key=lambda x: int(x[0]), reverse=True):
                anomaly_end = -1
                for face in anomaly.values():
                    if face['stop_index'] > anomaly_end:
                        anomaly_end = face['stop_index']
                    if face['start_index'] < anomaly_start:
                        anomaly_start = face['start_index']
                anomaly_end -= 1
                f.write(f"{self.tiles_df['DAT_DEFF'][int(anomaly_start)]},{self.tiles_df['DAT_DEFF'][int(anomaly_end)]},{self.tiles_df['DAT_DEFF'][int(anomaly_start)]},{self.tiles_df['DAT_DEFF'][int(anomaly_end)]},{'/'.join(anomaly.keys())}\n")

        self.tiles_df = Data.merge_dfs(self.tiles_df[self.y_cols + ['DAT_DEFF']], return_df, on_column='DAT_DEFF')
        if save_anomalies_plots:
            Plotter(df = merged_anomalies).plot_anomalies(type, support_vars, thresholds, self.tiles_df, self.y_cols_raw, self.y_cols_pred, show=False)

        return merged_anomalies, return_df

df = pd.read_csv('applications/intesa sanpaolo/ts_premi_polizze_def.csv')
new_df = df.pivot_table(index='DAT_DEFF', columns='DES_PRODOTTO', values='SUM_of_NUM_NNETAMOUNT', aggfunc='sum')
new_df = new_df.reset_index()
new_df['DAT_DEFF'] = pd.to_datetime(new_df['DAT_DEFF'], format='%Y-%m-%d')
new_df = new_df.sort_index(ascending=True)
new_df = new_df.fillna(0)
new_df = new_df[['PRODOTTO PAPERINO', 'PRODOTTO CUCCIOLO', 'PRODOTTO PIPPO', 'PRODOTTO DOTTO', 'PRODOTTO BRONTOLO', 'DAT_DEFF']]  # alcune polizze di prova
y_cols = list(set(new_df.columns) - {'DAT_DEFF'})

for col in y_cols:
    new_df[f'{col}_smooth'] = new_df[col].rolling(30, center=True).mean()
    new_df[f'{col}_std'] = new_df[col].rolling(30, center=True).std()
new_df = new_df.fillna(0)
new_df = new_df.dropna()
y_cols_mean = [f'{col}_smooth' for col in y_cols]
y_cols_std = [f'{col}_std' for col in y_cols]
new_df = Data.get_masked_dataframe(data=new_df,
                                    start='2023-01-15 00:00:00',
                                    stop='2023-02-15 00:00:00',
                                    column='DAT_DEFF',
                                    reset_index=True)
Plotter(df = new_df, label = 'Inputs').df_plot_tiles(x_col = 'DAT_DEFF',
                                                    y_cols=y_cols,
                                                    excluded_cols = ['DAT_DEFF'],
                                                    smoothing_key='smooth',
                                                    show = True, 
                                                    latex_y_cols={})

for col in y_cols:
    new_df[col] = (new_df[col] - new_df[f'{col}_smooth']) / new_df[f'{col}_std']
Plotter(df = new_df, label = 'Inputs').df_plot_tiles(x_col = 'DAT_DEFF',
                                                    y_cols=y_cols,
                                                    excluded_cols = ['DAT_DEFF'],
                                                    smoothing_key='smooth',
                                                    show = True, 
                                                    latex_y_cols={})
thresholds = {col: 2 for col in y_cols}
# Trigger(new_df, y_cols, y_cols, y_cols, [], []).run(thresholds, type='focus', save_anomalies_plots=True, support_vars=[], file='intesa')
