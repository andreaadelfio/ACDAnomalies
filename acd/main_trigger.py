'''
ACD-specific trigger detection script

This script demonstrates the use of the generic time series framework 
for ACD anomaly/trigger detection.

Author: Andrea Adelfio
Created date: 24/06/2024
Modified date: 13/09/2024
TODO:
'''

import os
from tslies.config import set_base_dir

set_base_dir('/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/applications/acd')
import numpy as np
import gc
gc.enable()

# Core framework modules
from tslies.background.rnnpredictor import RNNPredictor
from tslies.background.ffnnpredictor import FFNNPredictor
from tslies.background.pbnnpredictor import PBNNPredictor
from tslies.background.bnnpredictor import BNNPredictor
from tslies.utils import Data, File
from tslies.plotter import Plotter
from tslies.trigger import Trigger

# ACD-specific modules
from catalogs import CatalogsReader

from main_config import y_cols, y_cols_raw, y_pred_cols, x_cols, x_cols_excluded, units, y_smooth_cols, latex_y_cols, thresholds
import pandas as pd


def run_trigger_ffnn(inputs_outputs_df, y_cols, y_cols_raw, y_cols_pred, x_cols, file, catalog):
    '''Runs the model'''
    y_pred = None
    if y_pred is None or len(y_pred) == 0:
        nn = FFNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_raw, y_smooth_cols, y_cols_pred)
        nn.set_model(model_path='results/2025-05-26/background_prediction/1315/FFNNPredictor/0/model.keras', compile=False)
        nn.load_scalers()
        start, end = 0, -1
        _, y_pred = nn.predict(start, end, write_bkg=True, save_predictions_plot=False)
    
    y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    tiles_df = Data.merge_dfs(inputs_outputs_df, y_pred)
    # tiles_df = Data.get_masked_dataframe(data=tiles_df,
    #                                               start='2024-06-20 22:35:00',
    #                                               stop='2024-06-20 23:40:00', column='datetime').reset_index(drop=True)
    stats = []
    for face in y_cols:
        tiles_df[f'{face}_std'] = tiles_df[face].rolling(window=120, center=True).std()
        norm_face = (tiles_df[face] - tiles_df[f'{face}_pred']) / tiles_df[f'{face}_std']
        stats.append({
            'face': face,
            'std': round(norm_face.std(), 3),
            'mean': round(norm_face.mean(), 3)
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df.set_index('face').T.to_string(header=True))
    # for col in y_cols_raw:
    #     Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred', units=units, support_vars=['GOES_XRSA_HARD_EARTH_OCCULTED'])
    # Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', y_cols=y_cols, excluded_cols = [col for col in inputs_outputs_df.columns if col not in y_cols_pred + y_cols_raw + ['GOES_XRSA_HARD_EARTH_OCCULTED']], show = True, smoothing_key='pred')
    Trigger(tiles_df, y_cols_raw, y_cols_pred, y_cols_raw, units, latex_y_cols).run(thresholds, type='focus', save_anomalies_plots=True, support_vars=['GOES_XRSA_HARD_EARTH_OCCULTED'], file=file)

def run_trigger_pbnn(inputs_outputs_df, y_cols, y_cols_raw, y_cols_pred, x_cols, file, catalog):
    '''Runs the model'''
    nn = BNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_pred, latex_y_cols, units)
    nn.set_model(model_path='results/2025-10-24/background_prediction/1135/BNNPredictor/0/model.keras', compile=False)
    nn.load_scalers()
    # y_pred = File.read_df_from_file('results/2025-03-03/background_prediction/1644/BNNPredictor/0/pk/bkg')
    y_pred = None
    if y_pred is None or len(y_pred) == 0:
        start, end = 0, -1
        batch_size = len(inputs_outputs_df)
        for i in range(0, len(inputs_outputs_df), batch_size):
            _, y_pred = nn.predict(start=i, end=i + batch_size, write_bkg=True, num_batches=1, save_predictions_plot=False)
    tiles_df = Data.merge_dfs(inputs_outputs_df, y_pred)

    # tiles_df = Data.get_masked_dataframe(data=tiles_df,
    #                                               start='2024-06-20 22:35:00',
    #                                               stop='2024-06-20 23:40:00', column='datetime').reset_index(drop=True)
    # for col in y_cols_raw:
    #     Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred', units=units)
    # Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', y_cols=y_cols, excluded_cols = [col for col in inputs_outputs_df.columns if col not in y_cols_pred + y_cols_raw + ['GOES_XRSA_HARD_EARTH_OCCULTED']], show = True, latex_y_cols=latex_y_cols, units=units, smoothing_key='pred')

    for face, face_pred in zip(y_cols, y_pred_cols):
        tiles_df[f'{face}_norm'] = (tiles_df[face] - tiles_df[face_pred]) / tiles_df[f'{face}_std']
    reset = tiles_df['MET'].diff() > 60
    trigger = Trigger(tiles_df, y_cols, y_cols_pred, thresholds=thresholds, trigger_type='focus', units=units, latex_y_cols=latex_y_cols)
    trigger.run(reset_condition=reset)
    merged_anomalies, return_df = trigger.identify_and_merge_triggers(merge_interval=60)
    trigger.save_detections_csv_for_acd()
    trigger.plot_anomalies(merged_anomalies, return_df, support_vars=['GOES_XRSA_HARD_EARTH_OCCULTED'])

def run_trigger_bnn(inputs_outputs_df, y_cols, y_cols_raw, y_cols_pred, x_cols, file, catalog):
    '''Runs the model'''
    nn = BNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_pred, latex_y_cols, units)
    nn.set_model(model_path='results/2025-07-01/background_prediction/1224/BNNPredictor/0/model.keras', compile=False)
    nn.load_scalers()
    start, end = 0, -1
    batch_size = len(inputs_outputs_df)
    for i in range(0, len(inputs_outputs_df), batch_size):
        _, y_pred_bnn = nn.predict(start=i, end=i + batch_size, write_bkg=False, num_batches=1, save_predictions_plot=False)

    nn = FFNNPredictor(inputs_outputs_df, y_cols, x_cols, y_cols_raw, y_smooth_cols, y_cols_pred)
    nn.set_model(model_path='results/2025-05-26/background_prediction/1315/FFNNPredictor/0/model.keras', compile=False)
    nn.load_scalers()
    start, end = 0, -1
    _, y_pred_ffnn = nn.predict(start, end, write_bkg=True, save_predictions_plot=False)
    
    y_pred_ffnn = y_pred_ffnn.assign(**{col: y_pred_ffnn[cols_init] for col, cols_init in zip(y_cols_pred, y_cols)}).drop(columns=y_cols)
    # tiles_df = Data.merge_dfs(inputs_outputs_df, y_pred)

    tiles_df = inputs_outputs_df.copy()
    for face in y_cols:
        tiles_df[f'{face}_std'] = y_pred_bnn[f'{face}_std']
        tiles_df[f'{face}_pred'] = y_pred_ffnn[f'{face}_pred']
        norm_face = (tiles_df[face] - tiles_df[f'{face}_pred']) / tiles_df[f'{face}_std']
        print(f'Normality for {face}, std = {round(norm_face.std(), 3)}, mean = {round(norm_face.mean(), 3)}')
        
    # tiles_df = Data.get_masked_dataframe(data=tiles_df,
    #                                               start='2024-06-20 22:35:00',
    #                                               stop='2024-06-20 23:40:00', column='datetime').reset_index(drop=True)
    # for col in y_cols_raw:
    #     Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred', units=units)
    # Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', y_cols=y_cols, latex_y_cols=latex_y_cols, excluded_cols = x_cols + [f'{col}_smooth' for col in y_cols] + x_cols_excluded, show = True, smoothing_key='pred')
    reset = tiles_df['MET'].diff() > 60
    trigger = Trigger(tiles_df, y_cols, y_cols_pred, thresholds=thresholds, trigger_type='focus', units=units, latex_y_cols=latex_y_cols)
    trigger.run(reset_condition=reset)
    merged_anomalies, return_df = trigger.identify_and_merge_triggers(merge_interval=60)
    trigger.save_detections_csv_for_acd()
    trigger.plot_anomalies(merged_anomalies, return_df, support_vars=['GOES_XRSA_HARD_EARTH_OCCULTED'])

def run_trigger_mean(inputs_outputs_df, y_cols, y_cols_raw, y_cols_pred, x_cols, file, catalog):
    '''Runs the model'''
    tiles_df = inputs_outputs_df.copy()
    stats = []
    for face in y_cols:
        tiles_df[f'{face}_std'] = tiles_df[face].rolling(window=120, center=True, min_periods=1).std()
        tiles_df[f'{face}_pred'] = tiles_df[face].rolling(window=120, center=True, min_periods=1).mean()

        norm_face = (tiles_df[face] - tiles_df[f'{face}_pred']) / tiles_df[f'{face}_std']
        stats.append({
            'face': face,
            'std': round(norm_face.std(), 3),
            'mean': round(norm_face.mean(), 3)
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df.set_index('face').T.to_string(header=True))

    # tiles_df = Data.get_masked_dataframe(data=tiles_df,
    #                                     start='2024-01-28 01:50:32',
    #                                     stop='2024-01-29 22:10:32',
    #                                     column='datetime',
    #                                     reset_index=True)
    # for col in y_cols_raw:
    #     Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred', units=units)
    # Plotter(df = tiles_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', y_cols=y_cols_raw, excluded_cols = x_cols + [f'{col}_smooth' for col in y_cols_raw] + x_cols_excluded, show = True, smoothing_key='pred')
    reset = tiles_df['MET'].diff() > 60
    trigger = Trigger(tiles_df, y_cols, y_cols_pred, thresholds=thresholds, trigger_type='focus', units=units, latex_y_cols=latex_y_cols)
    trigger.run(reset_condition=reset)
    merged_anomalies, return_df = trigger.identify_and_merge_triggers(merge_interval=60)
    detections_df = trigger.get_detections_df(['MET'])
    trigger.save_detections_csv(detections_df=detections_df)

    detections_df, filtered_anomalies = trigger.filter_from_catalog(catalog, merged_anomalies, detections_df)
    trigger.save_detections_csv(detections_df=detections_df, suffix='_in_catalog')
    for x, v in filtered_anomalies.items():
        merged_anomalies[x] = v
    trigger.plot_anomalies(filtered_anomalies, return_df, support_vars=['GOES_XRSA_HARD_EARTH_OCCULTED'])

if __name__ == '__main__':
    # Read the catalog
    catalog = CatalogsReader().catalog_df
    print(catalog)
    x_cols = [col for col in x_cols if col not in x_cols_excluded]
    merge = 1
    for i in range(0, 1000, merge):
        inputs_outputs_df = File().read_dfs_from_weekly_pk_folder('/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/applications/acd/data/LAT_ACD/processed/new_with_correct_triggs/inputs_outputs_runs', start=i, stop=i+merge-1)
        # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
        #                                            start='2024-06-09 11:30:32',
        #                                            stop='2024-06-09 12:10:32',
        #                                            column='datetime',
        #                                            reset_index=True)
        if inputs_outputs_df is None or inputs_outputs_df.empty:
            continue
        
        # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [col for col in inputs_outputs_df.columns if col not in ['Xpos', 'SOLAR', 'SUN_IS_OCCULTED']], show = True, y_cols=y_cols, smoothing_key='smooth')
        run_trigger_mean(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols, i, catalog)
        # run_trigger_pbnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols, i, catalog)
        # run_trigger_ffnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols, i, catalog)
        gc.collect()