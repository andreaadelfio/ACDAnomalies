'''
Main file to run the project
Author: Andrea Adelfio
Created date: 03/02/2024
Modified date: 27/06/2024
TODO:
'''

import itertools
import sys
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modules.plotter import Plotter
from modules.utils import Data, File
from modules.config import BACKGROUND_PREDICTION_FOLDER_NAME
from modules.background import FFNNPredictor, RNNPredictor, MultiMedianKNeighborsRegressor, MultiMeanKNeighborsRegressor, get_feature_importance

from scripts.main_config import y_cols, y_cols_raw, y_pred_cols, x_cols, x_cols_excluded

def run_rnn(inputs_outputs, y_cols, y_cols_raw, cols_pred, x_cols):
    
    rnn = RNNPredictor(inputs_outputs, y_cols, x_cols, y_cols_raw, cols_pred, timestep=20)
    units_for_layers = itertools.product([90], [90], [90], [90], [90], [70], [50], [30])
    epochs_values = [10]
    bs_values = [1000]
    do_values = [0.02]
    norm_values = [0]
    drop_values = [0]
    opt_name_values = ['Adam']
    lr_values = [None]
    loss_type_values = ['mae']

    Plotter().plot_correlation_matrix(inputs_outputs, show=False, save=True)
    
    hyperparams_combinations = list(itertools.product(units_for_layers, norm_values,
                                                      drop_values, epochs_values,
                                                      bs_values, do_values, opt_name_values,
                                                      lr_values, loss_type_values))
    hyperparams_combinations = rnn.trim_hyperparams_combinations(hyperparams_combinations)
    # hyperparams_combinations = nn.use_previous_hyperparams_combinations(hyperparams_combinations)
    for model_id, units_for_layers, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
        params = {'model_id': model_id, 'units_for_layers': units_for_layers, 'norm': norm, 'drop': drop, 'epochs': epochs,
                  'bs': bs, 'do': do, 'opt_name': opt_name, 'lr': lr, 'loss_type': loss_type}
        rnn.set_hyperparams(params)
        rnn.create_model()
        history = rnn.train()
        if history.history['loss'][-1] > 0.0041:
            continue
        Plotter().plot_history(history, 'loss')
        Plotter().plot_history(history, 'accuracy')
        rnn.update_summary()
        Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params)
        for start, end in [(35000, 43000), (50000, 62000) , (507332, 568639)]:
            _, y_pred = rnn.predict(start=start, end=end, write=False)

            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(cols_pred, y_cols)}).drop(columns=y_cols)
            tiles_df = Data.merge_dfs(inputs_outputs[start:end][y_cols_raw + ['datetime', 'SOLAR']], y_pred)
            Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
                                                              show=False, smoothing_key='pred')
            for col in y_cols_raw:
                Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred')
            Plotter().plot_pred_true(tiles_df, cols_pred, y_cols_raw)
            Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params, (start, end))
        # if history.history['loss'][-1] < 0.0042:
        #     get_feature_importance(rnn.model_path, inputs_outputs, y_cols, x_cols, num_sample=10, show=False)

def run_ffnn(inputs_outputs, y_cols, y_cols_raw, cols_pred, x_cols):
    '''Runs the neural network model'''
    nn = FFNNPredictor(inputs_outputs, y_cols, x_cols, y_cols_raw, cols_pred)
    units_for_layers = itertools.product([90], [90], [90], [90], [90], [90], [90], [70], [50], [30])
    epochs_values = [1]
    bs_values = [1000]
    do_values = [0.02]
    norm_values = [0]
    drop_values = [0]
    opt_name_values = ['Adam']
    lr_values = [None, 0.00059, 0.00009]
    loss_type_values = ['mae']

    Plotter().plot_correlation_matrix(inputs_outputs, show=False, save=True)
    
    hyperparams_combinations = list(itertools.product(units_for_layers, norm_values,
                                                      drop_values, epochs_values,
                                                      bs_values, do_values, opt_name_values,
                                                      lr_values, loss_type_values))
    hyperparams_combinations = nn.trim_hyperparams_combinations(hyperparams_combinations)
    # hyperparams_combinations = nn.use_previous_hyperparams_combinations(hyperparams_combinations)
    for model_id, units_for_layers, norm, drop, epochs, bs, do, opt_name, lr, loss_type in hyperparams_combinations:
        params = {'model_id': model_id, 'units_for_layers': units_for_layers, 'norm': norm, 'drop': drop, 'epochs': epochs,
                  'bs': bs, 'do': do, 'opt_name': opt_name, 'lr': lr, 'loss_type': loss_type}
        nn.set_hyperparams(params)
        nn.create_model()
        history = nn.train()
        if history.history['loss'][-1] > 0.0040:
            continue
        Plotter().plot_history(history, 'loss')
        Plotter().plot_history(history, 'accuracy')
        nn.update_summary()
        Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params)
        for start, end in [('2024-03-10 12:08:00', '2024-03-10 12:30:00'),
                           ('2024-03-28 20:50:00', '2024-03-28 21:10:00'),
                           ('2024-05-08 20:30:00', '2024-05-08 23:40:00'),
                           ('2024-05-11 01:00:00', '2024-05-11 03:00:00'),
                           ('2024-05-15 14:15:00', '2024-05-15 15:40:00'),
                           ('2024-05-08 01:00:00', '2024-05-08 05:00:00')]:
            _, y_pred = nn.predict(start=start, end=end, write_bkg=False, save_plot=False)

            y_pred = y_pred.assign(**{col: y_pred[cols_init] for col, cols_init in zip(cols_pred, y_cols)}).drop(columns=y_cols)
            tiles_df = Data.merge_dfs(Data.get_masked_dataframe(data=inputs_outputs, start=start, stop=end)[y_cols_raw + ['datetime', 'SOLAR_a']], y_pred)
            Plotter(df=tiles_df, label='tiles').df_plot_tiles(x_col='datetime', marker=',',
                                                              show=False, smoothing_key='pred')
            for col in y_cols_raw:
                Plotter().plot_tile(tiles_df, face=col, smoothing_key = 'pred')
            Plotter().plot_pred_true(tiles_df, cols_pred, y_cols_raw)
            Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME, params, (start, end))
        # nn.predict(start=0, end=-1)
        # if history.history['loss'][-1] < 0.0040:
        #     get_feature_importance(nn.model_path, inputs_outputs, y_cols, x_cols, num_sample=10, show=False)

def run_multimean_knn(inputs_outputs, y_cols, x_cols):
    '''Runs the multi mean knn model'''
    multi_reg = MultiMeanKNeighborsRegressor(inputs_outputs, y_cols, x_cols)
    multi_reg.create_model(20)
    multi_reg.train()
    start, end = 60000, 63000
    _, y_pred = multi_reg.predict(start=start, end=end)
    y_pred = pd.DataFrame(y_pred, columns=y_cols)
    Plotter().plot_tile_knn(inputs_outputs[start:end].reset_index(), y_pred, face='Xpos')
    Plotter().show()
    Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME)

def run_multimedian_knn(inputs_outputs, y_cols, x_cols):
    '''Runs the multi median knn model'''
    multi_reg = MultiMedianKNeighborsRegressor(inputs_outputs, y_cols, x_cols)
    multi_reg.create_model(20)
    multi_reg.train()
    start, end = 60000, 73000
    _, y_pred = multi_reg.predict(start=start, end=end)
    y_pred = pd.DataFrame(y_pred, columns=y_cols)
    Plotter().plot_tile_knn(inputs_outputs[start:end].reset_index(), y_pred, face='Xpos')
    Plotter().show()
    Plotter.save(BACKGROUND_PREDICTION_FOLDER_NAME)

def run_median(inputs_outputs):
    '''Runs the multi median knn model'''

    mean_top = inputs_outputs['top'].mean()
    std_top = inputs_outputs['top'].std()
    for col in y_cols:
        mean_col = inputs_outputs[col].mean()
        std_col = inputs_outputs[col].std()

        inputs_outputs[f'{col}_orig'] = inputs_outputs[col]
        inputs_outputs[col] = (inputs_outputs[col] - mean_top) / std_top * std_col + mean_col
        inputs_outputs[col] = inputs_outputs[col].rolling(window=100).mean()

    median = np.median(inputs_outputs[['top', 'Xneg', 'Ypos', 'Yneg']], axis=1)
    y_median = pd.DataFrame(median, columns=['median'])
    y_median = y_median.rolling(window=50).mean()
    tiles_df = pd.concat([inputs_outputs[y_cols + ['datetime', 'SOLAR']], y_median], axis=1)
    mean_median = tiles_df['median'].mean()
    std_median = tiles_df['median'].std()
    for col in y_cols:
        mean_col = tiles_df[col].mean()
        std_col = tiles_df[col].std()
        tiles_df[f'{col}_smooth'] = (tiles_df['median'] - mean_median) / std_median * std_col + mean_col
        tiles_df[f'{col}_dif'] = inputs_outputs[f'{col}_orig'] - tiles_df['median']
    tiles_df.drop(columns=['median'], inplace=True)
    Plotter(df = tiles_df, label = 'median').df_plot_tiles(x_col = 'datetime', excluded_cols = [], marker = ',', show = True, smoothing_key='smooth')
    Plotter().show()

########### Main ############
if __name__ == '__main__':
    x_cols = [col for col in x_cols if col not in x_cols_excluded]
    inputs_outputs_df = File.read_dfs_from_weekly_pk_folder(start=819, stop=822)
    # inputs_outputs_df = Data.get_masked_dataframe(data=inputs_outputs_df,
    #                                               start=0,
    #                                               stop=20,
    #                                               column='index')

    # Plotter(df = inputs_outputs_df, label = 'Inputs and outputs').df_plot_tiles(x_col = 'datetime', excluded_cols = [], marker = ',', show = True, smoothing_key='smooth')

    run_rnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
    # run_ffnn(inputs_outputs_df, y_cols, y_cols_raw, y_pred_cols, x_cols)
    # run_multimean_knn(inputs_outputs_df, y_cols, x_cols)
    # run_multimedian_knn(inputs_outputs_df, y_cols, x_cols)
    # run_median(inputs_outputs_df, y_cols)
