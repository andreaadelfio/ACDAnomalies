'''
This is a config file to separate the dataframe columns for features and targets.
'''

h_names = ['hist_top', 'hist_Xpos', 'hist_Xneg', 'hist_Ypos', 'hist_Yneg']
# h_names = ['histNorm_top', 'histNorm_Xpos', 'histNorm_Xneg', 'histNorm_Ypos', 'histNorm_Yneg']

y_cols_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
y_smooth_cols = ['top_smooth', 'Xpos_smooth', 'Xneg_smooth', 'Ypos_smooth', 'Yneg_smooth']
# y_cols_raw = ['histNorm_top', 'histNorm_Xpos', 'histNorm_Xneg', 'histNorm_Ypos', 'histNorm_Yneg']
# y_smooth_cols = ['histNorm_top_smooth', 'histNorm_Xpos_smooth', 'histNorm_Xneg_smooth', 'histNorm_Ypos_smooth', 'histNorm_Yneg_smooth']
y_cols = y_smooth_cols
y_pred_cols = [col + '_pred' for col in y_cols_raw]

x_cols = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
    'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 
    'GEOMAG_LAT', 'LAMBDA', 'RA_SCZ', 'START', 'STOP', 'MET',
    'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'LIVETIME', 'DEC_SCZ', 'RA_SCX',
    'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE', 'ROCK_ANGLE',
    'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN',
    'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2',
    'SOLAR_a', 'SOLAR_b', 'SOLAR_a_EARTH_OCCULTED', 'SOLAR_b_EARTH_OCCULTED', 'TIME_FROM_SAA', 'SAA_EXIT']
x_cols_excluded = ['LIVETIME', 'SOLAR_a_EARTH_OCCULTED', 'SOLAR_b_EARTH_OCCULTED', 'STOP', 'MET']
# col_selected = inputs_outputs_df.columns

if __name__ == '__main__':
    vars_copy = vars().copy()
    filtered_vars = [var for var in vars_copy if '__' not in var]
    max_len = max(map(len, filtered_vars))

    for var in filtered_vars:
        print(f"{var:<{max_len}} {vars_copy[var]}")
