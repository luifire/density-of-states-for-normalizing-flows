STD_NORMAL_CLASS = 6000  # often used when we want to plot a Gaussian
MAIN_TRAIN_CLS = 5000  # this from training class the train set
AVERAGE_OUTLIERS = 7000

# this is for accesing results
IMG_IDX = 'img'
PD_IDX = 'lln'
NO_VOLUME_PD = 'no_volume'
Z_IDX = 'z'
#MANIFOLD_DATA = 'manifold_data'
#MANIFOLD_DATA_WITH_NOISE = 'manifold_data_with_noise'
#MANIFOLD_LOG_PD = 'manifold_log_pd'
#EMBEDDED_DATA = 'embedded_data'
STD_NORM_Z_IDX = 'z_normed'
VOLUME_IDX = 'volume'
TARGET_IDX = 'target'
T_SCORE = 't_score'
CORRECTED_PD = 'corrected_llh'
CORRECTION_TERM = 'correction_term'
CORRECTED_NO_VOL_PD = 'corrected_no_volume_llh'
CORRECTION_TERM_NO_VOL = 'correction_term_no_vol'
T_SCORE_NO_VOLUME = 't_score_no_volume'

INFLATED_MANIFOLD_DATA = 'inflated_manifold_data'
EMBEDDED_DATA = 'embedded_data'
NONE_EMBEDDED_DATA = 'none_embedded_data'  # without rotation
FULL_INFLATED_LOG_PD = 'inflated_full_log_pd'
INFLATED_MANIFOLD_LOG_PD = 'inflated_manifold_log_pd'

#NO_INFLATION_VAR = 1.23e-10
LOG_NO_INFLATION_VAR = -1.23

#ALL_LOG_VARIANCES = [-6, -4, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, LOG_NO_INFLATION_VAR]
#ALL_LOG_VARIANCES = [-6, -4, -2, -1, -0.5, 0, 0.25, LOG_NO_INFLATION_VAR]
ALL_LOG_VARIANCES = [-6, -4, -2, -1, None]
#ALL_LOG_VARIANCES = [LOG_NO_INFLATION_VAR]#, -0.5, -0.25, 0, 0.25, LOG_NO_INFLATION_VAR]
MISES_ARGS = ' --mises_kappa=0.5 --sphere_radius=0'
SMOOTH_STEP_ARGS = ' --data_start=-5 --data_end=5 --data_offset=1 --data_range=10 --data_growth=0.5'


LEGEND_FONT_SIZE = 13
TITLE_FONT_SIZE = 15
AXIS_FONT_SIZE = 15

PLOT_LENGTH = 25
DEVICE = 'cuda'

"""
all_models = 
    {model: 
        {
            cls: [ [IMG_IDX = 1,2,3,4...], LLH_IDX = [....], ...]
        }
    }
"""
