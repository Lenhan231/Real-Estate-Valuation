from .preprocessing import preprocess, mean_absolute_percentage_error, add_locality_features
from .plotting import plot_feature_importance, plot_pred_vs_actual

try:
    from .wandb_logging import log_to_wandb
    __all__ = [
        'preprocess',
        'mean_absolute_percentage_error',
        'add_locality_features',
        'plot_feature_importance',
        'plot_pred_vs_actual',
        'log_to_wandb',
    ]
except ImportError:
    # wandb is optional - only needed for training logging
    __all__ = [
        'preprocess',
        'mean_absolute_percentage_error',
        'add_locality_features',
        'plot_feature_importance',
        'plot_pred_vs_actual',
    ]
