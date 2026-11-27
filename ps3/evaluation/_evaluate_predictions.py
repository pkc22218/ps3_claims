import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_tweedie_deviance


def _lorenz_curve(y_true, y_pred, exposure):
    """Lorenz curve as in ps3_script: cumulative share of claims vs. share of policies.

    Policies are ordered from lowest to highest predicted risk.
    """
    ranking = np.argsort(y_pred)
    cumulated_claim_amount = np.cumsum(y_true[ranking] * exposure[ranking])
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


def evaluate_predictions(
    y_true, y_pred, sample_weight=None, tweedie_power=1.5, model_name="value"
):
    """Evaluate predictions against actuals, weighting each policy by exposure.

    Parameters
    ----------
    y_true : array-like
        Actual outcomes (e.g. pure premium).
    y_pred : array-like
        Predicted outcomes.
    sample_weight : array-like, optional
        Sample weights (e.g. exposure). Defaults to equal weights.
    tweedie_power : float, default 1.5
        Tweedie power used for the deviance (should match the model's objective).
    model_name : str, default "value"
        Column name of the returned DataFrame, handy when comparing models.

    Returns
    -------
    pd.DataFrame
        One column, with the metric names as index: mean_actual, mean_predicted,
        bias, deviance, mae, rmse and gini.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = (
        np.ones_like(y_true)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )

    # Exposure-adjusted means
    mean_actual = np.average(y_true, weights=w)
    mean_predicted = np.average(y_pred, weights=w)

    # Bias: relative deviation of the predicted mean from the actual mean
    # (> 0 means the model over-predicts on average, < 0 under-predicts)
    bias = (mean_predicted - mean_actual) / mean_actual

    deviance = mean_tweedie_deviance(
        y_true, y_pred, sample_weight=w, power=tweedie_power
    )

    errors = y_pred - y_true
    mae = np.average(np.abs(errors), weights=w)
    rmse = np.sqrt(np.average(errors**2, weights=w))

    # Gini as in the Lorenz curve plot of ps3_script: 1 - 2 * area under the curve
    ordered_samples, cum_claims = _lorenz_curve(y_true, y_pred, w)
    gini = 1 - 2 * auc(ordered_samples, cum_claims)

    return pd.DataFrame(
        {
            model_name: {
                "mean_actual": mean_actual,
                "mean_predicted": mean_predicted,
                "bias": bias,
                "deviance": deviance,
                "mae": mae,
                "rmse": rmse,
                "gini": gini,
            }
        }
    )
