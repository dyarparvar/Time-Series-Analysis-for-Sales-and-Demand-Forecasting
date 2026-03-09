
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

palette_set2 = sns.color_palette("Set2")

def plot_acf_pacf(residuals, config):
    import statsmodels.graphics.api as smgraphics
    model_config = config["model"]
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    axes = axes.flatten()
    smgraphics.tsa.plot_acf(residuals, lags=4*model_config["m"], alpha=model_config["alpha"], ax=axes[0])
    smgraphics.tsa.plot_pacf(residuals, lags=4*model_config["m"], alpha=model_config["alpha"], ax=axes[1])

def plot_sarima_results(data, fitted, predictions, conf_int, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data["units_sold"], alpha=0.5)
    plt.plot(fitted.index, fitted, alpha=0.5)
    plt.plot(predictions.index, predictions, c=palette_set2[3])
    plt.plot(predictions.index, conf_int[:, 0], "--", c=palette_set2[7])
    plt.plot(predictions.index, conf_int[:, 1], "--", c=palette_set2[7])
    plt.legend(["Original", "Fitted", "Forecast", "95% CI"])
    plt.title(f"{title} - SARIMA")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_sarima_results_zoom(data, fitted, predictions, conf_int, title, config):
    plt.figure(figsize=(6, 3))
    actual_data = data["units_sold"][-config["forecast_horizon"]:]
    plt.plot(actual_data)
    plt.plot(predictions.index, predictions, c=palette_set2[3])
    plt.plot(predictions.index, conf_int[:, 0], "--", c=palette_set2[7])
    plt.plot(predictions.index, conf_int[:, 1], "--", c=palette_set2[7])
    plt.legend(["Original", "Forecast", "95% CI"])
    plt.title(f"{title} - SARIMA")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_xgb_results(data, y_test, predictions, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data["units_sold"], alpha=0.5, c=palette_set2[0])
    plt.plot(y_test, alpha=1, c=palette_set2[0])
    plt.plot(predictions.index, predictions, c=palette_set2[3])
    plt.legend(["Original (train)", "Original (test)", "Forecast"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_xgb_results_zoom(data, predictions, title, config):
    plt.figure(figsize=(6, 3))
    plt.plot(data["units_sold"][-config["forecast_horizon"]:])
    plt.plot(predictions.index, predictions, c=palette_set2[3])
    plt.legend(["Original (test)", "Forecast"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_lstm_results(data, y_test_original, fitted, predictions, config, title):
    last_prediction = np.array(predictions[-1]).flatten()
    last_actual = np.array(y_test_original[-1]).flatten()
    forecast_horizon = config["forecast_horizon"]
    train_end_indx = len(data) - forecast_horizon
    test_dates = data.index[train_end_indx:]
    fitted = np.concatenate([fitted[:, 0], fitted[-1, 1:]])
    fit_length = len(fitted)
    lookback = config["lookback"]
    fitted_dates = data.index[lookback:lookback + fit_length]
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[:train_end_indx], data.values[:train_end_indx], alpha=0.5, label="Original data (train)")
    plt.plot(fitted_dates, fitted, alpha=0.5, label="Fitted data (train)", c=palette_set2[1])
    plt.plot(test_dates, last_actual, label="Original data (test)", c=palette_set2[0])
    plt.plot(test_dates, last_prediction, label="Forecast", c=palette_set2[3])
    mae = mean_absolute_error(last_actual, last_prediction)
    mape = mean_absolute_percentage_error(last_actual, last_prediction)
    plt.legend(["Original (train)", "Fitted", "Original (test)", "Forecast"])
    plt.title(f"{title} - LSTM - MAE: {mae:.2f} - MAPE: {mape:.1%}")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_lstm_results_zoom(data, y_test_original, predictions, config, title):
    last_prediction = np.array(predictions[-1]).flatten()
    last_actual = np.array(y_test_original[-1]).flatten()
    forecast_horizon = config["forecast_horizon"]
    train_end_indx = len(data) - forecast_horizon
    test_dates = data.index[train_end_indx:]
    plt.figure(figsize=(6, 3))
    plt.plot(test_dates, last_actual, label="Original data (test)", c=palette_set2[0])
    plt.plot(test_dates, last_prediction, label="Forecast", c=palette_set2[3])
    mae = mean_absolute_error(last_actual, last_prediction)
    mape = mean_absolute_percentage_error(last_actual, last_prediction)
    plt.legend(["Original (test)", "Forecast"])
    plt.title(f"{title} - LSTM - MAE: {mae:.2f} - MAPE: {mape:.1%}")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_sequential_results(data, predictions, config, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data["units_sold"])
    forecast_indx = data["units_sold"][-config["forecast_horizon"]:].index
    plt.plot(forecast_indx, predictions, c=palette_set2[3])
    plt.legend(["Original", "Forecast"])
    plt.title(f"{title} - Sequential (SARIMA>>LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_sequential_results_zoom(data, predictions, config, title):
    plt.figure(figsize=(6, 3))
    plt.plot(data["units_sold"][-config["forecast_horizon"]:])
    forecast_indx = data["units_sold"][-config["forecast_horizon"]:].index
    plt.plot(forecast_indx, predictions, c=palette_set2[3])
    plt.legend(["Original", "Forecast"])
    plt.title(f"{title} - Sequential (SARIMA>>LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_parallel_results(data, predictions, sarima_weight, config, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data["units_sold"])
    forecast_indx = data["units_sold"][-config["forecast_horizon"]:].index
    plt.plot(forecast_indx, predictions, c=palette_set2[3])
    plt.legend(["Original", "Forecast"])
    plt.title(f"{title} - Parallel (SARIMA:LSTM) - ({sarima_weight}:{1-sarima_weight})")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

def plot_parallel_results_zoom(data, predictions, sarima_weight, config, title):
    plt.figure(figsize=(6, 3))
    plt.plot(data["units_sold"][-config["forecast_horizon"]:])
    forecast_indx = data["units_sold"][-config["forecast_horizon"]:].index
    plt.plot(forecast_indx, predictions, c=palette_set2[3])
    plt.legend(["Original", "Forecast"])
    plt.title(f"{title} - Parallel (SARIMA:LSTM) - ({sarima_weight}:{1-sarima_weight})")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")
