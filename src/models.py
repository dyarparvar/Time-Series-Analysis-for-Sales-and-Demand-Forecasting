
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import GlorotUniform


# ── SARIMA ───────────────────────────────────────────────────────────────────

def sarima_train(data, title, config):
    data_arima = data.loc[config["train_start"]:]["units_sold"].copy()
    train_data = data_arima.iloc[:-config["forecast_horizon"]]
    test_data = data_arima.iloc[-config["forecast_horizon"]:]

    if config["transformation"]:
        train_data = np.log1p(train_data)

    exog = None
    if config["exogenous"]:
        exog = pd.DataFrame(
            {"covid": ((train_data.index.year >= 2020) &
                       (train_data.index.year <= 2021)).astype(int)},
            index=train_data.index
        )

    model = auto_arima(train_data, X=exog, **config["model"])
    print(model.summary())
    return model, train_data, test_data


def sarima_predict(data, model, config):
    forecast_indx = data.index[-config["forecast_horizon"]:]
    exog_forecast = None
    if config["exogenous"]:
        exog_forecast = pd.DataFrame(
            {"covid": np.zeros(config["forecast_horizon"], dtype=int)},
            index=forecast_indx
        )

    predictions, conf_int = model.predict(
        n_periods=config["forecast_horizon"],
        return_conf_int=config["conf_int"],
        alpha=config["model"]["alpha"],
        X=exog_forecast
    )
    fitted = model.fittedvalues()

    if config["transformation"]:
        predictions = np.expm1(predictions)
        conf_int = np.expm1(conf_int)
        fitted = np.expm1(fitted)

    return predictions, conf_int, fitted


def sarima_residuals(residuals, config):
    from src.visualization import plot_acf_pacf
    plot_acf_pacf(residuals, config)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3, 2))
    plt.hist(residuals[1:], bins=40)


# ── XGBOOST ──────────────────────────────────────────────────────────────────

def prep_data_xgb(data, resolution, config, title):
    data_xgb = data.loc[config["train_start"]:].copy()

    if resolution == "weekly":
        data_xgb.index = pd.to_datetime(data_xgb.index).to_period("W-SAT")
    else:
        data_xgb.index = pd.to_datetime(data_xgb.index).to_period("M")

    y = data_xgb["units_sold"].copy()
    y_train = y.iloc[:-config["forecast_horizon"]]
    y_test = data_xgb["units_sold"].iloc[-config["forecast_horizon"]:]

    covid_mask = (y_train.index.year >= 2020) & (y_train.index.year <= 2021)
    y_train[covid_mask] = np.nan
    y_train = y_train.interpolate(method="linear")

    if config["model"] == "multiplicative":
        y_train = y_train.clip(lower=0.1)

    return y_train, y_test


def create_predictor_transformed_xgb(config, title):
    regressor = XGBRegressor(**config["regressor"])
    predictor_config = config["predictor"]
    model = predictor_config["model"][title]

    predictor = TransformedTargetForecaster([
        ("deseasonalize", Deseasonalizer(model=model, sp=predictor_config["sp"])),
        ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=predictor_config["degree"][title]))),
        ("forecast", make_reduction(regressor, window_length=predictor_config["window_length"][title], strategy=predictor_config["strategy"])),
    ])
    return predictor


def grid_search_predictor(y_train, title, config, predictor):
    gscv_config = config["grid_search"]
    cv = ExpandingWindowSplitter(initial_window=int(len(y_train) * gscv_config["initial_window"]))

    gscv = ForecastingGridSearchCV(
        predictor,
        cv=cv,
        scoring=gscv_config["scoring"],
        strategy=gscv_config["strategy"],
        param_grid=gscv_config["param_grid"][title],
        n_jobs=gscv_config["n_jobs"],
        backend=gscv_config["backend"],
        error_score=gscv_config["error_score"],
        verbose=gscv_config["verbose"],
    )
    gscv.fit(y=y_train)
    predictions = gscv.predict(fh=np.arange(1, config["data"]["forecast_horizon"] + 1))
    print(f"Best parameters - {title} - XGBoost: {gscv.best_params_}")
    return predictions, gscv.best_forecaster_, gscv.best_params_


# ── LSTM ─────────────────────────────────────────────────────────────────────

def prep_data_lstm(data, config, title):
    data_lstm = data.loc[config["train_start"]:].copy() if hasattr(data, "loc") else data.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_lstm_scaled = scaler.fit_transform(data_lstm.values.reshape(-1, 1))

    X, y = [], []
    lookback = config["lookback"]
    forecast_horizon = config["forecast_horizon"]

    for i in range(lookback, len(data_lstm_scaled) - forecast_horizon + 1):
        X.append(data_lstm_scaled[i - lookback:i, 0])
        y.append(data_lstm_scaled[i:i + forecast_horizon, 0])

    X = np.array(X).reshape((len(X), lookback, 1))
    y = np.array(y)

    X_train_full = X[:-forecast_horizon]
    y_train_full = y[:-forecast_horizon]
    X_test = X[-forecast_horizon:]
    y_test = y[-forecast_horizon:]

    n_val = int(len(X_train_full) * config["validation_split"])
    X_train = X_train_full[:-n_val]
    y_train = y_train_full[:-n_val]
    X_val = X_train_full[-n_val:]
    y_val = y_train_full[-n_val:]

    return {
        "X_train_full": X_train_full, "y_train_full": y_train_full,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
    }


def create_lstm_model(X_train, config, title, seed):
    initializer = GlorotUniform(seed)
    model = Sequential()

    for i, n_units in enumerate(config["n_neuron"]):
        if i == 0:
            model.add(LSTM(units=n_units, activation=config["hidden_activation"],
                           return_sequences=True, input_shape=(config["lookback"], 1),
                           kernel_initializer=initializer))
        else:
            model.add(LSTM(units=n_units, activation=config["hidden_activation"],
                           return_sequences=(i < len(config["n_neuron"]) - 1),
                           kernel_initializer=initializer))
        if config["dropout_rate"] > 0:
            model.add(Dropout(config["dropout_rate"]))

    model.add(Dense(config["forecast_horizon"], activation=config["output_activation"],
                    kernel_initializer=initializer))

    optimizer = Adam(learning_rate=config["learning_rate"]) if config["optimizer"] == "adam"         else RMSprop(learning_rate=config["learning_rate"])
    model.compile(loss=config["loss"], optimizer=optimizer, metrics=config["metrics"])
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, config, title, directory, seed):
    tf.random.set_seed(seed)
    cp = config["checkpoint_monitor"]
    os.makedirs(f"{directory}/checkpoint", exist_ok=True)

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=f"{directory}/tensorboard", histogram_freq=1),
        EarlyStopping(monitor=config["early_stoppying_monitor"], patience=config["patience"],
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(filepath=f"{directory}/checkpoint/{title}_best_model_{cp['loss'][0]}.keras",
                        monitor=cp["loss"][0], save_best_only=True, mode=cp["loss"][1], verbose=0),
        ModelCheckpoint(filepath=f"{directory}/checkpoint/{title}_best_model_{cp['mae'][0]}.keras",
                        monitor=cp["mae"][0], save_best_only=True, mode=cp["mae"][1], verbose=0),
    ]

    model = create_lstm_model(X_train, config, title, seed)
    history = model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"],
                        validation_data=(X_val, y_val), verbose=0, callbacks=callbacks)
    return {"model": model, "history": history}


def lstm_forecast(X_test, y_test, lstm_model, scaler, title):
    predictions_scaled = lstm_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled.flatten().reshape(-1, 1)).reshape(predictions_scaled.shape)
    y_test_original = scaler.inverse_transform(y_test.flatten().reshape(-1, 1)).reshape(y_test.shape)

    mae = mean_absolute_error(y_test_original.flatten(), predictions.flatten())
    mape = mean_absolute_percentage_error(y_test_original.flatten(), predictions.flatten())
    print(f"{title} - LSTM - Overall MAE: {mae:.2f} - MAPE: {mape:.2%}")
    return y_test_original, predictions


def evaluate_final_lstm(y_test_original, predictions, title):
    last_prediction = predictions[-1].flatten()
    last_actual = np.array(y_test_original[-1]).flatten()
    mae = mean_absolute_error(last_actual, last_prediction)
    mape = mean_absolute_percentage_error(last_actual, last_prediction)
    print(f"{title} - LSTM - Final forecast MAE: {mae:.2f} - MAPE: {mape:.2%}")


# ── HYBRID ───────────────────────────────────────────────────────────────────

def parallel_results(data, sarima_prediction, lstm_prediction, config, title):
    best_mae = float("inf")
    best_weight = None
    best_predictions = None
    best_mape = None

    for w in config["weights"]:
        current_predictions = w * sarima_prediction + (1 - w) * lstm_prediction
        current_mae = mean_absolute_error(
            data["units_sold"].iloc[-len(sarima_prediction):].values,
            current_predictions
        )
        current_mape = mean_absolute_percentage_error(
            data["units_sold"].iloc[-len(sarima_prediction):].values,
            current_predictions
        )
        if current_mae < best_mae:
            best_mae = current_mae
            best_mape = current_mape
            best_weight = w
            best_predictions = current_predictions

    print(f"{title} - Best weight SARIMA:LSTM ({best_weight}:{1-best_weight}) - MAE: {best_mae:.2f} - MAPE: {best_mape:.2%}")
    return best_predictions, best_weight, best_mae, best_mape
