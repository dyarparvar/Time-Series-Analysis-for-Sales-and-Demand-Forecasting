
import os
import yaml
import numpy as np
import joblib
import pickle
from src.utils import set_seed
from src.data import load_data, clean_data, get_book_sales
from src.models import (
    sarima_train, sarima_predict,
    prep_data_xgb, create_predictor_transformed_xgb, grid_search_predictor,
    prep_data_lstm, train_lstm_model, lstm_forecast, evaluate_final_lstm,
    parallel_results
)
from src.visualization import (
    plot_sarima_results, plot_sarima_results_zoom,
    plot_xgb_results, plot_xgb_results_zoom,
    plot_lstm_results, plot_lstm_results_zoom,
    plot_parallel_results, plot_parallel_results_zoom
)

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.unsafe_load(f)

SEED = config["randomness_strategy"]["seed"]
set_seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir = "/content/drive/My Drive/Career Accelerator/Course_3/topic_project_2/data/raw"
file_url_meta  = f"{data_dir}/ISBN_List.xlsx"
file_url_sales = f"{data_dir}/UK_Weekly_Trended_Timeline_from_200101_202429.xlsx"

raw_meta_data, raw_sales_data = load_data(file_url_meta, file_url_sales)
meta_data, sales_data = clean_data(raw_meta_data, raw_sales_data)

cutoff_date = config["books"]["cutoff_date"]
alchemist_sales   = get_book_sales(sales_data, meta_data, "Alchemist", cutoff_date)
caterpillar_sales = get_book_sales(sales_data, meta_data, "Very Hungry Caterpillar", cutoff_date, binding="Hardback")

# ── SARIMA ────────────────────────────────────────────────────────────────────
for title, data in [("alchemist", alchemist_sales), ("caterpillar", caterpillar_sales)]:
    arima_config = config["auto_arima"]["weekly_data"][title]
    model, train, test = sarima_train(data, title, arima_config)
    predictions, conf_int, fitted = sarima_predict(data, model, arima_config)
    plot_sarima_results(data, fitted, predictions, conf_int, title)
    plot_sarima_results_zoom(data, fitted, predictions, conf_int, title, arima_config)

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb_config = config["xgboost"]["weekly_data"]
for title, data in [("alchemist", alchemist_sales), ("caterpillar", caterpillar_sales)]:
    y_train, y_test = prep_data_xgb(data, "weekly", xgb_config["data"], title)
    predictor = create_predictor_transformed_xgb(xgb_config, title)
    predictions, best_model, best_params = grid_search_predictor(y_train, title, xgb_config, predictor)
    plot_xgb_results(data, y_test, predictions, title)
    plot_xgb_results_zoom(data, predictions, title, xgb_config["data"])

# ── LSTM ──────────────────────────────────────────────────────────────────────
lstm_config = config["lstm_config"]
for title, data in [("alchemist", alchemist_sales), ("caterpillar", caterpillar_sales)]:
    lstm_data = prep_data_lstm(data["units_sold"], lstm_config, title)
    result = train_lstm_model(
        lstm_data["X_train"], lstm_data["y_train"],
        lstm_data["X_val"],   lstm_data["y_val"],
        lstm_config, title, f"models/LSTM/{title}", SEED
    )
    y_test_original, predictions = lstm_forecast(
        lstm_data["X_test"], lstm_data["y_test"],
        result["model"], lstm_data["scaler"], title
    )
    evaluate_final_lstm(y_test_original, predictions, title)
    plot_lstm_results(data["units_sold"], y_test_original, predictions, predictions, lstm_config, title)
    plot_lstm_results_zoom(data["units_sold"], y_test_original, predictions, lstm_config, title)
