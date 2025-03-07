# app.py

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

from aggregator_config import AGGREGATOR_DEFINITIONS, AGGREGATOR_PRIORITY_WEIGHTS
# Импортируем все нужные функции из services.py
from services import (
    load_and_clean_data,
    create_aggregator_dataframes,
    aggregated_search,
    predict_price_range,
    process_results,
    describe_uploaded_data,
    predict_price_with_rf  # <-- ключевая функция ML
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OFFERS_CSV = os.path.join(DATA_DIR, 'offers.csv')

app = Flask(__name__)
app.secret_key = "REPLACE_WITH_SECRET_KEY"

AGGREGATOR_DATA = {}
UPLOADED_DF = pd.DataFrame()

######################################
#  ИНИЦИАЛИЗАЦИЯ ДАННЫХ ПРИ СТАРТЕ
######################################
def init_data():
    """Читает offers.csv (если существует) и создаёт глобальный AGGREGATOR_DATA."""
    if os.path.exists(OFFERS_CSV):
        df_offers = load_and_clean_data(OFFERS_CSV)
        global AGGREGATOR_DATA
        AGGREGATOR_DATA = create_aggregator_dataframes(df_offers)
        print("Aggregator data ready.")
    else:
        print(f"File not found: {OFFERS_CSV} - aggregator not initialized.")

# Запускаем
init_data()

######################################
#  ГЛАВНАЯ СТРАНИЦА
######################################
@app.route('/')
def index():
    return render_template('index.html')

######################################
#  ОДНОРАЗОВЫЙ ПОИСК
######################################
@app.route('/search', methods=['POST'])
def search():
    """
    Обрабатывает форму поиска:
    1) Ищет через агрегаторы => Weighted Historical
    2) ML Prediction (RandomForest)
    3) Отображает results.html
    """
    size       = request.form.get('size') or None
    brand      = request.form.get('brand') or None
    model_     = request.form.get('model') or None
    load_speed = request.form.get('load_speed') or None

    # 1) Агрегаторный поиск
    results_df = aggregated_search(AGGREGATOR_DATA, size, brand, model_, load_speed)
    if results_df.empty:
        return render_template('results.html', message="No matches found for that input.")

    # 2) Weighted Historical
    prediction = predict_price_range(results_df)

    # 3) Формируем таблицу
    filtered_df, csv_file, xlsx_file = process_results(results_df, DATA_DIR)
    if filtered_df.empty:
        return render_template('results.html', message="Error processing results.")

    results_table_html = filtered_df.to_html(classes='table table-striped table-sm', index=False)
    search_criteria = f"{size or ''} {brand or ''} {model_ or ''} {load_speed or ''}".strip()

    # 4) ML Prediction (с доверительным интервалом)
    ml_prediction = predict_price_with_rf(size, brand, model_, load_speed)

    return render_template(
        'results.html',
        search_criteria=search_criteria,
        prediction=prediction,        # Weighted Historical dict
        ml_prediction=ml_prediction,  # dict с pred, ci_lower, ci_upper
        results_table=results_table_html,
        filtered_csv_link=url_for('download_file', filename=csv_file),
        filtered_xlsx_link=url_for('download_file', filename=xlsx_file)
    )

######################################
#  ЗАГРУЗКА CSV (BULK)
######################################
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files.get('file')
    if not file or file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))

    filepath = os.path.join(DATA_DIR, file.filename)
    file.save(filepath)

    # Допустим, CSV формат: uid, size, brand, model, loadspeed
    try:
        df_upload = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Error reading CSV: {e}")
        return redirect(url_for('index'))

    # Приводим к строковому типу
    for col in ['uid', 'size', 'brand', 'model', 'loadspeed']:
        if col in df_upload.columns:
            df_upload[col] = df_upload[col].astype(str)

    desc_info = describe_uploaded_data(df_upload)

    global UPLOADED_DF
    UPLOADED_DF = df_upload.copy()

    return render_template('bulk_upload_info.html', df_info=desc_info)

######################################
#  ПОКАЗ ПАКЕТНЫХ РЕЗУЛЬТАТОВ
######################################
@app.route('/bulk_results')
def bulk_results():
    """
    Для каждой строчки из UPLOADED_DF показывает
    1) Composite minPrice (Weighted)
    2) ML Price (только одно число)
    """
    global UPLOADED_DF, AGGREGATOR_DATA
    if UPLOADED_DF.empty:
        flash("No uploaded CSV data found. Please upload again.")
        return redirect(url_for('index'))

    rows = []
    for idx, row in UPLOADED_DF.iterrows():
        size_val       = row['size']
        brand_val      = row['brand']
        model_val      = row['model']
        load_speed_val = row['loadspeed']

        # Weighted aggregator
        results_df = aggregated_search(AGGREGATOR_DATA, size_val, brand_val, model_val, load_speed_val)
        if results_df.empty:
            w_predicted_price = None
        else:
            w_prediction = predict_price_range(results_df)
            w_predicted_price = w_prediction['weighted_composite']

        # ML (только одно число => берем 'pred' из dict)
        ml_dict = predict_price_with_rf(size_val, brand_val, model_val, load_speed_val)
        if ml_dict and isinstance(ml_dict, dict) and 'pred' in ml_dict:
            ml_price = ml_dict['pred']
        else:
            ml_price = None

        full_product = f"{size_val} {brand_val} {model_val} {load_speed_val}"
        rows.append({
            'uid': row['uid'],
            'full_product': full_product,
            'predicted_price': w_predicted_price,  # Weighted Composite
            'ml_price': ml_price                   # ML float
        })

    df_prices = pd.DataFrame(rows)
    return render_template('bulk_results.html', df_prices=df_prices)

######################################
#  ДЕТАЛЬНАЯ СТРАНИЦА UID
######################################
@app.route('/bulk_item/<uid>')
def bulk_item(uid):
    """
    Детальный вид для одной позиции:
    1) Weighted Historical
    2) ML Prediction (полный dict)
    """
    global UPLOADED_DF, AGGREGATOR_DATA
    if UPLOADED_DF.empty:
        flash("No uploaded CSV data found. Please upload again.")
        return redirect(url_for('index'))

    row = UPLOADED_DF.loc[UPLOADED_DF['uid'] == uid]
    if row.empty:
        flash(f"UID '{uid}' not found in uploaded data.")
        return redirect(url_for('index'))

    row_data = row.iloc[0]
    size_val       = row_data['size']
    brand_val      = row_data['brand']
    model_val      = row_data['model']
    load_speed_val = row_data['loadspeed']

    results_df = aggregated_search(AGGREGATOR_DATA, size_val, brand_val, model_val, load_speed_val)
    if results_df.empty:
        return render_template('results.html', message="No aggregator matches found for that item.")

    prediction = predict_price_range(results_df)
    filtered_df, csv_file, xlsx_file = process_results(results_df, DATA_DIR)
    if filtered_df.empty:
        return render_template('results.html', message="Error processing aggregator result.")

    results_table_html = filtered_df.to_html(classes='table table-striped table-sm', index=False)
    search_criteria = f"{size_val} {brand_val} {model_val} {load_speed_val}"

    # ML Prediction (полный словарь с доверительным интервалом)
    ml_prediction = predict_price_with_rf(size_val, brand_val, model_val, load_speed_val)

    return render_template(
        'results.html',
        search_criteria=search_criteria,
        prediction=prediction,
        ml_prediction=ml_prediction,
        results_table=results_table_html,
        filtered_csv_link=url_for('download_file', filename=csv_file),
        filtered_xlsx_link=url_for('download_file', filename=xlsx_file)
    )

######################################
#  DOWNLOAD FILE
######################################
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)

######################################
#  СТАРТ ПРИЛОЖЕНИЯ
######################################
if __name__ == '__main__':
    app.run(debug=True)
