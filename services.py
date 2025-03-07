# services.py

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from aggregator_config import AGGREGATOR_DEFINITIONS, AGGREGATOR_PRIORITY_WEIGHTS

############################################
#      1. DATA CLEAN/LOAD FUNCTIONS
############################################

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Загружает offers.csv, чистит данные и формирует вспомогательные колонки.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Приводим некоторые поля к строке (если они есть)
    for col in ['Size', 'Brand', 'Model', 'Load', 'Speed']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Удаляем неиспользуемые колонки
    columns_to_drop = [
        'To do', 'Steel', 'Condition', 'Ply', 'Quantity', 'Pmetric',
        'Tread', 'Antons raiting #1', 'Antons raiting #2', 'Antons raiting #3'
    ]
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Переименовываем некоторые колонки (если они есть)
    rename_map = {
        "Min Sale Price": "minPrice",
        "Min Sale Price URL": "minURL",
        "Min Sale Price Seller Name": "minPriceSeller",
        "AVG Sale Price": "AvgPrice",
        "MEDIAN Sale Price": "MedianPrice",
        "file_datetime": "datetime"
    }
    df.rename(columns=rename_map, inplace=True, errors="ignore")

    # Преобразуем minPrice в число
    if 'minPrice' in df.columns:
        df['minPrice'] = pd.to_numeric(df['minPrice'], errors='coerce')

    # Удаляем дубликаты, если есть 'Input String'
    if 'Input String' in df.columns:
        df.drop_duplicates(subset=['Input String'], keep='first', inplace=True)

    # Удаляем строки, где отсутствуют критические поля
    critical_cols = ['Model', 'Size', 'minPrice', 'Rim']
    for c in critical_cols:
        if c in df.columns:
            df.dropna(subset=[c], inplace=True)

    # Фильтр по minPrice
    if 'minPrice' in df.columns:
        df = df[df['minPrice'] > 1]

    # Создаем экземпляр модели, параметр nu определяет максимальную долю выбросов
    ocsvm = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale')
    # OneClassSVM ожидает вход в виде двумерного массива
    df['anomaly_ocsvm'] = ocsvm.fit_predict(df[['minPrice']])

    # Выбираем только нормальные наблюдения (метка 1)
    df = df[df['anomaly_ocsvm'] == 1].drop(columns='anomaly_ocsvm')

    # Создаём агрегаторные поля
    if {'Size', 'Brand', 'Model'}.issubset(df.columns):
        df['SBM'] = df['Size'].astype(str) + ' ' + df['Brand'].astype(str) + ' ' + df['Model'].astype(str)

    if {'Load', 'Speed'}.issubset(df.columns):
        df['LoadSpeed'] = df['Load'].astype(str) + df['Speed'].astype(str)
        df['SBMLoadSpeed'] = df['SBM'] + ' ' + df['LoadSpeed']

    return df

def create_aggregator_dataframes(df: pd.DataFrame) -> dict:
    """
    Группирует DataFrame по наборам полей из AGGREGATOR_DEFINITIONS
    и считает статистики. Возвращает словарь {aggregator_name: DataFrame}.
    """
    aggregator_data = {}

    for agg_name, group_cols in AGGREGATOR_DEFINITIONS.items():
        # Проверяем, что нужные колонки есть
        if not set(group_cols).issubset(df.columns):
            continue

        agg_price = df.groupby(group_cols).agg({
            'minPrice': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'AvgPrice': ['mean', 'median', 'std', 'min', 'max'],
            'minPriceSeller': ['nunique']
        }).round(2)

        # "Расплющиваем" мультииндекс
        agg_price.columns = ['_'.join(col).strip() for col in agg_price.columns.values]
        agg_price.reset_index(inplace=True)

        # Добавляем имя агрегатора
        agg_price['aggregator'] = agg_name

        aggregator_data[agg_name] = agg_price

    return aggregator_data

############################################
#      2. AGGREGATOR SEARCH
############################################

def aggregated_search(agg_data: dict,
                      size: str = None,
                      brand: str = None,
                      model: str = None,
                      load_speed: str = None) -> pd.DataFrame:
    """
    Ищет совпадения в каждом агрегаторе (AGGREGATOR_DEFINITIONS),
    возвращает объединённый DataFrame (все совпадения).
    """
    matched_frames = []
    inputs = {
        "Size": size,
        "Brand": brand,
        "Model": model,
        "LoadSpeed": load_speed
    }

    for agg_name, group_cols in AGGREGATOR_DEFINITIONS.items():
        if agg_name not in agg_data:
            continue

        # Если какое-либо из необходимых полей отсутствует, пропускаем
        if any(inputs.get(gc) is None for gc in group_cols):
            continue

        df_agg = agg_data[agg_name]
        # Формируем ключ (DataFrame) по group_cols
        row_dict = {gc: inputs[gc] for gc in group_cols}
        key_df = pd.DataFrame([row_dict])

        merged = key_df.merge(df_agg, on=group_cols, how='inner')
        if not merged.empty:
            merged['Match by'] = agg_name
            matched_frames.append(merged)

    if matched_frames:
        return pd.concat(matched_frames, ignore_index=True)
    else:
        return pd.DataFrame()

############################################
#      3. WEIGHTED PRICE PREDICTION
############################################

def predict_price_range(matched_df: pd.DataFrame) -> dict:
    """
    Возвращает три взвешенных показателя:
       1) average(minPrice_min)
       2) average( (minPrice_mean+minPrice_median)/2 )
       3) average(minPrice_max)
    Вес = (приоритет агрегатора) * (1 + minPrice_count / max_count).
    """
    if matched_df.empty:
        return {
            "weighted_min_minprice": None,
            "weighted_composite": None,
            "weighted_max_minprice": None
        }

    # Проверяем наличие необходимых полей
    necessary_cols = [
        'minPrice_count', 'minPrice_mean', 'minPrice_median',
        'minPrice_min', 'minPrice_max', 'Match by'
    ]
    for c in necessary_cols:
        if c not in matched_df.columns:
            matched_df[c] = np.nan

    matched_df['composite_score'] = ((matched_df['minPrice_mean'] + matched_df['minPrice_median']) / 2).round(2)

    def aggregator_weight(agg):
        return AGGREGATOR_PRIORITY_WEIGHTS.get(agg, 1.0)

    matched_df['layer1_agg_weight'] = matched_df['Match by'].apply(aggregator_weight)

    max_count = matched_df['minPrice_count'].max() or 1e-6
    matched_df['layer2_count_weight'] = 1.0 + (matched_df['minPrice_count'] / max_count)

    matched_df['final_weight'] = matched_df['layer1_agg_weight'] * matched_df['layer2_count_weight']
    valid_rows = matched_df[matched_df['final_weight'] > 0]
    if valid_rows.empty:
        return {
            "weighted_min_minprice": None,
            "weighted_composite": None,
            "weighted_max_minprice": None
        }

    wsum = valid_rows['final_weight'].sum()

    def wavg(col):
        return np.average(valid_rows[col], weights=valid_rows['final_weight'])

    weighted_min_minprice = wavg('minPrice_min')
    weighted_composite = wavg('composite_score')
    weighted_max_minprice = wavg('minPrice_max')

    return {
        "weighted_min_minprice": round(weighted_min_minprice, 2),
        "weighted_composite": round(weighted_composite, 2),
        "weighted_max_minprice": round(weighted_max_minprice, 2)
    }

############################################
#      4. PROCESS RESULTS (TABLE)
############################################

def process_results(matched_df: pd.DataFrame,
                    output_directory: str) -> (pd.DataFrame, str, str):
    """
    Подготавливает данные для HTML-таблицы (Aggregator Details).
    Возвращает (df_res, csv_filename, xlsx_filename).
    """
    if matched_df.empty:
        return pd.DataFrame(), None, None

    for c in ['Match by', 'minPrice_count', 'minPrice_min', 'minPrice_max', 'minPrice_mean', 'minPrice_median']:
        if c not in matched_df.columns:
            matched_df[c] = np.nan

    matched_df['Composite Score'] = ((matched_df['minPrice_mean'] + matched_df['minPrice_median']) / 2).round(2)

    df_res = pd.DataFrame()
    df_res['Aggregator'] = matched_df['Match by']

    def compute_queried_model(row):
        agg_name = row['Match by']
        group_cols = AGGREGATOR_DEFINITIONS.get(agg_name, [])
        values = [str(row[col]) for col in group_cols if col in row and pd.notnull(row[col])]
        return " ".join(values)

    df_res['Queried Item'] = matched_df.apply(compute_queried_model, axis=1)
    df_res['Events'] = matched_df['minPrice_count']
    df_res['Min minPrice'] = matched_df['minPrice_min'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    df_res['Average minPrice'] = matched_df['minPrice_mean'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    df_res['Max minPrice'] = matched_df['minPrice_max'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    df_res['Composite minPrice'] = matched_df['Composite Score'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")

    def aggregator_weight(agg):
        return AGGREGATOR_PRIORITY_WEIGHTS.get(agg, 1.0)
    df_res['AggPriority'] = df_res['Aggregator'].apply(aggregator_weight)
    df_res.sort_values('AggPriority', ascending=False, inplace=True)
    df_res.drop(columns=['AggPriority'], inplace=True)

    csv_file = "filtered_matched_results.csv"
    xlsx_file = "filtered_matched_results.xlsx"
    df_res.to_csv(os.path.join(output_directory, csv_file), index=False)
    df_res.to_excel(os.path.join(output_directory, xlsx_file), index=False)

    return df_res, csv_file, xlsx_file

############################################
#      5. DESCRIBE UPLOADED DATA
############################################

def describe_uploaded_data(df: pd.DataFrame) -> dict:
    info = {'shape': df.shape}
    for col in ['size', 'brand', 'model', 'loadspeed']:
        if col in df.columns:
            info[f'unique_{col}'] = df[col].nunique()
        else:
            info[f'unique_{col}'] = 0

    if 'size' in df.columns:
        info['top_10_size'] = df['size'].value_counts().head(10).to_dict()
    else:
        info['top_10_size'] = {}

    if 'brand' in df.columns:
        info['top_10_brand'] = df['brand'].value_counts().head(10).to_dict()
    else:
        info['top_10_brand'] = {}

    if 'model' in df.columns:
        info['top_10_model'] = df['model'].value_counts().head(10).to_dict()
    else:
        info['top_10_model'] = {}

    if 'loadspeed' in df.columns:
        info['top_10_loadspeed'] = df['loadspeed'].value_counts().head(10).to_dict()
    else:
        info['top_10_loadspeed'] = {}

    return info

############################################
#      6. ML MODEL LOADING / PREDICT
############################################

def parse_size(size_str: str):
    """Парсим '225/45R17' => (225.0, 45.0, 17.0) или (None, None, None)."""
    pattern = r'(\d{3})/(\d{2,3})[RZ]*(\d{2})'
    if not size_str:
        return None, None, None
    match = re.search(pattern, size_str.upper())
    if not match:
        return None, None, None
    section = float(match.group(1))
    aspect  = float(match.group(2))
    rim     = float(match.group(3))
    return section, aspect, rim

# Загрузим модель при старте. Если не хотим держать global, можно и в app.py.
# Но можно и здесь:
try:
    RF_BEST_PIPELINE = joblib.load("my_final_rf_model.joblib")
    print("Loaded RandomForest pipeline from my_final_rf_model.joblib")
except Exception as e:
    RF_BEST_PIPELINE = None
    print(f"Could not load 'my_final_rf_model.joblib': {e}")

def predict_price_with_rf(size_str, brand, model, load_speed):
    """
    Возвращает dict: {"pred": <float>, "ci_lower": <float>, "ci_upper": <float>}
    или None, если не удаётся предсказать (нет модели или не распарсили size).
    """
    if RF_BEST_PIPELINE is None:
        return None

    s, a, r = parse_size(size_str)
    if s is None or a is None or r is None:
        return None

    row_df = pd.DataFrame([{
        'Brand': brand if brand else 'missing',
        'Model': model if model else 'missing',
        'LoadSpeed': load_speed if load_speed else 'missing',
        'Aspect': a,
        'Rim': r,
        'Section': s
    }])

    try:
        # Чтобы получить доверительный интервал, смотрим предсказания каждого дерева
        forest = RF_BEST_PIPELINE['model']  # RandomForestRegressor
        X_trans = RF_BEST_PIPELINE['preprocessor'].transform(row_df)

        all_tree_preds = [est.predict(X_trans)[0] for est in forest.estimators_]
        preds_arr = np.array(all_tree_preds)
        mean_pred = preds_arr.mean()
        std_pred = preds_arr.std()

        ci_lower = mean_pred - 1.96 * std_pred
        if ci_lower < 0:
            ci_lower = 0
        ci_upper = mean_pred + 1.96 * std_pred

        return {
            "pred": float(mean_pred),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        }
    except Exception as ex:
        print(f"Error in predict_price_with_rf: {ex}")
        return None