"""Обучение детектора DDoS: загрузка датасета, отсев выбросов, PCA и XGBoost.

Конвейер построен так, чтобы оценка была честной (без утечки данных):
сначала отделяется тестовая выборка, и только потом на обучающей части
учатся отсев выбросов, стандартизация и PCA. Тест участвует лишь на этапе
`transform`/`predict`. После честной оценки финальные артефакты (scaler, PCA,
модель) переобучаются на всех данных и сохраняются для онлайн-детекции.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ddos_ml.features import (
    DATASET_DIR, LABEL_ENCODER_PATH, MODEL_PATH, PCA_N_COMPONENTS, PCA_PATH, SCALER_PATH,
)

# Признаки и метки честной оценки: train уже очищен от выбросов и преобразован,
# test преобразован тем же scaler/PCA, но в их обучении не участвовал.
EvalSplit = namedtuple("EvalSplit", ["X_train", "y_train", "X_test", "y_test", "scaler", "pca"])


def load_dataset():
    ddos_set = pd.DataFrame()
    ddos_set = pd.concat([
        ddos_set,
        pd.read_parquet(DATASET_DIR / 'Syn-training.parquet'),
        pd.read_parquet(DATASET_DIR / 'UDP-training.parquet'),
        pd.read_parquet(DATASET_DIR / 'DNS-testing.parquet')
    ])
    ddos_set = ddos_set[ddos_set['Label'] != 'MSSQL']
    ddos_set = ddos_set.drop_duplicates()
    ddos_set = ddos_set.dropna()
    print(ddos_set['Label'].value_counts())
    return ddos_set


def preprocess_dataset(dataset, low_unique_threshold=1, one_hot_threshold=3):
    cols_to_drop = []
    for col in dataset.columns:
        if dataset[col].nunique() <= low_unique_threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"Удалены столбцы с ≤ {low_unique_threshold} уникальными значениями: {cols_to_drop}")
        dataset.drop(columns=cols_to_drop, inplace=True)

    cols_to_encode = []
    for col in dataset.columns:
        unique_values = dataset[col].nunique()
        column_type = dataset[col].dtype
        if unique_values <= one_hot_threshold and column_type == 'object':
            cols_to_encode.append(col)

    if cols_to_encode:
        print(f"One-Hot кодирование применяется к столбцам: {cols_to_encode}")
        dataset = pd.get_dummies(dataset, columns=cols_to_encode)

    labels = ['Syn', 'DrDoS_DNS', 'UDP', 'Benign']
    balanced_dataset = pd.concat([dataset[dataset['Label'] == label].sample(n=3500) for label in labels], ignore_index=True)
    print(balanced_dataset['Label'].value_counts())
    return balanced_dataset


def outlier_mask(X: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> np.ndarray:
    """Булева маска инлайеров по числовым признакам (IsolationForest).

    Важно: лес обучается ровно на переданных строках. Чтобы оценка была честной,
    при подготовке теста сюда передают только обучающую часть.
    """
    features = X.select_dtypes(include=['number']).columns.tolist()
    iso_forest = IsolationForest(
        n_estimators=1000, max_samples='auto', contamination=contamination,
        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=random_state, verbose=0
    )
    return iso_forest.fit_predict(X[features]) == 1


def fit_preprocessors(X_train: pd.DataFrame):
    """Учит StandardScaler и PCA ТОЛЬКО на обучающих данных."""
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=PCA_N_COMPONENTS).fit(scaler.transform(X_train))
    return scaler, pca


def apply_preprocessors(scaler, pca, X) -> np.ndarray:
    """Применяет уже обученные scaler и PCA (только transform, без fit)."""
    return pca.transform(scaler.transform(X))


def prepare_eval_split(X: pd.DataFrame, y_encoded: np.ndarray,
                       test_size: float = 0.2, random_state: int = 42) -> EvalSplit:
    """Готовит честную оценку: split → fit на train → transform обеих частей.

    Тестовая выборка отделяется ПЕРВОЙ. Отсев выбросов, scaler и PCA учатся
    только на train; тест они лишь преобразуют. Так метрика не завышается.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    mask = outlier_mask(X_train, random_state=random_state)
    X_train, y_train = X_train[mask], y_train[mask]

    scaler, pca = fit_preprocessors(X_train)
    return EvalSplit(
        X_train=apply_preprocessors(scaler, pca, X_train), y_train=y_train,
        X_test=apply_preprocessors(scaler, pca, X_test), y_test=y_test,
        scaler=scaler, pca=pca,
    )


def build_model():
    """Конфигурация XGBoost-классификатора (общая для оценки и финального обучения)."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.1, min_child_weight=3,
        gamma=0.2, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, scale_pos_weight=1, tree_method='hist',
        grow_policy='lossguide', max_bin=256, predictor='auto',
        objective='multi:softmax', eval_metric='mlogloss', random_state=42,
        n_jobs=-1, verbosity=1
    )


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, le: LabelEncoder) -> None:
    """Печатает честные метрики на отложенном тесте и рисует матрицу ошибок."""
    y_pred = model.predict(X_test)
    print("Точность ~ ", accuracy_score(y_test, y_pred))
    print("\nОтчёт классификации:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.title('Матрица ошибок')
    plt.show()


def save_artifacts(model, pca, scaler, le: LabelEncoder) -> None:
    import joblib

    joblib.dump(model, MODEL_PATH)
    joblib.dump(pca, PCA_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print("Модель, PCA, стандартизатор и энкодер меток сохранены в файлы:\n"
          f"{MODEL_PATH}\n{PCA_PATH}\n{SCALER_PATH}\n{LABEL_ENCODER_PATH}")


def main():
    data = load_dataset()
    data = preprocess_dataset(data)

    X = data.drop(columns=['Label'])
    y = data['Label']

    # Кодировка меток — это просто отображение имён классов в числа, утечки нет.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- Честная оценка на отложенном тесте ---
    split = prepare_eval_split(X, y_encoded)
    eval_model = build_model()
    eval_model.fit(split.X_train, split.y_train)
    print("=== Честная оценка на отложенном тесте (артефакты этой модели не сохраняются) ===")
    evaluate(eval_model, split.X_test, split.y_test, le)

    # --- Финальные артефакты: переобучение на всех данных ---
    # Выбросы убираем со всего набора — честной оценки это уже не касается.
    mask = outlier_mask(X)
    X_full, y_full = X[mask], y_encoded[mask]
    final_scaler, final_pca = fit_preprocessors(X_full)
    final_model = build_model()
    final_model.fit(apply_preprocessors(final_scaler, final_pca, X_full), y_full)

    save_artifacts(final_model, final_pca, final_scaler, le)


if __name__ == '__main__':
    main()
