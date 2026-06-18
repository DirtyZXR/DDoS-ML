import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

from ddos_ml.features import (
    DATASET_DIR, PCA_N_COMPONENTS,
    MODEL_PATH, PCA_PATH, SCALER_PATH, LABEL_ENCODER_PATH,
)

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

def detect_and_remove_outliers(data: pd.DataFrame, features: list = None, contamination: float = 0.05, random_state: int = 42):
    if features is None:
        features = data.select_dtypes(include=['number']).columns.tolist()

    iso_forest = IsolationForest(
        n_estimators=1000, max_samples='auto', contamination=contamination,
        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=random_state, verbose=0
    )
    X = data[features]
    outlier_pred = iso_forest.fit_predict(X)
    mask = outlier_pred == 1
    cleaned_data = data.loc[mask].reset_index(drop=True)

    print(f"Размер данных до удаления выбросов: {data.shape}")
    print(f"Размер данных после удаления выбросов: {cleaned_data.shape}")
    print(f"Удалено выбросов: {data.shape[0] - cleaned_data.shape[0]}")
    print(cleaned_data['Label'].value_counts())
    return cleaned_data

def work_PCA(data: pd.DataFrame, label_col='Label'):
    data = data.reset_index(drop=True)
    X = data.drop(columns=[label_col])
    y = data[label_col]

    scaler = StandardScaler()
    print("Признаки перед scaler.fit_transform:", X.columns.tolist())
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(8,5))
    plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
    plt.xlabel('Количество главных компонент')
    plt.ylabel('Накопленная объяснённая дисперсия')
    plt.title('Зависимость объяснённой дисперсии от числа компонент PCA')
    plt.grid(True)
    plt.show()

    n_components = PCA_N_COMPONENTS
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i}' for i in range(1, n_components + 1)]
    )
    df_pca = pd.concat([df_pca, y.reset_index(drop=True)], axis=1)
    print(df_pca.head())

    return pca, X_scaled, y, scaler  # Возвращаем scaler

def train_and_save_models(data: pd.DataFrame, scaler, pca, label_col='Label',
                          model_path=MODEL_PATH, pca_path=PCA_PATH,
                          scaler_path=SCALER_PATH, le_path=LABEL_ENCODER_PATH):
    X = data.drop(columns=[label_col])
    y = data[label_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.1, min_child_weight=3,
        gamma=0.2, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, scale_pos_weight=1, tree_method='hist',
        grow_policy='lossguide', max_bin=256, predictor='auto',
        objective='multi:softmax', eval_metric='mlogloss', random_state=42,
        n_jobs=-1, verbosity=1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Точность ~ ", accuracy_score(y_test, y_pred))
    print("\nОтчёт классификации:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.title('Матрица ошибок')
    plt.show()

    joblib.dump(model, model_path)
    joblib.dump(pca, pca_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    print(f"Модель, PCA, стандартизатор и энкодер меток сохранены в файлы:\n{model_path}\n{pca_path}\n{scaler_path}\n{le_path}")

if __name__ == '__main__':
    data = load_dataset()
    data = preprocess_dataset(data)
    data = detect_and_remove_outliers(data)
    pca, X_scaled, y, scaler = work_PCA(data, label_col='Label')
    n_components = PCA_N_COMPONENTS
    principal_components = pca.transform(X_scaled)
    df_pca = pd.DataFrame(principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    df_pca['Label'] = y.reset_index(drop=True)
    train_and_save_models(df_pca, scaler, pca, label_col='Label')