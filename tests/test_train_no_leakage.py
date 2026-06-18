"""Регресс на утечку данных в обучающем конвейере.

scaler/PCA честной оценки должны обучаться ТОЛЬКО на train: если кто-то снова
начнёт фитить препроцессоры на всём датасете до split, scaler увидит все строки
и `n_samples_seen_` станет равен размеру датасета — тест это поймает.
"""

import numpy as np
import pandas as pd

from ddos_ml.features import PCA_N_COMPONENTS
from ddos_ml.train import prepare_eval_split


def _synthetic_dataset(n_per_class: int = 60, n_features: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = ["Syn", "DrDoS_DNS", "UDP", "Benign"]
    frames = []
    for shift, label in enumerate(labels):
        block = rng.normal(loc=shift, scale=1.0, size=(n_per_class, n_features))
        df = pd.DataFrame(block, columns=[f"f{i}" for i in range(n_features)])
        df["Label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_eval_preprocessors_fit_on_train_only():
    data = _synthetic_dataset()
    X = data.drop(columns=["Label"])
    y = pd.factorize(data["Label"])[0]

    split = prepare_eval_split(X, y, random_state=0)

    # Главная проверка: scaler обучен строго меньше чем на всех строках (только train).
    assert split.scaler.n_samples_seen_ < len(X)
    # И не больше обучающей доли (test_size=0.2 → train ≈ 80%).
    assert split.scaler.n_samples_seen_ <= int(len(X) * 0.8)


def test_eval_split_reduces_to_pca_dimensionality():
    data = _synthetic_dataset()
    X = data.drop(columns=["Label"])
    y = pd.factorize(data["Label"])[0]

    split = prepare_eval_split(X, y, random_state=0)

    assert split.X_train.shape[1] == PCA_N_COMPONENTS
    assert split.X_test.shape[1] == PCA_N_COMPONENTS
    # Размеры train/test и меток согласованы.
    assert split.X_train.shape[0] == split.y_train.shape[0]
    assert split.X_test.shape[0] == split.y_test.shape[0]
