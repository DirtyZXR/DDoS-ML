"""Общие константы проекта: пути к данным и моделям, схема признаков потока.

Пути строятся относительно корня репозитория, поэтому скрипты работают
из любой рабочей директории (например, `python -m ddos_ml.train`).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# Артефакты обучения. Создаются train.py, используются capture.py.
MODEL_PATH = PROJECT_ROOT / "xgb_model.pkl"
PCA_PATH = PROJECT_ROOT / "pca_model.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "label_encoder.pkl"

# Число главных компонент PCA, на которое обучены scaler/PCA/модель.
PCA_N_COMPONENTS = 20

# Признаки сетевого потока в порядке, который ожидают scaler и PCA.
# Тот же порядок используется при онлайн-захвате трафика (capture.py).
INPUT_FEATURES = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min",
    "Packet Length Max", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "SYN Flag Count", "RST Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "Down/Up Ratio", "Avg Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Act Data Packets", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]

# Имена компонент PCA — колонки матрицы признаков для классификатора.
PCA_FEATURES = [f"PC{i}" for i in range(1, PCA_N_COMPONENTS + 1)]
