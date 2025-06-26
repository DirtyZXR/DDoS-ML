import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
import netifaces
from collections import defaultdict
import time

# Параметры конфигурации
CAPTURE_DURATION = 5
MODEL_PATH = "xgb_model.pkl"
PCA_PATH = "pca_model.pkl"
SCALER_PATH = "scaler.pkl"
LE_PATH = "label_encoder.pkl"

def get_active_interface():
    try:
        interfaces = netifaces.interfaces()
        print("Доступные сетевые интерфейсы:", interfaces)
        for iface in interfaces:
            if iface != 'lo':
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    return iface
        print("Не найден активный интерфейс.")
        return None
    except Exception as e:
        print(f"Ошибка при поиске интерфейса: {e}")
        return None

INTERFACE = get_active_interface()
if INTERFACE is None:
    print("Программа завершена: не найден сетевой интерфейс.")
    exit(1)

# Загрузка моделей
model = joblib.load(MODEL_PATH)
pca = joblib.load(PCA_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LE_PATH)

# Список признаков
INPUT_FEATURES = [
    'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
    'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'Down/Up Ratio', 'Avg Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min'
]
PCA_FEATURES = [f'PC{i}' for i in range(1, 21)]

# Хранилище для потоков
flows = defaultdict(lambda: {
    'start_time': None, 'fwd_packets': 0, 'bwd_packets': 0, 'fwd_bytes': 0, 'bwd_bytes': 0,
    'fwd_packet_lengths': [], 'bwd_packet_lengths': [], 'protocol': None, 'packet_times': [],
    'fwd_packet_times': [], 'bwd_packet_times': [], 'fwd_psh_flags': 0, 'fwd_header_length': 0,
    'bwd_header_length': 0, 'syn_flags': 0, 'rst_flags': 0, 'ack_flags': 0, 'urg_flags': 0,
    'cwe_flags': 0, 'init_fwd_win_bytes': None, 'init_bwd_win_bytes': None,
    'fwd_act_data_packets': 0, 'fwd_seg_size_min': None, 'active_times': [], 'idle_times': []
})

def extract_features(packet):
    if IP not in packet:
        return
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    flow_key = (src_ip, dst_ip)
    current_time = time.time()

    if flows[flow_key]['start_time'] is None:
        flows[flow_key]['start_time'] = current_time
        flows[flow_key]['protocol'] = packet[IP].proto
        if TCP in packet:
            flows[flow_key]['init_fwd_win_bytes'] = packet[TCP].window
            flows[flow_key]['fwd_seg_size_min'] = 20

    is_forward = src_ip < dst_ip
    pkt_len = len(packet)
    pkt_time = current_time

    if is_forward:
        flows[flow_key]['fwd_packets'] += 1
        flows[flow_key]['fwd_bytes'] += pkt_len
        flows[flow_key]['fwd_packet_lengths'].append(pkt_len)
        flows[flow_key]['fwd_packet_times'].append(pkt_time)
        if TCP in packet:
            flows[flow_key]['fwd_header_length'] += len(packet[TCP])
            flows[flow_key]['fwd_psh_flags'] += 1 if packet[TCP].flags & 0x08 else 0
            flows[flow_key]['syn_flags'] += 1 if packet[TCP].flags & 0x02 else 0
            flows[flow_key]['rst_flags'] += 1 if packet[TCP].flags & 0x04 else 0
            flows[flow_key]['ack_flags'] += 1 if packet[TCP].flags & 0x10 else 0
            flows[flow_key]['urg_flags'] += 1 if packet[TCP].flags & 0x20 else 0
            flows[flow_key]['cwe_flags'] += 1 if packet[TCP].flags & 0x40 else 0
            if len(packet[TCP].payload) > 0:
                flows[flow_key]['fwd_act_data_packets'] += 1
    else:
        flows[flow_key]['bwd_packets'] += 1
        flows[flow_key]['bwd_bytes'] += pkt_len
        flows[flow_key]['bwd_packet_lengths'].append(pkt_len)
        flows[flow_key]['bwd_packet_times'].append(pkt_time)
        if TCP in packet:
            flows[flow_key]['bwd_header_length'] += len(packet[TCP])
            flows[flow_key]['init_bwd_win_bytes'] = packet[TCP].window if flows[flow_key]['init_bwd_win_bytes'] is None else flows[flow_key]['init_bwd_win_bytes']

    flows[flow_key]['packet_times'].append(pkt_time)

def process_flows():
    data = []
    current_time = time.time()

    for flow_key, flow_data in flows.items():
        if flow_data['start_time'] is None:
            continue
        duration = current_time - flow_data['start_time']
        if duration == 0:
            continue

        fwd_packet_lengths = flow_data['fwd_packet_lengths']
        bwd_packet_lengths = flow_data['bwd_packet_lengths']
        packet_lengths = fwd_packet_lengths + bwd_packet_lengths
        packet_times = flow_data['packet_times']
        fwd_packet_times = flow_data['fwd_packet_times']
        bwd_packet_times = flow_data['bwd_packet_times']

        flow_iat = np.array(np.diff(packet_times) if len(packet_times) > 1 else [0])
        fwd_iat = np.array(np.diff(fwd_packet_times) if len(fwd_packet_times) > 1 else [0])
        bwd_iat = np.array(np.diff(bwd_packet_times) if len(bwd_packet_times) > 1 else [0])

        features = {
            'Protocol': flow_data['protocol'],
            'Flow Duration': duration * 1000000,
            'Total Fwd Packets': flow_data['fwd_packets'],
            'Total Backward Packets': flow_data['bwd_packets'],
            'Fwd Packets Length Total': flow_data['fwd_bytes'],
            'Bwd Packets Length Total': flow_data['bwd_bytes'],
            'Fwd Packet Length Max': max(fwd_packet_lengths) if fwd_packet_lengths else 0,
            'Fwd Packet Length Min': min(fwd_packet_lengths) if fwd_packet_lengths else 0,
            'Fwd Packet Length Mean': np.mean(fwd_packet_lengths) if fwd_packet_lengths else 0,
            'Fwd Packet Length Std': np.std(fwd_packet_lengths) if fwd_packet_lengths else 0,
            'Bwd Packet Length Max': max(bwd_packet_lengths) if bwd_packet_lengths else 0,
            'Bwd Packet Length Min': min(bwd_packet_lengths) if bwd_packet_lengths else 0,
            'Bwd Packet Length Mean': np.mean(bwd_packet_lengths) if bwd_packet_lengths else 0,
            'Bwd Packet Length Std': np.std(bwd_packet_lengths) if bwd_packet_lengths else 0,
            'Flow Bytes/s': (flow_data['fwd_bytes'] + flow_data['bwd_bytes']) / duration if duration > 0 else 0,
            'Flow Packets/s': (flow_data['fwd_packets'] + flow_data['bwd_packets']) / duration if duration > 0 else 0,
            'Flow IAT Mean': np.mean(flow_iat) * 1000000 if len(flow_iat) > 0 else 0,
            'Flow IAT Std': np.std(flow_iat) * 1000000 if len(flow_iat) > 0 else 0,
            'Flow IAT Max': np.max(flow_iat) * 1000000 if len(flow_iat) > 0 else 0,
            'Flow IAT Min': np.min(flow_iat) * 1000000 if len(flow_iat) > 0 else 0,
            'Fwd IAT Total': np.sum(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0,
            'Fwd IAT Mean': np.mean(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0,
            'Fwd IAT Std': np.std(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0,
            'Fwd IAT Max': np.max(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0,
            'Fwd IAT Min': np.min(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0,
            'Bwd IAT Total': np.sum(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0,
            'Bwd IAT Mean': np.mean(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0,
            'Bwd IAT Std': np.std(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0,
            'Bwd IAT Max': np.max(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0,
            'Bwd IAT Min': np.min(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0,
            'Fwd PSH Flags': flow_data['fwd_psh_flags'],
            'Fwd Header Length': flow_data['fwd_header_length'],
            'Bwd Header Length': flow_data['bwd_header_length'],
            'Fwd Packets/s': flow_data['fwd_packets'] / duration if duration > 0 else 0,
            'Bwd Packets/s': flow_data['bwd_packets'] / duration if duration > 0 else 0,
            'Packet Length Min': min(packet_lengths) if packet_lengths else 0,
            'Packet Length Max': max(packet_lengths) if packet_lengths else 0,
            'Packet Length Mean': np.mean(packet_lengths) if packet_lengths else 0,
            'Packet Length Std': np.std(packet_lengths) if packet_lengths else 0,
            'Packet Length Variance': np.var(packet_lengths) if packet_lengths else 0,
            'SYN Flag Count': flow_data['syn_flags'],
            'RST Flag Count': flow_data['rst_flags'],
            'ACK Flag Count': flow_data['ack_flags'],
            'URG Flag Count': flow_data['urg_flags'],
            'CWE Flag Count': flow_data['cwe_flags'],
            'Down/Up Ratio': flow_data['bwd_packets'] / flow_data['fwd_packets'] if flow_data['fwd_packets'] > 0 else 0,
            'Avg Packet Size': np.mean(packet_lengths) if packet_lengths else 0,
            'Avg Fwd Segment Size': np.mean(fwd_packet_lengths) if fwd_packet_lengths else 0,
            'Avg Bwd Segment Size': np.mean(bwd_packet_lengths) if bwd_packet_lengths else 0,
            'Subflow Fwd Packets': flow_data['fwd_packets'],
            'Subflow Fwd Bytes': flow_data['fwd_bytes'],
            'Subflow Bwd Packets': flow_data['bwd_packets'],
            'Subflow Bwd Bytes': flow_data['bwd_bytes'],
            'Init Fwd Win Bytes': flow_data['init_fwd_win_bytes'] if flow_data['init_fwd_win_bytes'] is not None else 0,
            'Init Bwd Win Bytes': flow_data['init_bwd_win_bytes'] if flow_data['init_bwd_win_bytes'] is not None else 0,
            'Fwd Act Data Packets': flow_data['fwd_act_data_packets'],
            'Fwd Seg Size Min': flow_data['fwd_seg_size_min'] if flow_data['fwd_seg_size_min'] is not None else 0,
            'Active Mean': 0, 'Active Std': 0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0, 'Idle Std': 0, 'Idle Max': 0, 'Idle Min': 0
        }
        data.append(features)

    if not data:
        print("Нет данных для анализа.")
        return

    df = pd.DataFrame(data)
    for feature in INPUT_FEATURES:
        if feature not in df.columns:
            df[feature] = 0

    X_scaled = scaler.transform(df[INPUT_FEATURES])
    X_pca = pca.transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=PCA_FEATURES)

    predictions = model.predict(df_pca)
    predicted_labels = label_encoder.inverse_transform(predictions)

    print(f"\nРезультаты классификации (анализ за последние {CAPTURE_DURATION} сек):")
    for i, (label, flow_key) in enumerate(zip(predicted_labels, flows.keys())):
        print(f"Поток {i+1} ({flow_key[0]} -> {flow_key[1]}): {label}")

    flows.clear()

def capture_traffic():
    print(f"Начало захвата трафика на интерфейсе {INTERFACE}...")
    while True:
        try:
            sniff(iface=INTERFACE, prn=extract_features, timeout=CAPTURE_DURATION)
            process_flows()
        except Exception as e:
            print(f"Ошибка при захвате трафика: {e}")

if __name__ == "__main__":
    try:
        capture_traffic()
    except KeyboardInterrupt:
        print("\nЗахват трафика остановлен.")