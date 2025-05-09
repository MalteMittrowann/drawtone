from pythonosc.udp_client import SimpleUDPClient

# Ziel-IP und Port – anpassen!
ip = "172.20.10.14"  # IP-Adresse des Rechners, auf dem PureData läuft
port = 8000           # Port, auf dem PureData in [udpreceive 8000] hört

# OSC-Client einrichten
client = SimpleUDPClient(ip, port)

# Float-Wert zum Testen
wert = 50

# Sende den Float unter einer OSC-Adresse
client.send_message("/BPM", wert)
client.send_message("/morph", 1)

print(f"Float-Wert {wert} an {ip}:{port} gesendet.")
