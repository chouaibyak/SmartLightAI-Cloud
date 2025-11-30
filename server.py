import socket
import pickle
import struct
from ultralytics import YOLO
import cv2
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

HOST = "0.0.0.0"
PORT = 5000

# Charger YOLO
yolo = YOLO("yolov8n.pt")

def receive_exact(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

def save_count_to_csv(count):
    with open("people_count.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), count])

def plot_people_count():
    try:
        df = pd.read_csv("people_count.csv", header=None, names=["timestamp", "count"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        plt.figure(figsize=(10,5))
        plt.plot(df["timestamp"], df["count"], marker='o', linestyle='-')
        plt.xlabel("Temps")
        plt.ylabel("Nombre de personnes")
        plt.title("Nombre de personnes détectées au fil du temps")
        plt.grid(True)
        plt.tight_layout()
        
        # Sauvegarde en fichier PNG
        plt.savefig("people_count_graph.png")
        print("[GRAPH] Graphique sauvegardé : people_count_graph.png")
        
        # Optionnel : afficher si interface graphique disponible
        # plt.show()
        
    except Exception as e:
        print("[GRAPH] Impossible de tracer le graphique :", e)

# ------------------- Serveur -------------------
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"[SERVER] Serveur démarré sur {HOST}:{PORT}")

try:
    while True:
        client, addr = server_socket.accept()
        print(f"[SERVER] Client connecté : {addr}")

        try:
            while True:
                # 1) Lire la taille
                raw_len = receive_exact(client, 4)
                if not raw_len:
                    print("[SERVER] Client déconnecté")
                    break

                length = struct.unpack("!I", raw_len)[0]

                # 2) Lire la frame
                data = receive_exact(client, length)
                if not data:
                    print("[SERVER] Client déconnecté")
                    break

                frame = pickle.loads(data)

                # YOLO demande BGR → OK
                results = yolo(frame, verbose=False)

                # Compter personnes
                person_count = 0
                annotated = frame.copy()

                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:  # class 0 = person
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, "Person", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Enregistrer dans CSV
                save_count_to_csv(person_count)

                # 3) Renvoyer la frame annotée
                data_send = pickle.dumps(annotated)
                client.sendall(struct.pack("!I", len(data_send)))
                client.sendall(data_send)

        except Exception as e:
            print("[SERVER] Erreur:", e)

        finally:
            client.close()
            print("[SERVER] Connexion fermée")

except KeyboardInterrupt:
    print("[SERVER] Arrêt manuel du serveur")

finally:
    server_socket.close()
    print("[SERVER] Serveur arrêté")
    # Générer le graphique à partir du CSV
    plot_people_count()
