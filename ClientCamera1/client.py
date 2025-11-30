import cv2
import socket
import pickle
import struct
import os
import subprocess
import numpy as np

# ---------------- Configuration ----------------
VIDEO_URL = (
    "https://videos-3.earthcam.com/fecnetwork/4280.flv/"
    "chunklist_w308697748.m3u8?"
    "t=Z8SpBH92uYu3pHPoOMZ+9zmlj1I9el/0SgTs+xGjkmvKXPhAoZp+FjbKk/3uiR8P"
    "&td=202511221250"
)

OUTPUT_FOLDER = "output_frames"     # Sauvegarde des frames annotées
VM_HOST = "13.51.81.21"             # IP de ton serveur cloud
VM_PORT = 5000                      # Port serveur
WIDTH, HEIGHT = 640, 360

# ---------------- Create folders ----------------
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ---------------- Connect to server ----------------
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((VM_HOST, VM_PORT))
print("[CLIENT] Connecté au serveur")

# ---------------- FFmpeg capture ----------------
command = [
    "ffmpeg",
    "-headers", "User-Agent: Mozilla/5.0\r\nReferer: https://www.earthcam.com/\r\n",
    "-i", VIDEO_URL,
    "-vf", f"scale={WIDTH}:{HEIGHT}",
    "-f", "image2pipe",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo", "-"
]

pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

frame_id = 0

# ---------------- Function to receive exact size ----------------
def receive_exact(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

# ---------------- Main Loop ----------------
try:
    while True:
        raw_image = pipe.stdout.read(WIDTH * HEIGHT * 3)
        if len(raw_image) != WIDTH * HEIGHT * 3:
            print("[CLIENT] Flux terminé ou erreur.")
            break

        frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        frame_id += 1

        # ---- Send frame to server ----
        data_send = pickle.dumps(frame)
        client_socket.sendall(struct.pack("!I", len(data_send)))
        client_socket.sendall(data_send)

        # ---- Receive annotated frame ----
        raw_len = receive_exact(client_socket, 4)
        if not raw_len:
            print("[CLIENT] Déconnexion du serveur.")
            break

        length = struct.unpack("!I", raw_len)[0]
        data_recv = receive_exact(client_socket, length)

        frame_processed = pickle.loads(data_recv)

        # ---- Save frame locally ----
        output_path = f"{OUTPUT_FOLDER}/frame_{frame_id}.jpg"
        cv2.imwrite(output_path, frame_processed)
        print(f"[CLIENT] Frame annotée sauvegardée : {output_path}")

except KeyboardInterrupt:
    print("[CLIENT] Arrêt manuel.")

finally:
    client_socket.close()
    pipe.terminate()
    print("[CLIENT] Connexion fermée")
