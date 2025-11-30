import cv2
import socket
import pickle
import struct
import os
import subprocess
import numpy as np
import threading

# ------------------- Camera URLs -------------------
CAMERA_URLS = {
    "cam1": (
        "https://videos-3.earthcam.com/fecnetwork/4280.flv/"
        "chunklist_w308697748.m3u8?"
        "t=Z8SpBH92uYu3pHPoOMZ+9zmlj1I9el/0SgTs+xGjkmvKXPhAoZp+FjbKk/3uiR8P"
        "&td=202511221250"
    ),

    "cam2": (
        "https://videos-3.earthcam.com/fecnetwork/4282.flv/"
        "playlist.m3u8?"
        "t=z96LKaYnWVQXaYYFAm8IpsH9JS%2BQlVEXBYPS02dZInHvJCygBHC2wktYgx%2FTexwl"
        "&td=202511241913"
    )
}

# ------------------- Server config -------------------
VM_HOST = "13.51.81.21"
VM_PORT = 5000

WIDTH, HEIGHT = 640, 360


# ------------------- Receive exact bytes -------------------
def receive_exact(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


# ------------------- Camera Thread Function -------------------
def handle_camera(cam_name, url):

    print(f"[THREAD] Démarrage : {cam_name}")

    # Output folder
    output_folder = f"output_{cam_name}"
    os.makedirs(output_folder, exist_ok=True)

    # 1) TCP connection
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((VM_HOST, VM_PORT))
    print(f"[{cam_name}] Connecté au serveur")

    # 2) FFmpeg process
    command = [
        "ffmpeg",
        "-headers", "User-Agent: Mozilla/5.0\r\nReferer: https://www.earthcam.com/\r\n",
        "-i", url,
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo", "-"
    ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

    frame_id = 0

    try:
        while True:
            raw_image = pipe.stdout.read(WIDTH * HEIGHT * 3)
            if len(raw_image) != WIDTH * HEIGHT * 3:
                print(f"[{cam_name}] Flux terminé.")
                break

            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            frame_id += 1

            # ---------------- Send frame ----------------
            data_send = pickle.dumps(frame)
            client_socket.sendall(struct.pack("!I", len(data_send)))
            client_socket.sendall(data_send)

            # ---------------- Receive processed frame ----------------
            raw_len = receive_exact(client_socket, 4)
            if not raw_len:
                print(f"[{cam_name}] Déconnexion serveur")
                break

            length = struct.unpack("!I", raw_len)[0]
            data_recv = receive_exact(client_socket, length)

            annotated = pickle.loads(data_recv)

            output_path = f"{output_folder}/frame_{frame_id}.jpg"
            cv2.imwrite(output_path, annotated)
            print(f"[{cam_name}] Frame sauvegardée : {output_path}")

    except KeyboardInterrupt:
        print(f"[{cam_name}] Arrêt manuel")

    finally:
        pipe.terminate()
        client_socket.close()
        print(f"[{cam_name}] Connexion fermée")


# ------------------- Start the 2 threads -------------------
threads = []

for cam_name, url in CAMERA_URLS.items():
    t = threading.Thread(target=handle_camera, args=(cam_name, url))
    t.start()
    threads.append(t)

# Wait for all threads
for t in threads:
    t.join()
