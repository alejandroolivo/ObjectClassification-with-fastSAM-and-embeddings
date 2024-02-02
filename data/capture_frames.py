import cv2
import os

def capturar_frames(video_path, carpeta_destino='fruit_frames', num_frames=200):
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    # Capturar el video
    video = cv2.VideoCapture(video_path)

    # Total de frames en el video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcular el intervalo de frames a capturar
    intervalo = total_frames // num_frames

    # Capturar frames
    for i in range(num_frames):
        # Establecer la posición del frame actual
        video.set(cv2.CAP_PROP_POS_FRAMES, i * intervalo)
        
        # Leer el frame
        success, frame = video.read()

        # Verificar si el frame fue capturado con éxito
        if not success:
            break

        # Guardar el frame como imagen
        cv2.imwrite(os.path.join(carpeta_destino, f'frame_{i}.jpg'), frame)

    # Liberar el objeto de video
    video.release()

if __name__ == "__main__":
    # Ruta del archivo de video
    ruta_video = 'frames/videos/fruit.mp4'  # Reemplazar con la ruta de tu video

    # Capturar y guardar los frames
    capturar_frames(ruta_video)
