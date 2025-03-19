import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Diminuir o frame para maior velocidade
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        
        # Carregar imagens
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} Imagens sendo codificadas.".format(len(images_path)))

        # Armazenar codificações e nomes das imagens
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Obter apenas o nome do arquivo a partir do caminho inicial
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Obter codificação
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Armazenar o nome do arquivo e a codificação
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Imagens codificadas carregadas")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Encontrar todos os rostos e codificações faciais no quadro atual do vídeo
        # Converter a imagem de cor BGR (utilizada pelo OpenCV) para cor RGB (utilizada pelo face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Verificar se o rosto é uma correspondência para os rostos conhecidos
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconhecido"

            # # Se uma correspondência foi encontrada em known_face_encodings, use apenas a primeira.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Ou, em vez disso, use o rosto conhecido com a menor distância para o novo rosto
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Converter para array numpy para ajustar rapidamente as coordenadas com o redimensionamento do frame
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
