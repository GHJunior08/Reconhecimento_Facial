import cv2
from simple_facerec import SimpleFacerec

# Pegar as imagens da pasta
sfr = SimpleFacerec()
sfr.load_encoding_images('images/')

# Carregar a webcam
cap = cv2.VideoCapture(0)

# Dicionário para associar nomes a cargos
roles = {
    "Junior": "Ministro",
    "Giovani": "Diretor do Meio Ambiente",
    "Daniel": "Vice Diretor",
    "Weijin": "Permissão Geral"
}

while True:
    ret, frame = cap.read()

    # Detectar rostos
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    # Bloquear o acesso
    unlocked = False
   # Verificar se todos os rostos são conhecidos
    if face_names and all(name != "Desconhecido" for name in face_names):
        unlocked = True  # Desbloqueia apenas se todos os rostos forem conhecidos

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Verificar se há um cargo associado ao nome
        role = roles.get(name, "")
        display_name = f"{name} - {role}" if role else name

        # Mostrar o nome do rosto identificado com o cargo
        cv2.putText(frame, display_name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Exibir o estado de desbloqueio
    if unlocked:
        cv2.putText(frame, "Acesso permitido", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    else:
        if face_names:
            # Se houver rostos, mas algum for desconhecido
            cv2.putText(frame, "Bloqueado: rosto desconhecido", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        else:
            # Se não houver rostos
            cv2.putText(frame, "Bloqueado: nenhum rosto detectado", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    # Mostrar o frame
    cv2.imshow('Frame', frame)

    # Sair pressionando ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
