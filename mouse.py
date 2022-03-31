import cv2
import mediapipe as mp
import ctypes
import numpy as np
import pyautogui

userScreen = ctypes.windll.user32
userScreen.SetProcessDPIAware()
ancho, alto = userScreen.GetSystemMetrics(0), userScreen.GetSystemMetrics(1)
aspect_ratio = ancho / alto

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_mouse_pointer = (255, 0, 255)

xy_init = 100


def calcularDistancia(x1, y1, x2, y2):
    punto1 = np.array([x1, y1])
    punto2 = np.array(([x2, y2]))
    return np.linalg.norm(punto1 - punto2)

def click(hand_landmarks):
    dedo_abajo = False
    color_base = (255, 0, 112)
    color_index = (255, 200, 80)

    x_base = int(hand_landmarks.landmark[0].x * width)
    y_base = int(hand_landmarks.landmark[0].y * height)

    x_centro = int(hand_landmarks.landmark[9].x * width)
    y_centro = int(hand_landmarks.landmark[9].y * height)

    x_dedo = int(hand_landmarks.landmark[8].x * width)
    y_dedo = int(hand_landmarks.landmark[8].y * height)

    distancia_base = calcularDistancia(x_base, y_base, x_centro, y_centro)
    distancia_dedo = calcularDistancia(x_base, y_base, x_dedo, y_dedo)

    if distancia_dedo < distancia_base:
        dedo_abajo = True

    cv2.circle(output, (x_base, y_base), 5, (255, 0, 255), 2)
    cv2.circle(output, (x_dedo, y_dedo), 5, (255, 0, 255), 2)
    cv2.line(output, (x_base, y_base), (x_centro, y_centro), (255, 0, 255), 2)
    cv2.line(output, (x_base, y_base), (x_dedo, y_dedo), (255, 0, 255), 2)

    return  dedo_abajo

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Area de captura mano
        area_width = width - xy_init * 2
        area_height = int(area_width / aspect_ratio)
        aux = np.zeros(frame.shape, np.uint8)
        aux = cv2.rectangle(aux, (xy_init, xy_init), (xy_init + area_width, xy_init + area_height), (255, 125, 0), -1)

        output = cv2.addWeighted(frame, 1, aux, 0.8, 0)

        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks is not None:
            for hand_landmarks in result.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)

                xm = np.interp(x, (xy_init, xy_init + area_width), (0, ancho))
                ym = np.interp(x, (xy_init, xy_init + area_height), (0, alto))

                pyautogui.moveTo(int(xm), int(ym))

                if click(hand_landmarks):
                    pyautogui.click()

                cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
                cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)

        cv2.imshow('Output', output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()
