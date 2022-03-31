import cv2
import numpy as np
import pyautogui
import ctypes

userScreen = ctypes.windll.user32
userScreen.SetProcessDPIAware()
ancho, alto = userScreen.GetSystemMetrics(0), userScreen.GetSystemMetrics(1)
print(ancho, alto)

while True:
    screenshot = pyautogui.screenshot(region=(0, 0, 1366, 768))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    cv2.imshow('Screenshot', screenshot)
    if cv2.waitKey(1) & 0xFF == 27:
       break

cv2.destroyAllWindows()