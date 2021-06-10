import cv2
import time
import numpy as npqq
import HandTrackingModule as htm
import warnings
warnings.filterwarnings("ignore")
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

w_cam, h_cam = 640, 480
pTime = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, w_cam)
cap.set(4, h_cam)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# vol_range = volume.GetVolumeRange()  # RANGE : ( -65.25, 0.0 )
# volmin = vol_range[0]
# volmax = vol_range[1]

detector = htm.HandDetection(detect_conf=0.7, max_hands=1)
volColor = (255, 0, 0)
volBar = 400
volPer = 0
while True:
    success, img = cap.read()

    img = detector.findHands(img=img)
    lmlist, bbox = detector.findPosition(img=img, draw=True)

    if len(lmlist) != 0:
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        area = area//100
        if 300 < area < 1000:
            length, img, coordinates = detector.findDistance(4, 8, img)
            volBar = np.interp(length, [50, 180], [400, 150])
            volPer = np.interp(length, [50, 180], [0, 100])
            smoothness = 5
            volPer = smoothness * round(volPer/smoothness)
            fingers = detector.fingersUp()
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                cv2.circle(img, (coordinates[4], coordinates[5]), 7, (0, 0, 255), cv2.FILLED)
                volColor = (0, 0, 255)
            else:
                volColor = (255, 0, 0)

    cv2.rectangle(img, (50, 150), (80, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (80, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, text=f'{int(volPer)}%', org=(55, 430), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=2)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'VOL SET : {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, volColor, thickness=3)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img=img, text=f'FPS: {str(int(fps))}', org=(20, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0), thickness=3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()