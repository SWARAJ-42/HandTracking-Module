import cv2
import mediapipe as mp
import time
import keyboard


class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, min_detectionConf=0.5, min_trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.min_detectionConf=min_detectionConf
        self.min_trackConf=min_trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.min_detectionConf, self.min_trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, video, draw=True):
        self.videoRGB = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.videoRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(video, handLms, self.mpHands.HAND_CONNECTIONS,
                    self.mpDraw.DrawingSpec(color=(0, 0, 255)), self.mpDraw.DrawingSpec(color=(0, 255, 0)))
        
        return video
    


    def findPosition(self, video, handNo=0, draw=True):
            lmList = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = video.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(video, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            return lmList
    

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pTime = 0
    cTime = 0
    detector = HandDetector()
    while True:
        if keyboard.is_pressed('q'):
            break;
        success, video = cam.read()
        video = detector.find_hands(video)
        lmList = detector.findPosition(video, draw=False)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(video, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Video', video)
        cv2.waitKey(1)
        



if __name__ == "__main__":
    main()