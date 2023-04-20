import cv2
import mediapipe as mp
import time
import keyboard


class PoseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,
                                     self.model_complexity, self.smooth_landmarks, self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, video, draw=True):
        videoRGB = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(videoRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(video, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(0, 0, 255)), self.mpDraw.DrawingSpec(color=(0, 255, 0)))
        return video

    def findPosition(self, video, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = video.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(video, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pTime = 0
    cTime = 0
    detector = PoseDetector()
    while True:
        if keyboard.is_pressed('q'):
            break
        success, video = cam.read()
        video = detector.findPose(video)
        lmList = detector.findPosition(video)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(video, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Video', video)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
