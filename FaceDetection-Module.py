import cv2
import mediapipe as mp
import time
import keyboard


class FaceDetector:
    def __init__(self, min_detection_confidence=0.25, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.min_detection_confidence, self.model_selection)

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bbox_list = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_default = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                self.bbox_custom = int(bbox_default.xmin * w), int(bbox_default.ymin * h), \
                    int(bbox_default.width * w), int(bbox_default.height * h)
                bbox_list.append([id, self.bbox_custom, detection.score])
                cv2.rectangle(img, self.bbox_custom, (255, 0, 255), 2)
                cv2.putText(img, f"{str(int(detection.score[0] * 100))}%", (
                    self.bbox_custom[0], self.bbox_custom[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bbox_list

def main():
    cap = cv2.VideoCapture("./4.mp4")
    pTime = 0
    while True:
        if keyboard.is_pressed('q'):
            break
        success, img = cap.read()
        face = FaceDetector();
        img, bboxlist = face.findFace(img)
        print(bboxlist)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {str(int(fps))}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()