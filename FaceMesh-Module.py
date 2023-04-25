import cv2
import mediapipe as mp
import keyboard
import time


class FaceMeshDetector():
    def __init__(self,  static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks,
                                                 self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpec_color = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0) )
        self.drawSpec_border = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 0) )

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec_color, self.drawSpec_border)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
                    faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture('face.mp4')
    pTime = 0
    detector = FaceMeshDetector(max_num_faces=2)
    while True:
        if keyboard.is_pressed('q'):
            break;
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        # if len(faces) != 0:
        #     print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(3)


if __name__ == "__main__":
    main()
