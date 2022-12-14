import cv2
import numpy as np
import threading
import tensorflow as tf
from Pose import Pose


class Test:
    def __init__(self):
        self.pose = Pose()
        self.label = "Loading...."
        self.n_time_steps = 10  # số khung hình so sánh lần lượt
        self.lm_list = []

    def draw_class_on_image(self, img, fps):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, self.label, (10, 60), font, 1, (200, 255, 123), 2, 2)
        cv2.putText(img, 'FPS: ' + str(fps), (10, 30), font, 1, (100, 255, 123), 2, 2)
        return img

    def detect(self, model, lm_list):
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        print(lm_list.shape)
        results = model.predict(lm_list)
        print(results)
        if results[0][0] > 0.5:
            self.label = "SWING BODY"
        else:
            self.label = "SWING HAND"

    def test(self):
        model = tf.keras.models.load_model("model.h5")
        cap = cv2.VideoCapture(0)
        i = 0
        loading_frames = 60

        while True:
            _, frame = cap.read()
            frame, results = self.pose.detect_pose(frame)
            i = i + 1
            if i > loading_frames:
                if results.pose_landmarks:
                    c_lm = self.pose.make_landmark_timestep(results)

                    self.lm_list.append(c_lm)
                    if len(self.lm_list) == self.n_time_steps:
                        # predict
                        t1 = threading.Thread(target=self.detect, args=(model, self.lm_list,))
                        t1.start()
                        self.lm_list = []

                    frame = self.pose.draw_landmark_on_image(results, frame)

            frame = self.draw_class_on_image(frame, i)
            cv2.imshow("Test", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


test = Test()
test.test()

