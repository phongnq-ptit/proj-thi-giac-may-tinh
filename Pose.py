import mediapipe as mp
import cv2


class Pose:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_pose(self, image):
        # Chuyển RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lay ket qua dau ra qua model
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(image, landmark_list=results.pose_landmarks,
                                           connections=self.mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                                           connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 0, 255),
                                                                                               thickness=2))

        return image, results

    def make_landmark_timestep(self, results):
        c_lm = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm

    def draw_landmark_on_image(self, results, img):
        # Vẽ các đường nối
        self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.mp_drawing_styles.get_default_pose_landmarks_style())
        return img
