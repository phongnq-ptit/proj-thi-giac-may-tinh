import cv2
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from Pose import Pose


class Train:
    def __init__(self):
        self.pose = Pose()
        self.num_of_frames = 600  # số frame được train
        self.lm_list = []  # lưu lại các landmark

    def make_data(self, label):
        # Đọc  video
        cap = cv2.VideoCapture('./data/video/' + label + '.mp4')

        while len(self.lm_list) <= self.num_of_frames:
            ret, frame = cap.read()
            if ret:
                # Nhận diện pose
                frame, results = self.pose.detect_pose(frame)

                if results.pose_landmarks:
                    # Ghi nhận thông số khung xương
                    lm = self.pose.make_landmark_timestep(results)
                    self.lm_list.append(lm)
                    # Vẽ khung xương lên ảnh
                    frame = self.pose.draw_landmark_on_image(results, frame)

                cv2.putText(frame, 'FPS: ' + str(len(self.lm_list)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 123), 2, 2)
                cv2.imshow("Make data", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        #  viết vào file
        df = pd.DataFrame(self.lm_list)
        df.to_csv('./data/' + label + ".txt")
        cap.release()
        cv2.destroyAllWindows()

    def train_lstm(self):
        # Đọc dữ liệu
        nhintrom_df = pd.read_csv("./data/" + 'NHINTROM.txt')
        quayphao_df = pd.read_csv("./data/" + 'QUAYPHAO.txt')
        vietbai_df = pd.read_csv("./data/" + 'VIETBAI.txt')

        X = []
        y = []
        no_of_timesteps = 10

        dataset = nhintrom_df.iloc[:, 1:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(0)

        dataset = quayphao_df.iloc[:, 1:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(1)

        dataset = vietbai_df.iloc[:, 1:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i - no_of_timesteps:i, :])
            y.append(2)

        X, y = np.array(X), np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(optimizer="adam", metrics=['accuracy'], loss="binary_crossentropy")

        model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))
        model.save("model.h5")


file_video = ['./data/video/NHINTROM.mp4', './data/video/QUAYPHAO.mp4', './data/video/VIETBAI.mp4']
file_name = ['NHINTROM', 'QUAYPHAO', 'VIETBAI']
file_train = ['NHINTROM.txt', 'QUAYPHAO.txt', 'VIETBAI.txt']

train = Train()

# make data
# for i in range(0, 3, 1):
#     train.make_data(file_name[i], file_video[i])

# train.make_data('VIETBAI')

# train model
# for i in range(0, 3, 1):
#     train.train_lstm(file_train[i], i)

train.train_lstm()
