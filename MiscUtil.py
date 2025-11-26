import cv2
import matplotlib.pyplot as plt
import numpy as np

#function for graphing
#function for preprocessing
class Preprocessor:
    def __init__(self, frame, n_stack:int, resize_to=(106, 140)):
        self.n_stack = n_stack
        self.resize_to = resize_to
        frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_LINEAR)
        frame = frame/255.0



        # plt.imshow(frame)
        # plt.show()
        # print(frame.shape)
        self.prev_frames = [frame] * n_stack

    def get_obs(self, frame):
        frame = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_LINEAR)
        frame = frame/255.0

        self.prev_frames.append(frame)
        self.prev_frames = self.prev_frames[1:]

        return np.stack(self.prev_frames)

def clip_reward(reward):
    return np.sign(reward)

