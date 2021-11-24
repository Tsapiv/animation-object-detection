import numpy as np
import cv2
from google_doodle_dataset import GoogleDoodleDataset
if __name__ == '__main__':
    dataset = GoogleDoodleDataset('google_doodle_dataset/data/')
    image, label = dataset[0:10]
    cv2.imshow('', image[0])
    cv2.waitKey(0)
