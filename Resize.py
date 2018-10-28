import glob
import cv2

path_train = glob.glob('/home/haneul/Crosswalk_Image/train/img/*.jpg')
path_test = glob.glob('/home/haneul/Crosswalk_Image/test/img/*.jpg')

for path in path_train + path_test:
    cv2.imwrite(path, cv2.resize(cv2.imread(path),(300, 300), interpolation=cv2.INTER_AREA))