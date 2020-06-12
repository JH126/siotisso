import argparse
from time import time

import numpy as np
import cv2
import tensorflow as tf

from model import LPRNet
from loader import resize_and_normailze


classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "가", "나", "다", "라", "마", "거", "너", "더", "러",
              "머", "버", "서", "어", "저", "고", "노", "도", "로",
              "모", "보", "소", "오", "조", "구", "누", "두", "루",
              "무", "부", "수", "우", "주", "허", "하", "호"
              ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to image file")
    parser.add_argument("-w", "--weights", required=True, help="path to weights file")
    parser.add_argument("-s", "--save")
    args = vars(parser.parse_args())

    img_save_path = args["save"]+'.jpg'
    txt_save_path = args["save"]+'.txt'

    tf.compat.v1.enable_eager_execution()
    net = LPRNet(len(classnames) + 1)
    weights_path = args["weights"]

    net.load_weights(weights_path)
    
    img = cv2.imread(args["image"])
    x = np.expand_dims(resize_and_normailze(img), axis=0)
    t = time()
    result = net.predict(x, classnames)[0]
    #if nothing..? then result none..?
    #i think modify it..
    if (result != ''):
        cv2.imwrite(img_save_path, img)
        f = open(txt_save_path, "w")    
        f.write(result)
        f.close()
        print(result)
    else:
        print("none")
    #print(net.predict(x, classnames))
    #print(time() - t)

    
    
    #cv2.imshow("lp", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
