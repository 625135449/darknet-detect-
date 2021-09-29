import os
import cv2
from time import sleep
from tqdm import tqdm

def draw_label(pic_len,select,or_path):
    # path = input('input label path(../labels):')                  #"/media/vs/Data/darknet_train_result/Truck0406/labels2"  #label address
    # path1 = input('input images path(../images):')                # "/media/vs/Data/darknet_train_result/Truck0406/Truck0406"  #pictures address
    # path2 = input('input drawed iamges path(../val_images):')      #"/media/vs/Data/darknet_train_result/Truck0406/test"   #drawed pictures address
    path = or_path + '/labels'
    path1 = select
    path2 = or_path + '/val'
    count = 0
    if not os.path.exists(path2):
        os.makedirs(path2)
    for L in tqdm(range(1)):
        for root, dirs, files in os.walk(path):    #pic address
            for name in files:
                if name.endswith(".txt"):
                    filename = root + "/" + name    #/media/wst/Data/darknet训练结果/darknet训练结果/TruckCover/darknet_275_closetransportation/train/labels/31801994.txt
                    file_name = name.split('.')[0]      #name = 31801994.txt      file_name = 31801994

                    # filename = path + "/" + file_name + ".txt"
                    # file_path = root + "/" + name

                    file_path = path1 + "/" + file_name + ".jpg"        #label对应的图片
                    img = cv2.imread(file_path)       #读入图片
                    h ,w = img.shape[:2]

                    f = open(filename, "r")
                    for each_line in f:
                        each_line_list = each_line.split()  # 将每一行的数字分开放在列表中   1 0.858911 0.570299 0.276238 0.314587   类别，中心点，宽比，高比
                        xmin = (float(each_line_list[1]) - (1/2) * (float(each_line_list[3]))) * w
                        ymin = (float(each_line_list[2]) - (1/2) * (float(each_line_list[4]))) * h
                        xmax = (float(each_line_list[1]) + (1/2) * (float(each_line_list[3]))) * w
                        ymax = (float(each_line_list[2]) + (1/2) * (float(each_line_list[4]))) * h

                        cls = str(each_line_list[0])
                        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

                        # cv2.rectangle(img, c1, c2, (0, 0, 255), 2)            #左上角，右下角，颜色，边框粗细
                        # cv2.putText(img, cls, c2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        #####test
                        if cls == '1':           #head
                            cv2.rectangle(img, c1, c2, (0, 255,0), 2)
                            cv2.putText(img,cls, c2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,255), 2)
                        if cls == '0':  #close
                            cv2.rectangle(img, c1, c2, (70, 130, 180), 2)
                            cv2.putText(img, cls, c2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        if cls == '2':  # open
                            cv2.rectangle(img, c1, c2, (139, 129, 76), 2)
                            cv2.putText(img, cls, c2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        if cls == '3':  # close
                                cv2.rectangle(img, c1, c2, (152, 124, 255), 2)
                                cv2.putText(img, cls, c2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        if cls == '4':  # close
                                cv2.rectangle(img, c1, c2, (100, 100, 100), 2)
                                cv2.putText(img, cls, c2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        else:
                            cv2.rectangle(img, c1, c2, (139, 100, 30), 2)
                            cv2.putText(img, cls, c2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        count += 1
                    cv2.imwrite(path2 + "/" + file_name + ".jpg", img)
                    f.close()
    sleep(0.2)
    print('绘制图片完成')
# draw_label()