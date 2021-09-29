# coding=UTF-8

import argparse
import glob
import os
import random

import cv2
import numpy as np

import darknet


# reload(sys)
# sys.setdefaultencoding('utf-8')


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                             "txt with paths to them, or a folder. Image valid"
                             " formats are jpg, jpeg or png."
                             "If no input is given, ")
    # parser.add_argument("--batch_size", default=1, type=int,
    #                     help="number of images to be processed at the same time")
    # parser.add_argument("--weights", default="./yolov3.weights",
    #                     help="yolo weights path")
    # parser.add_argument("--dont_show", action='store_true',
    #                     help="windown inference display. For headless systems")
    # parser.add_argument("--ext_output", action='store_true',
    #                     help="display bbox coordinates of detected objects")
    # parser.add_argument("--save_labels", action='store_true',
    #                     help="save detections bbox for each image in yolo format")
    # parser.add_argument("--config_file", default="./cfg/yolov3.cfg",
    #                     help="path to config file")
    # parser.add_argument("--data_file", default="./cfg/coco.data",
    #                     help="path to data file")
    # parser.add_argument("--thresh", type=float, default=.25,
    #                     help="remove detections with lower confidence")
    # return parser.parse_args()

    # parser.add_argument("--batch_size", default=1, type=int,
    #                     help="number of images to be processed at the same time")
    # parser.add_argument("--weights",
    #                     default="/media/vs/Extreme SSD/WJX/fire/FireBig_50000_20210525.weights",
    #                     help="yolo weights path")
    # parser.add_argument("--dont_show", action='store_true',
    #                     help="windown inference display. For headless systems")
    # parser.add_argument("--ext_output", action='store_true',
    #                     help="display bbox coordinates of detected objects")
    # parser.add_argument("--save_labels", action='store_true',
    #                     help="save detections bbox for each image in yolo format")
    # parser.add_argument("--config_file",
    #                     default="/mnt/Qi/wjx/fire_tiny/model/20210407/fire.cfg",
    #                     help="path to config file")
    # parser.add_argument("--data_file",
    #                     default="/mnt/Qi/wjx/fire_tiny/model/20210407/fire.data",
    #                     help="path to data file")
    # parser.add_argument("--thresh", type=float, default=.25,
    #                     help="remove detections with lower confidence")
    # return parser.parse_args()

    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights",
                        default="/media/vs/Data/darknet_train_result/darknet/Helmet/helmet_top/model/Helmat_head_v3t_1030000.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file",
                        default="/media/vs/Data/darknet_train_result/darknet/Helmet/helmet_top/model/HelmetTop.cfg",
                        help="path to config file")
    parser.add_argument("--data_file",
                        default="/media/vs/Data/darknet_train_result/darknet/Helmet/helmet_top/model/data.txt",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise (ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
               glob.glob(os.path.join(images_path, "*.png")) + \
               glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32) / 255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def bbox2point(bbox):
    x, y, w, h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax


def point2bbox(point):
    x1, y1, x2, y2 = point
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = (x2 - x1)
    h = (y2 - y1)

    return x, y, w, h


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    new_detections = []
    for detection in detections:
        pred_label, pred_conf, (x, y, w, h) = detection
        new_x = x / width * orig_w
        new_y = y / height * orig_h
        new_w = w / width * orig_w
        new_h = h / height * orig_h

        # 可以约束一下
        (x1, y1, x2, y2) = bbox2point((new_x, new_y, new_w, new_h))
        x1 = x1 if x1 > 0 else 0
        x2 = x2 if x2 < orig_w else orig_w
        y1 = y1 if y1 > 0 else 0
        y2 = y2 if y2 < orig_h else orig_h

        (new_x, new_y, new_w, new_h) = point2bbox((x1, y1, x2, y2))

        new_detections.append((pred_label, pred_conf, (new_x, new_y, new_w, new_h)))

    # image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), new_detections, class_colors


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x / width, y / height, w / width, h / height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections, = batch_detection(network, images, class_names,
                                          class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def main():
    args = parser()
    check_arguments_errors(args)
    random.seed(3)  # deterministic bbox colors
    print(args.config_file)
    print(args.data_file)
    print(args.weights)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )



    label_save_path = "/media/vs/Data/darknet_train_result/all_train/test0425/test.txt"
    oir_json_path = r'/home/wst/darknet_datareverse/coco.json'
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
    a = {}
    file1 = open(label_save_path,'w', encoding='utf-8')


    i = 0
    # j = 0

    #读取类别
    class_path = '/media/vs/Data/darknet_train_result/darknet/Helmet/helmet_top/model/name.txt'
    yes_path = "/media/vs/qi/data/SSH/24899/test_yes"
    result = []
    with open(class_path, "r") as f:
        lines = f.readlines()           #["{'bbb', 'aaa'}"]
        print(len(lines))
        print(lines)
        print(lines[0])
        # lines[0][1:-1].split(',')
        # result.append(lines)

    for root, dirs, files in os.walk(r"/media/vs/qi/data/SSH/24899/20210607"):  # 这里就填文件夹目录就可以了
        for file in files:
            # 获取文件路径
            if '.jpg' in file:
                src1 = os.path.join(root, file)
                # a = time.time()
                image, detections, class_colors = image_detection(
                    src1, network, class_names, class_colors, args.thresh
                )
                # b = time.time()
                # print(b-a)
                # print()
                # x_, y_ = image.shape[0:2]
                # with open(root + '/' + file[:-4] + '.txt', 'w') as k:

                if image is None:
                    continue

                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = darknet.draw_boxes(detections, image, class_colors)
                # dst1 = os.path.join(yes_path, file[:-4] + str(i) + '.jpg')
                # cv2.imwrite(dst1, image)

                for label, confidence, bbox in detections:
                    # for m in range(len(lines)):
                    # print(label)
                    for name in lines:
                        # print(11111111)
                        # print(type(name))
                        names = name.split('\n')[0]
                        if names in label:         #have \n'
                            print(2222222222)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            dst1 = os.path.join(yes_path, file[:-4] + '.jpg')
                            cv2.imwrite(dst1, image)
                            i += 1
                            a['imagename'] = file
                            a['classes'] = names
                            a['bbox'] = bbox
                            i += 1
                            print('i = ', i)
                            s = str(a)
                            print(s)
                            file1.write(s+'\n')
                            break

    file1.close()
    print('识别图片 %s 张' % i)


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
