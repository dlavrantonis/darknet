from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    # parser.add_argument("--input", type=str, default='/mnt/testingDataSet/araxnes/',
    #                     help="video source. If empty, uses webcam 0 stream")
    # parser.add_argument("--input", type=str, default='/mnt/dionisis/Desktop/pagotaria/pisw',
    #                     help="video source. If empty, uses webcam 0 stream")
    # parser.add_argument("--input", type=str, default='/mnt/FalseDetectionsPolemi/tt1/',
    #                     help="video source. If empty, uses webcam 0 stream")
    # parser.add_argument("--input", type=str, default='/mnt/FalseDetectionsKalamatas/tt1',
    #                     help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--input", type=str, default='/mnt/FalseDetectionsKalamatas/33/dataset',
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")

    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")



    #MODEL
    # parser.add_argument("--weights", default="../yolov7/darknet/backupTGD/yolov7-tiny_final.weights", help="yolo weights path")
    # parser.add_argument("--config_file", default="../yolov7/darknet/cfgden/yolov7-tinyI.cfg",help="path to config file")
    # parser.add_argument("--data_file", default="../yolov7/darknet/data/objTheGreatDataset.data", help="path to data file")

    # parser.add_argument("--weights", default="../yolov7/darknet/backupPolemiCH/yolov7-tiny-6000_objectness_final.weights", help="yolo weights path")
    # parser.add_argument("--config_file", default="../yolov7/darknet/cfgden/yolov7-tinyI.cfg",help="path to config file")
    # parser.add_argument("--data_file", default="../yolov7/darknet/data/objPolemiCH.data", help="path to data file")

    # parser.add_argument("--weights", default="../yolov7/darknet/backupCombinedCH/yolov7-tiny-7000_objectness_final.weights", help="yolo weights path")
    # parser.add_argument("--config_file", default="../yolov7/darknet/cfgden/yolov7-tinyI.cfg",help="path to config file")
    # parser.add_argument("--data_file", default="../yolov7/darknet/data/objCombinedCH.data", help="path to data file")

    parser.add_argument("--weights", default="../yolov7/darknet/backupCombinedCH/yolov7-tiny-12000_objectness_final.weights", help="yolo weights path")
    parser.add_argument("--config_file", default="../yolov7/darknet/cfgden/yolov7-tinyI.cfg",help="path to config file")
    parser.add_argument("--data_file", default="../yolov7/darknet/data/objCombinedCH.data", help="path to data file")

    # parser.add_argument("--weights", default="../yolov7/darknet/backupKalamatasCH/yolov7-tiny-6000_objectness_final.weights", help="yolo weights path")
    # parser.add_argument("--config_file", default="../yolov7/darknet/cfgden/yolov7-tinyI.cfg",help="path to config file")
    # parser.add_argument("--data_file", default="../yolov7/darknet/data/objKalamatasCH.data", help="path to data file")

    # parser.add_argument("--weights", default="../yolov7/darknet/backupCombinedCH/yolov7-tiny-6000_final.weights", help="yolo weights path")
    # parser.add_argument("--config_file", default="../yolov7/darknet/cfgden/yolov7-tinyI.cfg",help="path to config file")
    # parser.add_argument("--data_file", default="../yolov7/darknet/data/objCombinedCH.data", help="path to data file")

    # parser.add_argument("--weights", default="backupCombinedCH/yolov4-tiny-objA6000_final.weights", help="yolo weights path")
    # parser.add_argument("--config_file", default="cfgden/yolov4-tiny-objI.cfg",help="path to config file")
    # parser.add_argument("--data_file", default="data/objCombinedCH.data", help="path to data file")




    parser.add_argument("--thresh", type=float, default=.65,
                        help="remove detections with confidence below this value")

    parser.add_argument("--output_folder", type=str, default='/mnt/FalseDetectionsKalamatas/results2/',
                        help="folder to put the snapshots")
    # parser.add_argument("--output_folder", type=str, default='/mnt/FalseDetectionsPolemi/14/dataset/',
    #                     help="folder to put the snapshots")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping



if __name__ == '__main__':

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    output_folder = args.output_folder
    #save_snapshots = args.save_snapshots

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(input_path):
        for file in f:
            for filetype in ('.jpg', '.jpeg','.png'):
                if filetype in file:
                    files.append(os.path.join(r, file))

    frn = 0

    bOnce = True
    for file in files:
        frame = cv2.imread(file)
        dim = (640, 360)

        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_LANCZOS4)

        frn += 1
        (video_height, video_width) = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, img_for_detect, thresh=args.thresh)
        fps = int(1 / (time.time() - prev_time))
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(img_for_detect)


        detections_adjusted = []

        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            if label=="person":
                detections_adjusted.append((str(label), confidence, bbox_adjusted))

        cv2.imwrite(output_folder+"sample" + str(frn) + ".jpg", frame)
        f = open(output_folder+"sample" + str(frn) + ".txt", "w")
        for det in detections_adjusted:
            if det[0] == "person":
                f.write("0 "
                        + str(det[2][0] / video_width) + " "
                        + str(det[2][1] / video_height) + " "
                        + str(det[2][2] / video_width) + " "
                        + str(det[2][3] / video_height) + "\n")
        f.close()
        if not args.dont_show:
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            cv2.imshow('Inference', image)
        # if args.out_filename is not None:
        #     video.write(image)
        if cv2.waitKey(1) == 27:
            break

        if frn ==72:
            bb=1
        time.sleep(0.8)




