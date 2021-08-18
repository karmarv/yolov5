"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys, os
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

device, model, class_names = 'cpu', None, None # cuda device, i.e. 0 or 0,1,2,3 or cpu
""" Initialize the model weights """
def init_model(weights='yolov5s.pt'):
    global device, model, class_names
    # Load model
    device = select_device(device)
    model = attempt_load(weights, map_location=device)   # load FP32 model
    # Get names and colors
    class_names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    print("Initialized Model. \nClass names: ", class_names, " \n")
    return device, model, class_names

""" loads an image from dataset, returns img, original hw, resized hw """
def load_image(image_path, img_size=640):
    if type(image_path) == str:
        img0 = cv2.imread(image_path)     # load file path as cv2 BGR
    else: 
        img0 = image_path                 # cv2 loaded image assigned
    assert img0 is not None, 'Image Not Found ' + image_path
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img0, (h0, w0), img, img.shape[:2]  # img, hw_original, hw_resized

""" Format the detection results """
def format_detect_result(path, img_og, img, preds, names):
    detection_results = []
    # Process detections per image
    for i, det in enumerate(preds):
        #print(i, " ---> ")
        if det is not None and len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_og.shape).round()
            # Write results
            for *xyxy, conf, clf in reversed(det):
                conf = conf.detach().numpy()
                (xmin, ymin, xmax, ymax) = torch.tensor(xyxy).view(1, 4).numpy()[0]
                box_list = list(map(int, [xmin, ymin, xmax, ymax]))
                detection_results.append({"rect": box_list, "conf": float(conf),"cls": int(clf)})
    return detection_results

@torch.no_grad()
def detect( image, 
            source='data/images',   # file/dir/URL/glob, 0 for webcam
            imgsz=640,              # inference size (pixels)
            conf_thres=0.25,        # confidence threshold
            iou_thres=0.45,         # NMS IOU threshold
            classes=None,           # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,     # class-agnostic NMS
            augment=False,          # augmented inference
            ):
    global device, model, class_names
    # Prepare image
    image_names = []
    img0s, img_hw0s, imgs = [], [], []
    if type(image) == str:
        image_names.append(image)
        img0, img_hw0, img, _ = load_image(image_path=image, img_size=imgsz)
        img0s.append(img0); img_hw0s.append(img_hw0); imgs.append(img)
    elif type(image) == list:
        for image_item in image:
            image_names.append(os.path.basename(image_item))
            img0, img_hw0, img, _ = load_image(image_path=image_item, img_size=imgsz)
            img0s.append(img0); img_hw0s.append(img_hw0); imgs.append(img)
    else:
        image_names.append(source)
        img0, img_hw0, img, _ = load_image(image_path=image, img_size=imgsz)
        img0s.append(img0); img_hw0s.append(img_hw0); imgs.append(img)
    
    # Process
    preds, formatted_results = [], []
    t0 = time.time()
    for path, img0, img_hw0, img in zip(image_names, img0s, img_hw0s, imgs):
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
        elapsed = time_synchronized() - t1
        formatted_results.append(format_detect_result(path, img0, img, pred, class_names))
        print('Time [Inference + NMS]: (%.3fs)' % (elapsed))
    print("\nAll Detections: {}".format(formatted_results))
    return formatted_results
    
@torch.no_grad()
def stream( source="http://192.168.4.25:8080/video",   # file/dir/URL/glob, 0 for webcam
            imgsz=640,              # inference size (pixels)
            conf_thres=0.25,        # confidence threshold
            iou_thres=0.45,         # NMS IOU threshold
            max_det=1000,           # maximum detections per image
            classes=None,           # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,     # class-agnostic NMS
            augment=False,          # augmented inference
            view_img=True,          # show results
            line_thickness=3        # bounding box thickness (pixels)
            ):
    global device, model, class_names

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                                    ('rtsp://', 'rtmp://', 'http://', 'https://'))
    stride = int(model.stride.max())  # model stride
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    bs = len(dataset)  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Process
    formatted_results = []
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        elapsed = time_synchronized() - t1
        
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {class_names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{class_names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

            # Print time (inference + NMS)
            print('Time [Inference + NMS]: (%.3fs)' % (elapsed))
        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        print(f'Done. ({time.time() - t0:.3f}s)')
    return pred


if __name__ == "__main__":
    pass