"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys, os
import time
from pathlib import Path

import cv2
import csv
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.general import is_ascii, xyxy2xywh, xywh2xyxy

device, model, class_names = 'cpu', None, None # cuda device, i.e. 0 or 0,1,2,3 or cpu

""" Initialize the model weights """
def init_model(weights=ROOT / 'yolov5s.pt', imgsz=640, dnn=False, half=False):
    global device, model, class_names
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)

    stride, class_names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
    print("Initialized Model. \nClass names: ", class_names, " ")
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



def plot_one_box(box, im, color=(128, 128, 128), txt_color=(255, 255, 255), label=None, line_width=3, use_pil=False):
    # Plots one xyxy box on image im with label
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    lw = line_width or max(int(min(im.size) / 200), 2)  # line width

    if use_pil or not is_ascii(label):  # use PIL
        im = Image.fromarray(im)
        draw = ImageDraw.Draw(im)
        draw.rectangle(box, width=lw + 1, outline=color)  # plot
        if label:
            font = ImageFont.truetype("Arial.ttf", size=max(round(max(im.size) / 40), 12))
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
            draw.text((box[0], box[1] - txt_height + 1), label, fill=txt_color, font=font)
        return np.asarray(im)
    else:  # use OpenCV
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            c2 = c1[0] + txt_width, c1[1] - txt_height - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        return im

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
    t1 = time_sync()
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
    for path, img0, img_hw0, img in zip(image_names, img0s, img_hw0s, imgs):
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        # Inference
        pred = model(img, augment=augment)[0]
        t3 = time_sync() 
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
        formatted_results.append(format_detect_result(path, img0, img, pred, class_names))
        # Print time (inference + NMS)
        t4 = time_sync()
        print('Prep:{0:3.1f}ms,\t  Infr:{1:3.1f}ms,\t  Post:{2:3.1f}ms, \t\t {3} detections in {4:3.1f}ms'.format((t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, len(formatted_results), (t4-t1)*1000))
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
            view_img=False,          # show results
            half=False,  # use FP16 half-precision inference
            line_thickness=3        # bounding box thickness (pixels)
            ):
    global device, model, class_names

    #view_img = check_imshow()
    dataset = LoadImages(source, img_size=imgsz, stride=model.stride, auto=True)

    if not view_img:
        f = open('{}.dump'.format(source), 'w')
        writer = csv.writer(f)
        header = ['frame_id', 'idx', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'conf_str']
        writer.writerow(header)

    # Process
    t0 = time.time()    
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Inference
        pred = model(im, augment=augment)
        t3 = time_sync() 
        dt[1] += t3 - t2
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            #p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()
            annotator = Annotator(imc, line_width=line_thickness, example=str(class_names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {class_names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{class_names[c]} {conf:.2f}'
                        #imc = plot_one_box(xyxy, imc, label=label, color=colors(c, True), line_width=line_thickness)
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference + NMS)
            t4 = time_sync()
            print('{0}.)\t\t Prep:{1:3.1f}ms,\t  Infr:{2:3.1f}ms,\t  Post:{3:3.1f}ms, \t\t {4} detections in {5:3.1f}ms'.format(frame, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000, len(det), (t4-t1)*1000))
            # Stream results
            imc = annotator.result()
            if view_img:
                cv2.imshow(str(p), imc)
                cv2.waitKey(1)  # 1 millisecond
            else:
                det_rows = format_detection_result(source, frame, imc, det, class_names)
                if det_rows is not None:
                    writer.writerows(det_rows)

    if not view_img:
        f.close() # close the file
    return pred

def format_detection_result(path, frame_id, img, det, names):
    rows_data = None
    # Process detections per image
    if det is not None and len(det):
        # process results
        idx = 0
        rows_data = []
        for xmin, ymin, xmax, ymax, conf, cls in reversed(det):
            idx += 1
            c = int(cls)  # integer class
            label = f'{names[c]}'
            conf_str = f'{conf:.2f}'
            print("\t[{}.{}]\t\t ({} {} {} {}) \t\t {} {}".format(frame_id, idx, int(xmin), int(ymin), int(xmax), int(ymax), label, conf_str))
            row_item = [frame_id, idx, int(xmin), int(ymin), int(xmax), int(ymax), label, conf_str]
            rows_data.append(row_item)
    return rows_data

if __name__ == "__main__":
    pass