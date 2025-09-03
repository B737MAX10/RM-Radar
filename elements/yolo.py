import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode


class YOLO:
    def __init__(self, device='cpu', model_path='best.pt', classes={}, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, max_det=1000, augment=False):
        print(f"YOLO weights path: {model_path}")

        # Load model
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model = DetectMultiBackend(model_path, device=self.device, fp16=self.half)
        print("YOLO model loaded!")

        self.stride, self.pt = self.model.stride, self.model.pt

        self.names = classes if classes else self.model.names

        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.augment = augment

    @smart_inference_mode()
    def detect(self, img):
        # with torch.no_grad():
        # Run inference
        img0 = img.copy()
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img, augment=self.augment)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)

        # Process predictions
        results = []
        for det in pred:  # per image
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = self.names[int(cls)]
                    score = np.round(conf.item(), 2)
                    xmin = int(xyxy[0].item())
                    ymin = int(xyxy[1].item())
                    xmax = int(xyxy[2].item())
                    ymax = int(xyxy[3].item())

                    result = {'label': label,
                              'score': score,
                              'bbox': [(xmin, ymin), (xmax, ymax)]}

                    results.append(result)

        return results
