import cv2
import torch

from utils.parser import get_config
from models.deep_sort import DeepSort


class DEEPSORT:
    def __init__(self, deepsort_config):
        cfg = get_config()
        cfg.merge_from_file(deepsort_config)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

        print('DeepSort model loaded!')

    def detection_to_deepsort(self, objects, img):
        xywh_bboxs = []
        confs = []

        # Adapt detections to deep sort input format
        for obj in objects:
            if obj['label'] == 'car':
                xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                conf = obj['score']
                # to deep sort format
                x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf])

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # pass detections to deepsort
        outputs = self.deepsort.update(xywhs, confss, img)

        # draw boxes for visualization
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            draw_boxes(img, bbox_xyxy, identities)

    def xyxy_to_xywh(self, *xyxy):
        """ Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h


color_for_labels = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in color_for_labels]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
