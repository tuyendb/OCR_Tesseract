import json
from operator import ge
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from utils.util import pdf2jpg, expand_box
from config.config import pdf_path


def get_boxes(predictor, img):
    img_height, img_width = img.shape[:2]
    class_labels = ['text', 'title', 'list', 'table', 'figure']
    colors = [(125,125,0), (255,0,255), (255,255,0), (0,255,255), (255,0,0)]
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    img_boxes = []
    for i in range(0, len(pred_boxes)):
        box = pred_boxes[i].tensor.numpy()[0].astype(int)
        score = round(float(scores[i].numpy()), 2)
        label_key = int(pred_classes[i].numpy())
        label = class_labels[label_key]
        x1, y1, x2, y2 = expand_box(box)
        xc = str(round((x1+x2)/2/img_width, 4))
        yc = str(round((y1+y2)/2/img_height, 4))
        w = str(round((x2-x1)/img_width, 4))
        h = str(round((y2-y1)/img_height, 4))
        img_boxes.append([label_key, xc, yc, w, h])
    return img_boxes


def main():
    model_path = '../module/detectron2-publaynet/model/faster_rcnn_R_101_FPN_3x.pth'
    model_zoo_config_name = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    prediction_score_threshold = 0.7
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    cfg.MODEL.NMS_THRESH_SCORE = 0.4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)
    page_imgs = pdf2jpg(pdf_path, None, None)
    for i, img in enumerate(page_imgs):
        img_boxes = get_boxes(predictor, img)
        with open('./dataset/lichsu10/lichsu10_%s.txt' %str(i), 'w', encoding='utf-8') as txtfile:
            for box in img_boxes:
                txtfile.write('{} {} {} {} {}\n'.format(box[0], box[1], box[2], box[3], box[4]))


if __name__ == '__main__':
    main()