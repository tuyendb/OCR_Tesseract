import pytesseract
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from path.path import *


def models():
    prediction_score_threshold = 0.7
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.MODEL.WEIGHTS = detectron2_model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.DEVICE = 'cpu'
    #Detectron predictor
    text_block_detector = DefaultPredictor(cfg)
    text_block_recognizer = pytesseract.image_to_string
    return text_block_detector, text_block_recognizer
    