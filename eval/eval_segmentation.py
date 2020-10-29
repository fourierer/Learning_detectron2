import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

assert torch.__version__.startswith("1.6") # 必须是1.6版本的

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


im = cv2.imread('../monkey_analyse/jpg/image_00381.jpg')
#im = cv2.imread('./input.jpg')
assert im is not None
# Inference with a panoptic segmentation model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
#cv2_imshow(out.get_image()[:, :, ::-1])

output_img = out.get_image()[:, :, ::-1]
#print(type(output))

cv2.imwrite('../monkey_analyse/jpg_output/output_00381_segmentation.jpg', output_img)
#cv2.imwrite('./output_segmentation.jpg', output_img)

