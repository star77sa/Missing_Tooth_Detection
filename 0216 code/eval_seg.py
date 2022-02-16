import os

from pkg_resources import evaluate_marker
import torch, torchvision 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import json,os,cv2,random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, PascalVOCDetectionEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import argparse
import shutil
import copy

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format='BGR')
    transform_list = [
        T.Resize((1000,2000))
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class CustomPredictor(DefaultPredictor):
    @classmethod
    def build_eval_loader(cls, cfg):
        return build_detection_test_loader(cfg, 'test' , custom_mapper)

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = 2
    
    #val
    #json_file_val = "/SSD4/kyeongsoo/implant/Instance_Segmentation/val/coco_inst_teeth_val.json"
    #image_root_val =  "/SSD4/kyeongsoo/implant/Instance_Segmentation/val/"
    
    #test
    json_file_test = "/SSD4/kyeongsoo/implant/Instance_Segmentation/test/coco_inst_teeth_test.json"
    image_root_test =  "/SSD4/kyeongsoo/implant/Instance_Segmentation/test/"
    


    register_coco_instances("test",{},json_file_test,image_root_test) # valid 데이터셋 등록

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, '/SSD4/kyeongsoo/implant_code/output/seg_35000.pth') # item에 ~~~.pth 넣기
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, '/SSD4/kyeongsoo/implant_code/output/예전seg가중치/lr_iter_10000BEST.pth') # item에 ~~~.pth 넣기
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, '/SSD4/kyeongsoo/implant_code/output/final_seg_10000.pth') # item에 ~~~.pth 넣기

    predictor = CustomPredictor(cfg)
    evaluator = COCOEvaluator("test",{"bbox","segm"},False, output_dir = cfg.OUTPUT_DIR)
    # evaluator = PascalVOCDetectionEvaluator("test")
    
    test_loader = predictor.build_eval_loader(cfg)
    results = detectron2.evaluation.inference_on_dataset(predictor.model, test_loader, evaluator) # 이 부분이 결과출력 부분인듯
    AP_seg = results['segm']['AP']
    print(results)

    print('done')



main()