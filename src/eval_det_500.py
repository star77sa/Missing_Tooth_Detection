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
import argparse


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format='P')
    transform_list = [
        T.Resize((300,600))
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    # dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    dataset_dict["image"] = torch.as_tensor(image)

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
    # json_file_test = "/SSD4/kyeongsoo/implant/Empty_Detection/val/json/missing_teeth_inst.json"
    # image_root_test =  "/SSD4/kyeongsoo/implant/Empty_Detection/val/img"
    
    #test
    json_file_test = "/SSD4/kyeongsoo/implant/Empty_Detection/test/json/missing_teeth_inst.json"
    image_root_test =  "/SSD4/kyeongsoo/implant/Empty_Detection/test/img"
    


    register_coco_instances("test",{},json_file_test,image_root_test) # valid 데이터셋 등록 이름만 test

    cfg = get_cfg()
    
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    
    
    
    epoch_list = np.arange(499,72499,500) ## 가운데 나중에 100000으로 조정
    weight_list = []
    for item in epoch_list:
        weight_name = 'model_{0:07d}.pth'.format(item)
        weight_list.append(weight_name)
    # if os.path.exists(os.path.join(args.target_folder, 'model_final.pth')):
        # wieght_list.append('model_final.pth')
    best_AP = 0

    temp_dir = '/SSD4/kyeongsoo/implant_code/output/det_faster/faster'

    for item in weight_list:
        cfg.MODEL.WEIGHTS = os.path.join(temp_dir, item) # OUTPUT_DIR 수정? or 새로운 변수 넣기
        predictor = CustomPredictor(cfg)
        evaluator = COCOEvaluator("test",{"bbox"},False, output_dir = cfg.OUTPUT_DIR)
        test_loader = predictor.build_eval_loader(cfg)
        results = detectron2.evaluation.inference_on_dataset(predictor.model, test_loader, evaluator) # 이 부분이 결과출력 부분인듯
        AP_seg = results['bbox']['AP']
        if AP_seg > best_AP:
            result_best = results
            model_best = cfg.MODEL.WEIGHTS
            best_AP = AP_seg

    print("bestmodel :" + model_best)
    print("bestAP :" + best_AP)

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, '/SSD4/kyeongsoo/implant_code/output/retina_det_100000.pth') # item에 ~~~.pth 넣기
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, '/SSD4/kyeongsoo/implant_code/output/det_100000.pth') # item에 ~~~.pth 넣기
    
    # evaluator = COCOEvaluator("test",{"bbox","segm"},False, output_dir = cfg.OUTPUT_DIR)
    # evaluator = COCOEvaluator("test",{"bbox"},False, output_dir = cfg.OUTPUT_DIR)
    # evaluator = PascalVOCDetectionEvaluator("test")
    
    # test_loader = predictor.build_eval_loader(cfg)
    # 
    # AP_seg = results['segm']['AP']
    print(results)

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Detection')
    
    parser.add_argument('--output_dir', type=str, default='output/m_det/_/')
    args = parser.parse_args()
    main()