# Missing_Tooth_Detection
*Deep_Learning_based_Missing_Teeth_Detection_for_Dental_Implant_Planning_in_Panoramic_Radiographic_Images* 

논문 재구현

## Code
- vis_utils.py : pred_seg / pred_det를 할 때 라벨링 이름 조정을 위한 코드

### Tooth Instance Segmentation
- train_seg2.py : train code
- eval_seg.py : evaluation code
- pred_seg.py : prediction code

### Missing Tooth Regions Detection
- train_det.py : train code
- eval_det.py : evaluation code
- eval_det_500.py : evaluation code 500번마다 저장된 가중치를 전부 평가하여 best weight를 뽑아내준다.
- pred_det.py : prediction code

## 과제설명
### 과제목적 : 
panoramic radiographic image 만으로 임플란트 식립 위치를 진단하는 것


### 과정 : 

<img src="https://user-images.githubusercontent.com/73769046/154214843-66ec88be-e563-40cf-ab4f-d9ccf0da53fa.png" width="600" height="400">

1. Segmentation model 이 파노라마 이미지에서 치아를 segment 한 뒤 치아의 마스크를 생성
2. Detection model 이 치아의 mask 를 인풋으로 하여 치아가 없는 지역을 예측


### 데이터셋 : 
1. Tooth Instance Segmentation : 사랑니를 제외한 28 개의 치아를 라벨링 한 데이터 셋을 사용
2. Missing Tooth Regions Detection : Tooth Instance Segmentation을 통해 만든 teeth mask와 치아 28개가 다 있는 mask로부터 임의로 치아
를 제거한 합성 데이터 사용


### 사용 모델 : 
1. Tooth Instance Segmentation: Mask R-CNN
2. Missing Tooth Regions Detection: Faster R-CNN

## 결과
### Tooth Instance Segmentation

<img src="https://github.com/star77sa/Missing_Tooth_Detection/blob/main/Result_img/SEG1.jpg" width="600" height="300">

|  |Batch Size|Learning Rate|Iterator|AP[0.5]|AP[0.5 : 0.95]|
|------------|---|---|---|---|---|
|논문|4|0.01|70000|91.14%|76.78%|
|설정1|4|0.01|70000|91.72%|76.88%|
|설정2|4|**0.0015**|10000|92.43%|78.41%|


### Missing Tooth Regions Detection

<img src="https://github.com/star77sa/Missing_Tooth_Detection/blob/main/Result_img/DETBEST.jpg" width="600" height="300">

|  |Batch Size|Learning Rate|Iterator|AP[0.5]|AP[0.5 : 0.95]|
|------------|---|---|---|---|---|
|논문|32|0.01|100000|59.09%|20.40%|
|설정1 Faster R-CNN|32|0.01|100000|56.23%|20.34%|
|설정2 Retina Net|16|0.01|100000|56.47%|20.05%|

- Tooth Instance Segmentation의 경우는 하이퍼 파라미터 튜닝을 통해 더 좋은 성능을 내었다.
- Missing Tooth Regions Detection의 경우는 논문과 근사하지만 59%의 성능을 내지는 못하였다.
