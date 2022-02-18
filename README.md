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

|Model|AP[0.5]|AP[0.5:0.95]|
|------|---|---|
|Mask R-CNN|92.43%|78.41%|



### Missing Tooth Regions Detection

<img src="https://github.com/star77sa/Missing_Tooth_Detection/blob/main/Result_img/DETBEST.jpg" width="600" height="300">

|Model|AP[0.5]|AP[0.5:0.95]|
|------|---|---|
|Faster R-CNN|test%|test%|
