# Missing_Tooth_Detection
*Deep_Learning_based_Missing_Teeth_Detection_for_Dental_Implant_Planning_in_Panoramic_Radiographic_Images* 논문 재구현

## Code

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
- 과제목적 : panoramic radiographic image 만으로 임플란트 식립 위치를 진단하는 것
- 과정 : 
 ![image](https://user-images.githubusercontent.com/73769046/154214843-66ec88be-e563-40cf-ab4f-d9ccf0da53fa.png)
  1. Segmentation model 이 파노라마 이미지에서 치아를 segment 한 뒤 치아의 마스크를 생성
  2. Detection model 이 치아의 mask 를 인풋으로 하여 치아가 없는 지역을 예측
