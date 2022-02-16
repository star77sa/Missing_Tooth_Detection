# Missing_Tooth_Detection
Missing Tooth Detection 논문 재구현

## Code

Tooth Instance Segmentation
- train_seg2.py : train code / CNN 꿑팁 논문기반 lr 재설정

- eval_seg.py : evaluation code / 평가코드. val set, test set 설정 가능

- pred_seg.py : prediction code / 예측코드. 결과물 산출

Missing Tooth Regions Detection
- train_det.py : train code / 원 논문 그대로
- train_det_retina.py : train code / retina net 사용

- eval_det.py : evaluation code / 평가코드. val set, test set 설정 가능
- eval_det_500.py : evaluation code / 평가코드. 500번마다 저장된 가중치를 전부 평가하여 best weight를 뽑아내준다.

- pred_det.py : prediction code / 결과산출
- pred_det_retina.py : prediction code / retina net 사용
- pred_det_train.py : prediction code / train 데이터에서도 잘 예측하는지 확인을 
