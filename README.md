# Cost-Sensitive Learning for Object Detection
### 23-1 자기주도연구2 사이버보안학과 박지희
***
This project is for solving the negative effects of imbalanced in object detection.
- method: cost-sensitive learning using effective number of class balanced loss
- model: centerNet
- data: PASCAL VOC 2012


# Requirements
```torch >= 1.2.0  
torchvision>=0.4.1  
timm >= 0.1.14
Pillow >= 6.2.2
opencv-python >= 4.2.0
albumentations >= 0.4.5
```

# How to train
Modify ```config/{}.py``` according to your needs, then```python train_voc.py```.

# How to test
Modify ```config/{}.py``` according to your needs, then ```python test_voc.py```.


