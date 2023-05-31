from dataset.voc_test import VOCTestDataset
from config.voc import Config
from torch.utils.data import DataLoader
from progress.bar import Bar
from model.centernet import CenterNet
import torch
from torchvision import transforms
import numpy as np
import os
import xml.etree.ElementTree as ET
import json
import torch.utils.data as data
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def preprocess_img(img, input_ksize):
    min_side, max_side = input_ksize
    h, w = img.height, img.width
    _pad = 32
    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    img_resized = np.array(img.resize((nw, nh)))

    pad_w = _pad - nw % _pad
    pad_h = _pad - nh % _pad

    img_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    img_paded[:nh, :nw, :] = img_resized

    return img_paded, {'raw_height': h, 'raw_width': w}

def tensor_to_image(tensor):
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    return image

def show_img(img, boxes, clses, scores):
    #boxes, scores = [i.cpu() for i in [boxes, scores]]

    #boxes = boxes.long()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(xy=box.tolist(), outline='red', width=3)

    boxes = boxes.tolist()
    scores = scores.tolist()
    plt.figure(figsize=(10, 10))
    for i in range(len(boxes)):
        plt.text(x=boxes[i][0], y=boxes[i][1], s='{}: {:.4f}'.format(clses[i], scores[i]), wrap=True, size=15,
                 bbox=dict(facecolor="r", alpha=0.7))
    plt.imshow(img)
    plt.show()

def test(cfg):
    
    result = []
    class_mapping = {
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9, 
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor':20
    }
    test_ds = VOCTestDataset(cfg.root, mode=cfg.split, resize_size=cfg.resize_size)

    torch.manual_seed(1234)

    # torch.util.data.Dataset
    # 클래스를 사용하여 구성한 데이터셋의 경우, torch.utils.data.random_split()
    # 함수를 사용하여 데이터셋을 분할하고, 일부분만 사용할 수 있음
    dataset_size = len(test_ds)
    test_size = int(dataset_size * 0.75)
    #test_size = int(dataset_size * 0.95)
    #_, subset = data.random_split(test_ds, [train_size, dataset_size - train_size], random_state=torch.Generator().manual_seed(1234))
    _, subset = data.random_split(test_ds, [test_size, dataset_size - test_size])
    

    print("len of test data size: ", len(subset))
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=test_ds.collate_fn, pin_memory=True)

    model = CenterNet(cfg).cuda()
    model.load_state_dict(ckp['model'])
    model = model.eval()
    for step, (gt) in enumerate(test_dl):
       print(step)
       #print(gt)
       imgs, gt_boxes, gt_classes, gt_hm, infos, sample, image_id = gt
       #print(gt_boxes, gt_classes)
       for batch in range(imgs.size(0)):
           img = tensor_to_image(imgs[batch])
           img_paded, info = preprocess_img(img, cfg.resize_size)
           infos = [info]
           input = transforms.ToTensor()(img_paded)
           input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
           inputs = input.unsqueeze(0).cuda()
           info = [{'raw_height': img.height, 'raw_width': img.width}]
           #detects = model.inference(inputs, infos, topK=40, return_hm=False, th=0.25)
           input = transforms.ToTensor()(img)
           input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
           inputs = input.unsqueeze(0).cuda()
           detects = model.inference(inputs, infos, topK=40, return_hm=False, th=0.25)
           #print(detects)
           #boxes = detects[0].item()
           #scores = detects[1].item()
           #classes = detects[2].item()
           
           detects = detects[0]
           '''
           print("===detects===")
           print(type(detects))
           print(detects)
           '''
           #boxes = detects[0].detach().cpu().numpy()
           #scores = detects[1].detach().cpu().numpy()
           #classes = detects[2].detach().cpu().numpy()
           boxes = detects[0].cpu().detach().numpy()
           scores = detects[1].cpu().detach().numpy()
           classes = detects[2]
           '''
           print('boxes:', boxes)
           print('boxes type:', type(boxes))
           print('len:', len(boxes))
           print('scores:', scores)
           print('scores type:', type(scores))
           print('classes', classes)
           '''
           '''
           boxes = detects[0]
           scores = detects[1]
           classes = detects[2]
           print('boxes:', boxes)
           print('scores:', scores)
           print('classes', classes)
           '''
           #if len(boxes) > 0:
           # show_img(img, boxes, classes, scores)
           for i in range(len(boxes)):
               x = float(boxes[i][0])
               y = float(boxes[i][1])
               width = float(boxes[i][2]) - x
               height = float(boxes[i][3]) - y
               res = {
                   'image_id' : image_id,
                   'category_id': class_mapping[classes[i]],
                   'bbox': [x, y, width, height],
                   'score': float(scores[i])
               }
               print(sample,res)
               result.append(res)
            

    return result


def make_cocoformat(result):
    ann = []
    # COCO 형식으로 변환된 어노테이션 저장 경로
    coco_annotation_file = '/content/drive/MyDrive/colab_data/dataset/voc/VOCdevkit/VOC2012/CocoAnnotations/res_v3_annotations.json'
    for res in result:
      print(res)
      print(type(res))
      info = {
          'image_id': int(res['image_id'][0]),
          'category_id': int(res['category_id']),
          'bbox': res['bbox'],
          'score': float(res['score'])
      }
      ann.append(info)

    # COCO 형식으로 변환된 어노테이션 저장
    with open(coco_annotation_file, 'w') as f:
        json.dump(ann, f)
        



if __name__ == '__main__':
    ckp = torch.load('./ckp/my_best_checkpoint3.pth')
    cfg = ckp['config']
    print(cfg)
    # cfg = Config
    cfg.split = 'val'
    cfg.resume = True
    cfg.root = Config.root
    result = test(cfg)
    print(result)
    make_cocoformat(result)
    print('===complete===')