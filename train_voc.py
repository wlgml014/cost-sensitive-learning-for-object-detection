from trainer.trainer import Trainer
from dataset.voc import VOCDataset
from model.centernet import CenterNet
from config.voc import Config
from loss.loss import Loss
from torch.utils.data import DataLoader
import torch.utils.data as data

def train(cfg):
    train_ds = VOCDataset(cfg.root, mode=cfg.split, resize_size=cfg.resize_size)
    
    dataset_size = len(train_ds)
    train_size = int(dataset_size * 0.75)
    #train_size = int(dataset_size * 0.5)
    _, subset = data.random_split(train_ds, [train_size, dataset_size - train_size])
    
    print('len of subset: ', len(subset))
    print('len of train size: ', len(train_ds))
    train_dl = DataLoader(subset, batch_size=1, shuffle=True,
                           num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)

    #train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
    #                      num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)
    '''
    print('train_ds value: ',train_ds)
    print('train_ds type: ',type(train_ds))
    print('trian_dl value: ', train_dl)
    print('train_dl type: ', type(train_dl))
    print(train_ds.__getitem__(0))
    print(train_ds.__len__())
    print('subset value: ', subset)
    print('subset type: ',type(subset))
    print(subset.__len__())
    '''
    model = CenterNet(cfg)
    if cfg.gpu:
        model = model.cuda()
    loss_func = Loss(cfg)

    #epoch = 100
    epoch = 100
    cfg.max_iter = len(train_dl) * epoch
    cfg.steps = (int(cfg.max_iter * 0.6), int(cfg.max_iter * 0.8))

    trainer = Trainer(cfg, model, loss_func, train_dl, None)
    trainer.train()


if __name__ == '__main__':
    cfg = Config
    # cfg.resume = False
    # cfg.resume = True
    train(cfg)
