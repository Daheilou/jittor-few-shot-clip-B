import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR
from jclip.convnext import convnext_base,convnext_large
from jclip.convnextv2 import convnextv2_base,convnextv2_large,convnextv2_tiny
from jittor.optim import SGD
from scipy.ndimage.filters import gaussian_filter
import random
from random import choice, shuffle

jt.flags.use_cuda = 1


def get_train_transforms():
    return transform.Compose([
        # transform.Lambda(lambda img: data_augment(img)),
        transform.Resize((320, 320)),
        transform.RandomCrop((280,280)),
        
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_valid_transforms():
    return transform.Compose([
        transform.Resize(384),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


class CUB200(Dataset):
    def __init__(self, img_path, img_label, batch_size, part='train', shuffle=False, transform=None):
        super(CUB200, self).__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.img_path),
            shuffle=shuffle
        )


    def __getitem__(self, index):
        img = os.path.join(imgs_dir, self.img_path[index])
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        img = np.asarray(img)
            
        label = self.img_label[index]
        
        return img, label


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []


    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (train_img, labels) in enumerate(pbar):
        

        output = model(train_img)
        loss = criterion(output, labels)
        optimizer.step(loss)


        pred = np.argmax(output.numpy(), axis=1)
        acc = np.sum(pred == labels.numpy())
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])

        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f}'
                             f'acc={total_acc / total_num:.2f}')
    scheduler.step()


def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for images, labels in val_loader:

        output = model(images)
        
        pred = np.argmax(output.numpy(), axis=1)

        acc = np.sum(pred == labels.numpy())

        total_acc += acc
        total_num += labels.shape[0]
        
        

        pbar.set_description(f'Epoch {epoch}' f'acc={total_acc / total_num:.4f}')

    acc = total_acc / total_num
    return acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=31)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data', type=str, default='train_data/train_711.txt')
    parser.add_argument('--val_data', type=str, default='train_data/val_711.txt')
    parser.add_argument('--pretrain_convnext_model', type=str, default='pretrain/convnextv2_base_1k_224_ema.pkl')
    parser.add_argument('--save_model', type=str, default='./out/convnextv2-base.pkl')
    parser.add_argument('--imgs_dir', type=str, default='./Dataset/')
    args = parser.parse_args()
    
    jt.set_global_seed(args.seed)
    
    imgs_dir = args.imgs_dir
    
    options = {
        'num_classes': 130,
        'threshold': 0.74,
        'lr': args.lr,
        'eta_min': args.eta_min,
        'T_max': args.T_max,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'accum_iter': args.accum_iter,
    }
    
    train_data = open(args.train_data).read().splitlines()
    val_data = open(args.val_data).read().splitlines()

    train_imgs,train_labels=[],[]
    val_imgs,val_labels=[],[]
    num = 0
    for l in train_data:
        a = int(l.split(',')[1])
        b = l.split(',')[0]
        if a >= 244:
            train_imgs.append(b)
            train_labels.append(a-244)


    for l in val_data:
        num = num+1
        a = int(l.split(',')[1])
        b = l.split(',')[0]
        if a >= 244 and num % 2 == 0:
            val_imgs.append(b)
            val_labels.append(a-244)
        
    train_loader = CUB200(train_imgs, train_labels, 32, 'train', shuffle=True,
                          transform=get_train_transforms())
    val_loader = CUB200(val_imgs, val_labels, 32, 'valid', shuffle=True,
                        transform=get_valid_transforms())


    
    # model = convnext_base(num_classes=130)
    # state_dict = jt.load('pretrain/convnext_base_1k_224_ema.pkl')
    model = convnextv2_base(num_classes=130)
    state_dict = jt.load(args.pretrain_convnext_model)


    model.load_parameters(state_dict)

    criterion = nn.CrossEntropyLoss()

    
    optimizer = nn.AdamW(model.head.parameters(), options['lr'])
    scheduler = CosineAnnealingLR(optimizer, options['T_max'], options['eta_min'])

    best_acc = options['threshold']
    for epoch in range(options['epochs']):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, options['accum_iter'], scheduler)
        if epoch >= 30:
            acc = valid_one_epoch(model, val_loader, epoch)
            print(acc)
            best_acc = acc
            model.save(args.save_model)
