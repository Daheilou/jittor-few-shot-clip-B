import random
import pandas as pd

train_data = open('Dataset/train.txt').read().splitlines()

train_imgs,train_labels=[],[]
val_imgs,val_labels=[],[]
num = 0

cnt = {}
new_train_imgs = []
new_train_labels = []


random.shuffle(train_data)
for l in train_data:
    num = num+1
    a = l.split(' ')[0]
    b = int(l.split(' ')[1])

    if b not in cnt:
        cnt[b] = 0
    if cnt[b] < 4:
        new_train_imgs.append(a)
        new_train_labels.append(b)
        cnt[b] += 1
    else:
        val_imgs.append(a)
        val_labels.append(b)  
    
print(len(new_train_imgs))

        
with open('./data/train.txt', 'w') as save_file:
    for k,z in zip(new_train_imgs,new_train_labels):
        save_file.write(k + ',' + str(z) + '\n')

with open('./data/val.txt', 'w') as save_file:
    for k,z in zip(val_imgs,val_labels):
        save_file.write(k + ',' + str(z) + '\n')
print('训练集统计标签频率验证:')        
train = pd.read_csv('data/train.txt',names=['name','label'])
print(train['label'].value_counts())

        
