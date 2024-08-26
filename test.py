import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
import numpy as np
from jittor import transform
from sklearn.metrics import accuracy_score
import random
from sklearn import svm
import jittor.nn as nn
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize,RandomHorizontalFlip
from jclip.clip import _convert_image_to_rgb,ImageToTensor,Resize
from jittor.dataset import Dataset
from scipy.linalg import eigh
from jclip.convnextv2 import convnextv2_base
from jclip.convnext import convnext_base
import pickle


jt.flags.use_cuda = 1

def image_opt(feat, init_classifier, plabel, lr=10, iter=200, tau_i=0.04, alpha=0.6):
    ins, dim = feat.shape
    idx,val = jt.argmax(plabel, dim=1)

    mask = val > alpha
    plabel[mask, :] = 0
    plabel[mask, idx[mask]] = 1
    
    base = feat.transpose(0, 1) @ plabel
    classifier = init_classifier
    pre_norm = -100
    for i in range(0, iter):
        prob = jt.nn.softmax(feat @ classifier / tau_i, dim=1)
        grad = feat.transpose(0, 1) @ prob - base
        # print(grad.shape)
        
        grad_1 = jt.view(grad,(1,512*403))
#         print(grad.shape)
        temp = jt.norm(grad_1)
        temp = temp.item()

        if temp > pre_norm:
            lr /= 2.
        pre_norm = temp
        classifier -= (lr / (ins * tau_i)) * grad

    return classifier


def sinkhorn(M, tau_t=0.01, gamma=0, iter=20):
    row, col = M.shape
    P = jt.nn.softmax(M / tau_t, dim=1)
    P /= row
    if gamma > 0:
        q = jt.sum(P, dim=0, keepdim=True)
        q = q**gamma
        q /= jt.sum(q)
    for it in range(0, iter):
        # total weight per column must be 1/col or q_j
        P /= jt.sum(P, dim=0, keepdim=True)
        if gamma > 0:
            P *= q
        else:
            P /= col
        # total weight per row must be 1/row
        P /= jt.sum(P, dim=1, keepdim=True)
        P /= row
    P *= row  # keep each row sum to 1 as the pseudo label
    return P

def get_train_transforms(n_px):
    return Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        RandomHorizontalFlip(0.5),
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

def get_valid_transforms(n_px):
    return Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(0.5),
        _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

def get_test_transforms(n_px):
    return Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(0.5),
        _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])


def zero_classifier(classnames, templates, clip_model):
    with jt.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = jt.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def clip_classifier(classnames, template, clip_model):
    with jt.no_grad():
        clip_weights = []
        num = 0
        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = template[num].replace('_', ' ')
            texts = clip.tokenize(texts)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
            num += 1

        clip_weights = jt.stack(clip_weights, dim=1)
    return clip_weights




def pre_load_features(clip_model, loader):
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = jt.cat(features), jt.cat(labels)
    
    return features, labels

class CUB200_test(Dataset):
    def __init__(self, img_path, img_label, batch_size, part='train', shuffle=False, transform=None):
        super(CUB200_test, self).__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.img_path),
            shuffle=shuffle
        )


    def __getitem__(self, index):
        img = os.path.join(test_imgs_dir, self.img_path[index])
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        img = np.asarray(img)
            
        label = self.img_label[index]
        
        return img, self.img_path[index]




def LP(clip_model,new_classes, template, test_features, alpha_vec,LP):
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(
        new_classes, template, clip_model)

    classifier = nn.Linear(test_features.shape[1],len(new_classes))


    
    with open(alpha_vec, 'rb') as file:
        alpha_vec = pickle.load(file)
        
    state_dict = jt.load(LP)
    classifier.load_parameters(state_dict)    

    vision_logits_test = classifier(test_features)
    text_logits_test = test_features.detach() @ clip_weights
    logits_test = vision_logits_test + jt.ones(test_features.shape[0], 1) @ alpha_vec * text_logits_test
    logits_test = logits_test.numpy()
    
    return logits_test





class Dogtest(Dataset):
    def __init__(self, img_path, batch_size, part='test', shuffle=False, transform=None):
        super(Dogtest, self).__init__()
        self.img_path = img_path
        self.transform = transform
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.img_path),
            shuffle=shuffle
        )


    def __getitem__(self, index):
        img = os.path.join(test_imgs_dir, self.img_path[index])
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        img = np.asarray(img)
        
        return img
    
def get_dog_test_transforms():
    return transform.Compose([
        transform.Resize(384),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def Inmap(image_feat,text_classifier):
    
    logits_t = image_feat @ text_classifier.softmax(dim=-1)
    
    tau_t = 0.01
    tau_i = 1
    lr = 10
    iters_proxy = 20
    alpha = 1
    gamma = 0.
    iters_sinkhorn = 100

    print('obtain vision proxy without Sinkhorn distance')
    plabel = jt.nn.softmax(logits_t / tau_t, dim=1)
    image_classifier = image_opt(image_feat, text_classifier, plabel, lr, iters_proxy, tau_i, alpha)
    logits_i = image_feat @ image_classifier
    
    plabel = sinkhorn(logits_i, tau_t, gamma, iters_sinkhorn)

    image_classifier = image_opt(image_feat, text_classifier, plabel, lr, iters_proxy, tau_i, alpha)
    unseen_test = image_feat @ image_classifier
    
    return unseen_test
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_img', type=str, default='TestSetB/')
    parser.add_argument('--all_class', type=str, default='Dataset/classes_b.txt')
    parser.add_argument('--pretrain_vit_model', type=str, default='pretrain/ViT-B-32.pkl')
    parser.add_argument('--pretrain_rn_model', type=str, default='pretrain/RN101.pkl')
    parser.add_argument('--pretrain_convnext_model', type=str, default='out/convnextv2-base.pkl')
    parser.add_argument('--vit_alpha_vec', type=str, default='out/vit_alpha_vec_526.pkl')
    parser.add_argument('--vit_LP', type=str, default='out/LP-526-0.748.pkl')
    parser.add_argument('--rn_alpha_vec', type=str, default='out/rn_alpha_vec_711.pkl')
    parser.add_argument('--rn_LP', type=str, default='out/LP-711-RN-0.726.pkl')  
    parser.add_argument('--output', type=str, default='result.txt')
    

    args = parser.parse_args()
    
    jt.set_global_seed(args.seed)
    
    vit_clip_model, vit_preprocess = clip.load(args.pretrain_vit_model)

    classes = open(args.all_class).read().splitlines()
    
    seen_template,all_template = [],[]
    new_seen_classes,new_all_classes = [],[]
    
    describe = dict()
    describe['Audi_TTS_Coupe_2012'] = 'Sporty_Elegant'
    describe['Audi_R8_Coupe_2012'] = 'Exotic_Premium'    

    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
            a = 'a photo of a ' + c
            new_seen_classes.append(c)
            seen_template.append(a)
            new_all_classes.append(c)
            all_template.append(a)
        if c.startswith('Thu-dog'):
            c = c[8:]
            a = 'a photo of a ' + c + ', a type of dog' 
            new_seen_classes.append(c)
            seen_template.append(a)
            new_all_classes.append(c)
            all_template.append(a)

        if c.startswith('Caltech-101'):
            c = c[12:]
            a = 'a photo of a ' + c
            new_seen_classes.append(c)
            seen_template.append(a)
            new_all_classes.append(c)
            all_template.append(a)
            
        if c.startswith('Food-101'):
            c = c[9:]
            a = 'a photo of a ' + c + ', a type of food'
            new_seen_classes.append(c)
            seen_template.append(a)
            new_all_classes.append(c)
            all_template.append(a)
            
        if c.startswith('Stanford-Cars'):
            c = c[14:]
            if c in describe.keys():
                c = c + '_' + describe[c]
            a = 'a photo of a ' + c
            new_all_classes.append(c)
            all_template.append(a)
            
    print(new_all_classes)

    new_seen_classes = list(new_seen_classes)
    new_all_classes = list(new_all_classes)

    print('seen clasees length:',len(new_seen_classes))
    print('all clasees length:',len(new_all_classes))
    

    text_all = clip.tokenize(new_all_classes)
    text_all_features = vit_clip_model.encode_text(text_all)
    text_all_features /= text_all_features.norm(dim=-1, keepdim=True)
    
    
    test_imgs_dir = args.test_img
    test_imgs = os.listdir(test_imgs_dir)
    label = len(test_imgs) * [0]
    
    test_loader = CUB200_test(test_imgs, label, 1, 'test', shuffle=False,
                          transform=get_test_transforms(224))
    

    
    print("\n extracting visual vit-b-32 features and classes unseen predict from teat B set.")

    vit_test_features, test_img, text_probs, unseen = [], [], [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(test_loader)):
            image_features = vit_clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)         
            vit_test_features.append(image_features)
            test_img.append(target)
        
    vit_test_features = jt.cat(vit_test_features)
    test_img = np.concatenate(test_img)

    text_classifier = text_all_features.transpose(0, 1).float()
    vit_unseen = Inmap(vit_test_features,text_classifier)
    
    vit_logits_test = LP(vit_clip_model,new_seen_classes, seen_template, vit_test_features, args.vit_alpha_vec,args.vit_LP)
    
    del vit_clip_model
    
    rn_clip_model, rn_preprocess = clip.load(args.pretrain_rn_model)
    
    
    text_rn_all_features = rn_clip_model.encode_text(text_all)
    text_rn_all_features /= text_rn_all_features.norm(dim=-1, keepdim=True)
    

    print("\n extracting visual RN101 features from teat B set.")
    rn_test_features = []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(test_loader)):
            image_features = rn_clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            rn_test_features.append(image_features)
        
    rn_test_features = jt.cat(rn_test_features)
    
    text_rn_classifier = text_rn_all_features.transpose(0, 1).float()
    rn_unseen = Inmap(rn_test_features,text_rn_classifier)

    num = 0
        
    rn_logits_test = LP(rn_clip_model,new_seen_classes,seen_template,rn_test_features,args.rn_alpha_vec,args.rn_LP)

    
    model = convnextv2_base(num_classes=130)
    state_dict = jt.load(args.pretrain_convnext_model)
    model.load_parameters(state_dict)
    

    
    
    with open(args.output, 'w') as save_file:
        for i in test_img:
            
            ## 第一步zero clip预测未知类别,如果出现直接预测结束，没有出现进行下一步预测
            text_probs = vit_unseen[num] * 0.5 + rn_unseen[num] * 0.5
            
            _, top_labels = text_probs.topk(5)
            top_labels = top_labels.numpy()
            
            
            
            if top_labels[0] >= 374:
                save_file.write(i + ' ' +
                                ' '.join([str(p.item()) for p in top_labels]) + '\n')
            else:
                
                ##如果没有未知类别，LP融合模型预测是否出现狗的类别，出现直接预测结束，没有转下一步
                
                logits_test = vit_logits_test * 0.6 + rn_logits_test * 0.4
                top5_idx = logits_test[num].argsort()[-1:-6:-1]
                top5_idx = list(top5_idx)
                test_dir = []

                if int(top5_idx[0]) >= 244:
                    ## 第三步The dog 分类识别
                    test_dir.append(i)
                    test_loader = Dogtest(test_dir, 1, 'test', shuffle=False,transform=get_dog_test_transforms())
                    for images in test_loader:
                        prediction = model(images)[0].numpy()
                        top5_idx = prediction.argsort()[-1:-6:-1]
                    save_file.write(i + ' ' +
                                    ' '.join(str(idx+244) for idx in top5_idx) + '\n')
                else:
                    save_file.write(i + ' ' +
                                    ' '.join(str(idx) for idx in top5_idx) + '\n')

            num+=1



    
