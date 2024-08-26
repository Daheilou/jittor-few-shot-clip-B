import torch
import jittor as jt

clip = torch.load('../pretrain/RN101.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'pretrain/RN101.pkl')

clip = torch.load('../pretrain/ViT-B-32.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'pretrain/ViT-B-32.pkl')

clip = torch.load('../pretrain/convnextv2_base_1k_224_ema.pt')
clip_1 = dict()


for k in clip['model'].keys():
    clip_1[k] = clip['model'][k].float().cpu()
jt.save(clip_1, 'pretrain/convnextv2_base_1k_224_ema.pkl')

