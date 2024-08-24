# jittor-few-shot-clip-B
第四届计图人工智能挑战赛-jittor-[深度玄学]-开放域少样本视觉分类赛题B榜

### 环境安装
 - pip install jittor==1.3.8.5
 - pip install ftfy regex tqdm
 - pip install scipy
 - pip install scikit-learn
 - pip install pandas
 - pip install numpy
 - pip install opencv-python==4.8.1.78
 - pip install Pillow==10.1.0
 - pip install torch==2.0.1+cu118

### 方案阐述
 - 第一步借鉴InMap(Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP )方案预测unseen类别, 有直接预测，预测出可见类别转第二步 [Code link](https://github.com/idstcv/InMaP/) [Paper link](https://arxiv.org/abs/2310.19752)
 - 第二步借鉴LP++（LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP）进行可见类别4-shot的学习,预测出狗的类别转第三步，不是狗的类别直接预测 [Paper line](https://arxiv.org/abs/2404.02285)
 - 第三步根据上一步识别狗类别，重新用4-shot微调后的ConvnextV2模型重新识别

### 预训练模型
 - Oepn CLIP [ViT-B/32]( https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)
 - Oepn CLIP [ResNet101](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt)
 - [ConvNeXt-V2-B](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt) [Github code](https://github.com/facebookresearch/ConvNeXt-V2) [HF镜像](https://hf-mirror.com/facebook/convnextv2-base-1k-224)
 
### InMap
 - 不需要训练，核心代码在test.py243行至265行，实现非常简单, 任何unseen类别都可以用
 - 文献里提到13个数据集rn50平均提升5个点，car类别提升7个点
 - 需要CLIP提取图片和文本特征，无其它模型参数
 - 需要对超参数进行细调
 - 运行时间很快，5s
   
### LP++
 - 对线性层进一步优化，主要在图像特征和文本特征的权重上
 - 文献中显示与Tip-Adapter-F相比，各有千秋，分数接近，4-shot表现中LP++在11个数据集平均分数为69.16±0.79, Tip-Adapter-F为68.71 ± 0.96
 - 会生成中间文件，这个中间文件是一个常量矩阵，vit-b-32及rn101方案中这个文件大小均为1.6K
 - 3090训练时间小于1min
 
 
### 最终参数之和
 - 第一种以加载参数算361M
   - ViT-B-32.pkl (151M) RN101.pkl (120M) + LP-ViT.pkl (0.2M) + LP-RN.pkl  (0.2M) + convnextv2-base.pkl (89M)
 - 第二种从使用次数算489M
   - InMap 用到ViT-B-32和RN101, LP也用到ViT-B-32和RN101，按道理应该乘以(151M+120M) * 2
   - 采用图像特征共享，因为图像特征都一样，建议用encode_image搜索， 代码出现两次
     - test.py代码355行image_features = vit_clip_model.encode_image(images)
     - test.py代码381行image_features = rn_clip_model.encode_image(images)
   - 文本没法共享，InMap用的全部类别，LP++用的374个类别，文本不一样，建议用encode_text搜索， 代码出现4次
   - 相当于在第一种基础上增加了vit-b-32及rn101在encode_text使用的参数辆（63.4M+63.4M）[参考link](https://blog.csdn.net/bblingbbling/article/details/136511701)
 
### 预处理(第一次需要,torch模型转换及4-shot训练集)
 - python conver.py
 - python process.py
   - 可以对训练集统计标签频率验证, process 42～43行代码如下
   - train = pd.read_csv('data/train.txt',names=['name','label'])
   - print(train['label'].value_counts())
 
### 训练
 - bash train.sh
    - train_data 4-shot训练集路径及标签（提供在data）
    - val_data 验证集路径及标签 (提供在data)
    - seen_class 可见图片类别374类 (
<span style="color: red;">自行补充</span>)
    - img_dir 数据集路径 (<span style="color: red;">自行补充</span>)
    - pretrain_vit_model Open预训练模型 [ViT-B/32]( https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) （<span style="color: red;">自行补充</span>, conver.py文件转换）
    - pretrain_rn_model Open预训练模型 [ResNet101](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt)（<span style="color: red;">自行补充</span>, conver.py文件转换）
    - pretrain_convnext_model ImageNet-1K模型 [ConvNeXt-V2-B](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt) （<span style="color: red;">自行补充</span>, conver.py文件转换）
    - save_lp_vit_model LP++ 通过CLIP Vit-b-32抽取特征微调后模型权重路径 （无需补充，需要路径名称, 建议默认）
    - save_lp_vit_vecer 中间文件 （无需补充，需要路径名称, 建议默认）
    - save_lp_rn_model LP++ 通过clip RN101抽取特征微调后模型权重路径 （无需补充，需要路径名称, 建议默认）
    - save_lp_rn_vecer 中间文件 （无需补充，需要路径名称, 建议默认）
    - save_convnext_model 4-shot dog图像分类微调后的模型 (无需补充，需要路径名称, 建议默认）
    - batch_size=32 （默认,不建议改，影响较大）
  
### 测试
 - bash test.sh
    - test_img 测试图片路径（<span style="color: red;">自行补充</span>）
    - all_class 全部类别文件 (<span style="color: red;">自行补充</span>）
    - pretrain_vit_model Open预训练模型 [ViT-B/32]( https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) （<span style="color: red;">自行补充</span>,conver.py文件转换）
    - pretrain_rn_model Open预训练模型 [ResNet101](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt)（<span style="color: red;">自行补充</span>, conver.py文件转换）
    - pretrain_convnext_model (已提供在out文件)
    - vit_alpha_vec (已提供在out文件)
    - vit_LP (已提供在out文件)
    - rn_alpha_vec (已提供在out文件)
    - rn_LP (已提供在out文件)
    - out 最终文件 (默认，可以修改)
  
### 说明
 - conver.py需要手动转化，注意下载到模型在上一层目录, 转换后模型在当前目录，下载模型路径可以自行修改，在4，10，16行，转话后的目录默认，如有必要改，相应脚本路径也需要保持一致，默认是相对路径
 - 训练显卡>16G,测试显卡>22G(团队训练显卡3090 24G,测试A6000 48G), 3090训练时间小于1h(1min+1min+15min)，A6000预测时间20min
 - 训练或测试脚本运行前注意红色部分，训练5个，测试4个,除红色部分其它可以不动，模型转化成功后（在说明里第一点），即可运行
   
### 收获
 - 使用Jittor实现InMap,Tip Adapter,LP++,Convnext,Convnextv2, RMT, MMLA
