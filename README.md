# ✨MuSc (ICLR 2024)✨

**论文“MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images”的官方复现代码**

作者:  [李煦蕤](https://github.com/xrli-U)<sup>1*</sup> | [黄子鸣](https://github.com/ZimingHuang1)<sup>1*</sup> | [薛峰](https://xuefeng-cvr.github.io/)<sup>3</sup> | [周瑜](https://github.com/zhouyu-hust)<sup>1,2</sup>

单位: <sup>1</sup>华中科技大学 | <sup>2</sup>武汉精测电子集团股份有限公司 | <sup>3</sup>特伦托大学

### 🧐 论文下载地址： [Arxiv](https://arxiv.org/pdf/2401.16753.pdf) | [OpenReview](https://openreview.net/forum?id=AHgc5SMdtd)

## <a href='#all_catelogue'>**转到目录**</a>

<span id='all_catelogue'/>

## 📖目录

* <a href='#abstract'>1. 论文介绍</a>
* <a href='#setup'>2. 代码运行环境配置</a>
* <a href='#datasets'>3. 数据集下载</a>
* <a href='#run_musc'>4. 运行代码</a>
* <a href='#rscin'>5. 单独运行RsCIN分类优化模块</a>
* <a href='#results_datasets'>6. 在不同数据上的结果</a>
* <a href='#results_backbones'>7. 使用不同特征提取器的结果</a>
* <a href='#inference_time'>8. 推理时间</a>
* <a href='#FAQ'>9. 常见问题</a>
* <a href='#citation'>10. 引用格式</a>
* <a href='#thanks'>11. 致谢</a>
* <a href='#license'>12. 使用许可</a>

<span id='abstract'/>

## 👇论文介绍: <a href='#all_catelogue'>[返回目录]</a>

该论文研究了工业视觉领域中的零样本异常检测和分割任务。
零样本，即不使用任何与测试图像同源的有标注图像，以往的方法基于CLIP的图文对齐能力和SAM的提示工程，忽略了无标签测试图像本身蕴含的丰富正常先验信息。
本论文的关键发现在于工业产品图像中，图像的正常区域可以在其他无标注的图像中找到相对大量的相似的正常区域，而异常区域只能找到少量相似的区域。
我们利用这种特性设计了一种新的零样本异常检测/分割方法MuSc，该方法的核心在于对无标注的图像进行相互打分，正常区域会被赋予较低的分数，异常区域会被赋予较高的分数。
该方法不需要任何辅助数据集进行训练，也不需要额外的文本模态进行提示。

具体而言，我们首先使用多聚合度邻域聚合模块(**LNAMD**)来获取能够表征不同大小缺陷的区域级特征。
然后我们提出了互打分模块(**MSM**)，使用无标注图像进行相互打分，分数越高表示该图像区域异常概率越大。
最后，我们提出了一个分类优化模块，名为图像级受限邻域的重打分(**RsCIN**)，来优化分类结果，减少噪声带来的误检。

我们通过在MVTec AD和VisA数据集上的优异性能证明了我们方法的有效性，与当前SOTA零样本异常检测方法相比，MuSc在MVTec AD数据集上实现了**21.1**%的PRO提升(从72.7％到93.8％)，在VisA上实现了**19.4**%的AP分割提升和**14.7**%的AUROC分割提升。
此外，我们的零样本方法甚至优于当前大多数少样本方法，并且与一些无监督方法相媲美。

![pipline](./assets/pipeline.png) 

## 😊与其它零样本异常检测方法比较

![Compare_0](./assets/compare_zero_shot.png) 

## 😊与其它少样本异常检测方法比较

![Compare_4](./assets/compare_few_shot.png) 

<span id='setup'/>

## 🎯代码环境配置: <a href='#all_catelogue'>[返回目录]</a>

### 环境:

- Python 3.8
- CUDA 11.7
- PyTorch 2.0.1

使用如下命令克隆该项目到本地:

```
git clone https://github.com/lianghaoyu99/MuSc-WG.git
```

创建虚拟环境:

```
conda create --name musc python=3.8
conda activate musc
```

安装依赖库:

```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

<span id='datasets'/>

## 👇数据集下载: <a href='#all_catelogue'>[返回目录]</a>

点击下载MVTec AD数据集[mvtec-musc.zip](https://pan.baidu.com/s/1cIsO7YHRv3XEVXk5CeN-gQ?pwd=xgfh)，提取码: xgfh 

把数据集解压后放在项目根目录下。

<span id='datatets_mvtec_ad'/>

### [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

```
data
|---mvtec_anomaly_detection
|-----|-- bottle
|-----|-----|----- ground_truth
|-----|-----|----- test
|-----|-----|----- train
|-----|-- cable
|-----|--- ...
```

<span id='run_musc'/>

## 💎运行主程序: <a href='#all_catelogue'>[返回目录]</a>

### python运行

```
python examples/musc_main.py
```
遵循`./configs/musc.yaml`中的设置。

关键参数如下：

- `--device`: GPU_id。
- `--data_path`: 数据集路径。
- `--dataset_name`: 数据集名称。
- `--class_name`: 进行测试的类别，如果该参数设置为`ALL`，将对所有的类别进行测试；##如果要对单一类别进行测试可设置对应的类别名称如`transistor`、`wood`等。
- `--backbone_name`: 特征提取器的名称，我们的代码兼容CLIP，DINO和DINO_v2，详见`configs/musc.yaml`。
- `--pretrained`: 选择预训练的CLIP模型，可选`openai`，`laion400m_e31`和`laion400m_e32`。
- `--feature_layers`: backbone中用于提取特征的层。
- `--img_resize`: 输入到模型中的图像大小。
- `--divide_num`: 将完整的无标签测试集划分为子集的数量。
- `--r_list`: LNAMD模块中的多个聚合度。
- `--output_dir`: 保存该方法预测的异常概率图和检测分割指标的路径。
- `--vis`: 是否保存该方法预测的异常概率图。
- `--vis_type`: 可在`single_norm`和`whole_norm`中进行选择，`single_norm`意思是将每张异常概率图进行归一化后再可视化，`whole_norm`意思是将全部异常概率图统一进行归一化后再可视化。
- `--save_excel`: 是否保存该方法异常检测和分割的指标。

### 针对LNAMD和MSM的修改

在`./models/musc.py`中的调用
```
# --- ABLATION STUDY CONFIGURATION ---  # 配置区，针对WTConv的参数主要在这里修改
# Change these values to test different settings:
ablation_wt_type = 'db1'        # 选择小波基：'db1' (Haar), 'db2', 'sym2', 'coif1', etc.
ablation_padding = 'reflect'    # 选择padding模式：'reflect', 'zeros', 'replicate'
ablation_level0  = False        # 是否包含原始特征
ablation_intra_weight = False    # 自评分，无效，保持False
ablation_gamma   = 2.0          # 伽马校正的参数，设为1时即关闭 (e.g., 2.0 - 4.0)
# Band-Pass / High-Pass Configuration
ablation_use_details = True     # 带通开关，是否保留LH, HL, HH细节
ablation_detail_start = 1       # 从第1层开始保留细节，0表示保留所有细节
ablation_keep_ll = True         # True: 保留低频 False: 仅保留高频细节

# print(f"Using Original LNAMD with r={r}, intra_weight={ablation_intra_weight}, gamma={ablation_gamma}")
# LNAMD_r = LNAMD(device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features_list)  # 如需切换原版LNAMD则取消注释这两行，并注释下面两行，反之亦然。

print(f"Using WTConvLNAMDStatic with r={r}, wt={ablation_wt_type}, pad={ablation_padding}, level0={ablation_level0}, intra_weight={ablation_intra_weight}, gamma={ablation_gamma}")
LNAMD_r = WTConvLNAMDStatic(device=self.device, feature_dim=feature_dim, feature_layer=self.features_list, r=r,
                            wt_type=ablation_wt_type, padding_mode=ablation_padding, include_level0=ablation_level0)  # WTConv的入口
```

原版LNAMD目前已替换成WTConv模块`./models/modules/WTConvStatic.py`，可以在这个模块上进行改进，入口即上面一行的代码。

原版MSM模块`./models/modules/_MSM.py`目前已引入伽马校正，可以在此模块上进行改进，入口在`./models/musc.py`中下面这行代码。
```
anomaly_maps_msm = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3, 
                                           use_intra_weight=ablation_intra_weight, gamma=ablation_gamma)  # 调用MSM算法生成异常图（同一层互相计算）
```


<span id='results_datasets'/>

## 🎖️不同数据集的结果: <a href='#all_catelogue'>[返回目录]</a>

以下所有的结果均按照论文中的默认设置复现。

### MVTec AD

|            | Classification |            |        | Segmentation |             |         |          |
| :--------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|  Category  |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|   bottle   |     99.92      |   99.21    | 99.98  |    98.48     |    79.17    |  83.04  |  96.10   |
|   cable    |     98.99      |   97.30    | 99.42  |    95.76     |    60.97    |  57.70  |  89.62   |
|  capsule   |     96.45      |   94.88    | 99.30  |    98.96     |    49.80    |  48.45  |  95.49   |
|   carpet   |     99.88      |   99.44    | 99.96  |    99.45     |    73.33    |  76.05  |  97.58   |
|    grid    |     98.66      |   96.49    | 99.54  |    98.16     |    43.94    |  38.24  |  93.92   |
|  hazelnut  |     99.61      |   98.55    | 99.79  |    99.38     |    73.41    |  73.28  |  92.24   |
|  leather   |     100.0      |   100.0    | 100.0  |    99.72     |    62.84    |  64.47  |  98.74   |
| metal_nut  |     96.92      |   97.38    | 99.25  |    86.12     |    46.22    |  47.54  |  89.34   |
|    pill    |     96.24      |   95.89    | 99.31  |    97.47     |    65.54    |  67.25  |  98.01   |
|   screw    |     82.17      |   88.89    | 90.88  |    98.77     |    41.87    |  36.12  |  94.40   |
|    tile    |     100.0      |   100.0    | 100.0  |    97.90     |    74.71    |  78.90  |  94.64   |
| toothbrush |     100.0      |   100.0    | 100.0  |    99.53     |    70.19    |  67.79  |  95.48   |
| transistor |     99.42      |   95.00    | 99.19  |    91.38     |    59.24    |  58.40  |  77.21   |
|    wood    |     98.51      |   98.33    | 99.52  |    97.24     |    68.64    |  74.75  |  94.50   |
|   zipper   |     99.84      |   99.17    | 99.96  |    98.40     |    62.48    |  61.89  |  94.46   |
|    mean    |     97.77      |   97.37    | 99.07  |    97.11     |    62.16    |  62.26  |  93.45   |

<span id='results_backbones'/>

## 🎖️使用不同特征提取器的结果: <a href='#all_catelogue'>[返回目录]</a>

我们论文中使用的默认特征提取器是CLIP的ViT-large-14-336。
我们还提供了CLIP、DINO和DINO_v2的vision transformer作为特征提取器的运行程序，具体信息详见`configs/musc.yaml`。

### MVTec AD

|                   |              |            | Classification |            |        | Segmentation |             |         |          |
| :---------------: | :----------: | :--------: | :------------: | :--------: | :----: | :----------: | :---------: | :-----: | :------: |
|     Backbones     | Pre-training | image size |   AUROC-cls    | F1-max-cls | AP-cls |  AUROC-segm  | F1-max-segm | AP-segm | PRO-segm |
|     ViT-B-32      |     CLIP     |    256     |     87.99      |   92.31    | 94.38  |    93.08     |    42.06    |  37.21  |  72.62   |
|     ViT-B-32      |     CLIP     |    512     |     89.91      |   92.72    | 95.12  |    95.73     |    53.32    |  52.33  |  83.72   |
|     ViT-B-16      |     CLIP     |    256     |     92.78      |   93.98    | 96.59  |    96.21     |    52.48    |  50.23  |  87.00   |
|     ViT-B-16      |     CLIP     |    512     |     94.20      |   95.20    | 97.34  |    97.09     |    61.24    |  61.45  |  91.67   |
| ViT-B-16-plus-240 |     CLIP     |    240     |     94.77      |   95.43    | 97.60  |    96.26     |    52.23    |  50.27  |  87.70   |
| ViT-B-16-plus-240 |     CLIP     |    512     |     95.69      |   96.50    | 98.11  |    97.28     |    60.71    |  61.29  |  92.14   |
|     ViT-L-14      |     CLIP     |    336     |     96.06      |   96.65    | 98.25  |    97.24     |    59.41    |  58.10  |  91.69   |
|     ViT-L-14      |     CLIP     |    518     |     95.94      |   96.32    | 98.30  |    97.42     |    63.06    |  63.67  |  92.92   |
|   ViT-L-14-336    |     CLIP     |    336     |     96.40      |   96.44    | 98.30  |    97.03     |    57.51    |  55.44  |  92.18   |
|   ViT-L-14-336    |     CLIP     |    518     |     97.77      |   97.37    | 99.07  |    97.11     |    62.16    |  62.26  |  93.45   |
|  dino_vitbase16   |     DINO     |    256     |     89.39      |   93.77    | 95.37  |    95.83     |    54.02    |  52.84  |  84.24   |
|  dino_vitbase16   |     DINO     |    512     |     94.11      |   96.13    | 97.26  |    97.78     |    62.07    |  63.20  |  92.49   |
|   dinov2_vitb14   |   DINO_v2    |    336     |     95.67      |   96.80    | 97.95  |    97.74     |    60.23    |  59.45  |  93.84   |
|   dinov2_vitb14   |   DINO_v2    |    518     |     96.31      |   96.87    | 98.32  |    98.07     |    64.65    |  65.31  |  95.59   |
|   dinov2_vitl14   |   DINO_v2    |    336     |     96.84      |   97.45    | 98.68  |    98.17     |    61.77    |  61.21  |  94.62   |
|   dinov2_vitl14   |   DINO_v2    |    518     |     97.08      |   97.13    | 98.82  |    98.34     |    66.15    |  67.39  |  96.16   |

<span id='inference_time'/>

## ⌛推理时间: <a href='#all_catelogue'>[返回目录]</a>

在下表中，我们展示了使用不用backbone和image size时的推理速度。
在计算推理速度时，我们设定一次性参与互打分的图像数量为**200**，所用GPU为单卡NVIDIA RTX 3090。

|                   |              |            |                 |
| :---------------: | :----------: | :--------: | :-------------: |
|     Backbones     | Pre-training | image size | times(ms/image) |
|     ViT-B-32      |     CLIP     |    256     |      48.33      |
|     ViT-B-32      |     CLIP     |    512     |      95.74      |
|     ViT-B-16      |     CLIP     |    256     |      86.68      |
|     ViT-B-16      |     CLIP     |    512     |      450.5      |
| ViT-B-16-plus-240 |     CLIP     |    240     |      85.25      |
| ViT-B-16-plus-240 |     CLIP     |    512     |      506.4      |
|     ViT-L-14      |     CLIP     |    336     |      266.0      |
|     ViT-L-14      |     CLIP     |    518     |      933.3      |
|   ViT-L-14-336    |     CLIP     |    336     |      270.2      |
|   ViT-L-14-336    |     CLIP     |    518     |      955.3      |
|  dino_vitbase16   |     DINO     |    256     |      85.97      |
|  dino_vitbase16   |     DINO     |    512     |      458.5      |
|   dinov2_vitb14   |   DINO_v2    |    336     |      209.1      |
|   dinov2_vitb14   |   DINO_v2    |    518     |      755.0      |
|   dinov2_vitl14   |   DINO_v2    |    336     |      281.4      |
|   dinov2_vitl14   |   DINO_v2    |    518     |     1015.1      |

<span id='FAQ'/>

## 🙋🙋‍♂️常见问题: <a href='#all_catelogue'>[返回目录]</a>

Q: 可视化图中正常的图像上为什么会出现大面积较高的异常分数？

A: 在可视化时，为了突出异常区域，我们默认采用了单图归一化，即便单图响应整体较低，经过归一化后也会出现大量的高亮区域。可通过在shell脚本中添加`vis_type`参数，并设置为`whole_norm`来进行全部图像一同归一化，也可通过修改`./configs/musc.yaml`配置文件中的`testing->vis_type`参数来实现相同的效果。

Q: 输入到模型中的图像分辨率如何选取？

A: 输入到模型中的图像分辨率`img_resize`一般为ViT patch size的倍数，可以防止边缘部分产生误检，常用的值为224、240、256、336、512、518，我们在上一节<a href='#results_backbones'>*(跳转)*</a>中展示了不同特征提取器常用的两种输入图像分辨率的大小，可供参考。
可通过修改shell脚本中的`img_resize`参数更改图像分辨率，也可通过修改`./configs/musc.yaml`配置文件中的`datasets->img_resize`参数来更改。



<span id='citation'/>

## 引用: <a href='#all_catelogue'>[返回目录]</a>
```
@inproceedings{Li2024MuSc,
  title={MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images},
  author={Li, Xurui and Huang, Ziming and Xue, Feng and Zhou, Yu},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

<span id='thanks'/>

## 致谢: <a href='#all_catelogue'>[返回目录]</a>

Our repo is built on [PatchCore](https://github.com/amazon-science/patchcore-inspection) and [APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN), thanks their clear and elegant code !

<span id='license'/>

## 使用许可: <a href='#all_catelogue'>[返回目录]</a>
MuSc is released under the **MIT Licence**, and is fully open for academic research and also allow free commercial usage. To apply for a commercial license, please contact yuzhou@hust.edu.cn.