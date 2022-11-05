## BETL

**[English](https://github.com/Jorwnpay/TGRS_BETL/blob/main/README.md)|简体中文**

这个仓库是IEEE TGRS 2022文章： [**Sonar Images Classification While Facing Long-Tail and Few-Shot**](https://ieeexplore.ieee.org/document/9910166) 的实现代码。在本文中，为同时解决声纳图像分类任务中的长尾和小样本问题，我们提出了一个平衡集成迁移学习（BETL）方法。

<img src=".\img\overview.png" alt="overview" style="zoom:60%;" />

## 运行实验

### 主要依赖包

* Python == 3.6.12
* torch == 1.9.0
* torchvision == 0.10.0
* tensorboardX == 2.4.1

也可以通过下面的命令安装环境依赖：

```shell
pip install -r requirements.txt
```

我们推荐使用anaconda来构建您的代码环境.

### 实验环境

我们在一台配置了英特尔 Xeon E3-1275 v6 3.8 GHz CPU、32-GB RAM 和 **NVIDIA GeForce RTX 2080Ti** GPU的服务器上对代码进行了实验. 操作系统是 **Ubuntu 20.04**. CUDA 和 CUDNN 的版本分别为 **10.1** 和 **7.6.5**。

### 数据集

我们使用了SeabedObjects-KLSG (KLSG)、long-tailed sonar image dataset (LTSID)和marine-debris-fls-datasets (FLSMDD)数据集来测试BETL。KLSG和FLSMDD数据集可以分别从[Huo's repo](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset) and [Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0)下载到。LTSID数据集是从网上收集整理得到的，由于版权问题，暂时不能开源。为了使用者的方便，预先准备好的KLSG和FLSMDD数据集也可以从这个仓库里下载，只需克隆这个仓库并解压data文件夹下的所有`.rar`文件即可。**请注意：** 所有这些数据集都是由Huo 和 Valdenegro提供的，我们这里仅仅是方便使用者跳过数据准备的步骤，直接运行代码进行实验。如果您准备在您的工作中使用这些数据集，请您引用他们的文章，并star他们的Github仓库，以对他们的贡献表示感谢。

```
% KLSG dataset is proposed in:
@article{huo2020underwater,
  title={Underwater object classification in sidescan sonar images using deep transfer learning and semisynthetic training data},
  author={Huo, Guanying and Wu, Ziyin and Li, Jiabiao},
  journal={IEEE access},
  volume={8},
  pages={47407--47418},
  year={2020},
  publisher={IEEE}
}
% FLSMDD dataset is proposed in:
@inproceedings{valdenegro2021pre,
  title={Pre-trained models for sonar images},
  author={Valdenegro-Toro, Matias and Preciado-Grijalva, Alan and Wehbe, Bilal},
  booktitle={OCEANS 2021: San Diego--Porto},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```

### 数据准备

如果您使用了[Huo's repo](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset) 和[Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0)的原始数据集，请阅读这个准备步骤。请注意，如果您使用了这个仓库中准备好的数据集，仅需要解压data文件夹下的所有 `.rar` 文件，然后直接**跳过**这个步骤。

首先，从[Huo's repo](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset) 和[Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0)下载数据集，并把原始文件的结构调整到下面这样：

```
data
├── KLSG
│   ├── plane
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── wreck
│       ├── 1.png
│       ├── 2.png
│       └── ...
└── FLSMDD
    ├── bottle
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    ├── can
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    └── ...
```

然后，运行下面的命令来生成 路径-标签 列表（train.txt）和十次五折交叉验证的序号列表（kfold_train.txt, kfold_val.txt）。

```shell
# generate data direction-label list, use KLSG dataset as an example 
cd ./tool/
python generate_dir_lbl_list.py --dataset KLSG
# generate 10-trail 5-fold cross-validation index list, use KLSG dataset as an example 
python generate_kfold_idx_list.py --dataset KLSG
```

现在，你应该得到了这样的文件结构：

```
data
├── KLSG
│   ├── train.txt
│   ├── kfold_train.txt
│   ├── kfold_val.txt
│   ├── plane
│   │   └── ...
│   └── wreck
│       └── ...
└── FLSMDD
    ├── train.txt
    ├── kfold_train.txt
    ├── kfold_val.txt
    ├── bottle
    │   └── ...
    ├── can
    │   └── ...
    └── ...
```

### 训练

要训练BETL，这里是一个快速开始的命令：

```shell
# Demo: training on KLSG
cd ./code/
python betl.py --dataset KLSG
```

这里是一些重要参数的解释：

```
--dataset:      "the name of dataset, can be KLSG or FLSMDD, default is KLSG"
--p_value:      "the trail index of 10-trail 5-fold cross-validation, default is 0"
--k_value:      "the fold index of 10-trail 5-fold cross-validation, default is 0"
--backbone:     "the name of backbone, default is resnet18, can be resnet18, resnet34, resnet50, vgg16, vgg19"
--save_prop:    "classifier save proportion in ensemble pruning phase, default is 0.6"
--save_results: "if you want to save the validation results, default is True"
--save_models:  "if you want to save the models, default is False"
```

通过下面的命令来对BETL进行十次五折交叉验证训练：

```shell
# Demo: training on KLSG, using resnet18 as backbone
cd ./tool/
./auto_run.sh ../code/betl.py KLSG resnet18
```

### 结果分析

在通过十次五折交叉验证训练完BETL之后，您会得到y_hat, y_true和logits结果，默认存放在`"/output/results/{dataset}/{method}/{backbone}/"`路径下，例如，`"/output/results/KLSG/betl/resnet18/y_hat.txt"`。然后，您可以使用下面的命令来得到Gmean、Macro-F1、混淆矩阵和P-R曲线结果：

```shell
# Demo: analyzing on KLSG, using resnet18 as backbone
cd ./code/
python analyse_result.py --dataset KLSG --method betl --backbone resnet18 --get_conf_matrix True --get_pr True
```

这里是一些重要参数的解释：

```
--dataset:           "the name of dataset, can be KLSG or FLSMDD, default is KLSG"
--method:            "the name of method, default is betl"
--backbone:          "the name of backbone, default is resnet18, can be resnet18, resnet34, resnet50, vgg16, vgg19"
--get_gmean:         "If you want to get Gmean result, default is True"
--get_f1:            "If you want to get Macro-F1 result, default is True"
--get_conf_matrix:   "If you want to get confusion matrix result, default is False"
--get_pr:            "If you want to get Precision-Recall curves result, default is False"
--get_macro_pr_all:  "If you want to get macro average Precision-Recall curves result of all classes, default is False"
--get_macro_pr_tail: "If you want to get macro average Precision-Recall curves result of tail classes, default is False"
--show_pic:          "If you want to show confusion matrix or Precision-Recall curves, default is False"
```

##  引用

如果您觉得这份代码对您的研究有帮助，请考虑引用我们：

```
@article{jiao2022sonar,
  title={Sonar Images Classification While Facing Long-Tail and Few-Shot},
  author={Jiao, Wenpei and Zhang, Jianlei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  volume={60},
  pages={1-20},
  publisher={IEEE}
}
```

