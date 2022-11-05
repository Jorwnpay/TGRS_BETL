## BETL

**English|[简体中文](https://github.com/Jorwnpay/TGRS_BETL/blob/main/README_zh.md)**

This repo shows the source code of IEEE TGRS 2022 article: [**Sonar Images Classification While Facing Long-Tail and Few-Shot**](https://ieeexplore.ieee.org/document/9910166). In this work, we propose a pipeline entitled balanced ensemble transfer learning (BETL), which simultaneously overcomes the long-tail and few-shot problems in sonar image classification tasks. 

<img src=".\img\overview.png" alt="overview" style="zoom:60%;" />

## Running the Experiments

### Main Requirements

* Python == 3.6.12
* torch == 1.9.0
* torchvision == 0.10.0
* tensorboardX == 2.4.1

You can also install dependencies by

```shell
pip install -r requirements.txt
```

We recommend using anaconda to build your code environments.

### Experimental Environments

This repository is performed on an Intel Xeon E3-1275 v6 3.8 GHz central processing unit (CPU) with 32-GB RAM and an **NVIDIA GeForce RTX 2080Ti** graphic processing unit (GPU). The operating system is **Ubuntu 20.04**. The CUDA nad CUDNN version is **10.1** and **7.6.5** respectively.

### Dataset

We use SeabedObjects-KLSG (KLSG), long-tailed sonar image dataset (LTSID) and marine-debris-fls-datasets (FLSMDD) to test our BETL. KLSG and FLSMDD can be downloaded from [Huo's repo](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset) and [Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0), respectively. And LTSID is collected and sorted from the Internet. Due to copyright issue, LTSID cannot be open source at present. For the users' convenience, the prepared KLSG and FLSMDD can also be downloaded from this repo, just by cloning this repo and uncompressing all the `.rar` files under the data folder. NOTE that all these datasets were provided by Huo and Valdenegro and we just select what we need here. If you are going to use these datasets in your work, please cite their papers and star their repositories.

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

### Prepare

Here is a preparation step if you use the orginal datasets from [Huo's repo](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset) and [Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0). Note that if you use the prepared datasets in this repo, just uncompress all the `.rar` files under the data folder and **skip** this step.

Firstly, Download datasets from [Huo's repo](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset) and [Valdenegro's repo](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0) first, and adjust their file structure to:

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

Secondly, run following codes to generate data direction-label list (train.txt) and 10-trail 5-fold cross-validation index list (kfold_train.txt, kfold_val.txt).

```shell
# generate data direction-label list, use KLSG dataset as an example 
cd ./tool/
python generate_dir_lbl_list.py --dataset KLSG
# generate 10-trail 5-fold cross-validation index list, use KLSG dataset as an example 
python generate_kfold_idx_list.py --dataset KLSG
```

Now, you might get a data file structure like this:

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

### Training

For training BETL, here is an example for quick start,

```shell
# Demo: training on KLSG
cd ./code/
python betl.py --dataset KLSG
```

Here are explanations of some important args,

```
--dataset:      "the name of dataset, can be KLSG or FLSMDD, default is KLSG"
--p_value:      "the trail index of 10-trail 5-fold cross-validation, default is 0"
--k_value:      "the fold index of 10-trail 5-fold cross-validation, default is 0"
--backbone:     "the name of backbone, default is resnet18, can be resnet18, resnet34, resnet50, vgg16, vgg19"
--save_prop:    "classifier save proportion in ensemble pruning phase, default is 0.6"
--save_results: "if you want to save the validation results, default is True"
--save_models:  "if you want to save the models, default is False"
```

If you want to train BETL via a 10-trail 5-fold cross-validation scheme, run:

```shell
# Demo: training on KLSG, using resnet18 as backbone
cd ./tool/
./auto_run.sh ../code/betl.py KLSG resnet18
```

### Analyze Results

After training BETL via a 10-trail 5-fold cross-validation scheme, by default, you will get y_hat, y_true, and logits results in `"/output/results/{dataset}/{method}/{backbone}/"`, e.g., `"/output/results/KLSG/betl/resnet18/y_hat.txt"`. Then, you can get Gmean, Macro-F1, confusion matrix, and Precision-Recall curves results through:

```shell
# Demo: analyzing on KLSG, using resnet18 as backbone
cd ./code/
python analyse_result.py --dataset KLSG --method betl --backbone resnet18 --get_conf_matrix True --get_pr True
```

Here are explanations of some important args,

```
--dataset:          "the name of dataset, can be KLSG or FLSMDD, default is KLSG"
--method:           "the name of method, default is betl"
--backbone:         "the name of backbone, default is resnet18, can be resnet18, resnet34, resnet50, vgg16, vgg19"
--get_gmean:        "If you want to get Gmean result, default is True"
--get_f1:           "If you want to get Macro-F1 result, default is True"
--get_conf_matrix:  "If you want to get confusion matrix result, default is False"
--get_pr:           "If you want to get Precision-Recall curves result, default is False"
--get_macro_pr_all: "If you want to get macro average Precision-Recall curves result of all classes, default is False"
--get_macro_pr_tail: "If you want to get macro average Precision-Recall curves result of tail classes, default is False"
--show_pic: 		 "If you want to show confusion matrix or Precision-Recall curves, default is False"
```

##  Cite

If you find this code useful in your research, please consider citing us:

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


