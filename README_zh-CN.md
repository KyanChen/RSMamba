<div align="center">
    <h2>
        RSMamba: Remote Sensing Image Classification with State Space Model
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/RSMamba.png" width="800"/>
</div>
<br>
<div align="center">
  <a href="https://kychen.me/RSMamba">
    <span style="font-size: 20px; ">项目主页</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2403.19654">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="resources/RSMamba.pdf">
    <span style="font-size: 20px; ">PDF</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/KyanChen/RSMamba">
    <span style="font-size: 20px; ">HFSpace</span>
  </a>
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/KyanChen/RSMamba)](https://github.com/KyanChen/RSMamba)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2403.19654-b31b1b.svg)](https://arxiv.org/abs/2403.19654)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/KyanChen/RSMamba)

<br>
<br>

<div align="center">

[English](README.md) | 简体中文

</div>


## 简介

本项目仓库是论文 [RSMamba: Remote Sensing Image Classification with State Space Model](https://arxiv.org/abs/2403.19654) 的代码实现，基于 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 项目进行开发。

当前分支在 Linux 系统，PyTorch 2.x 和 CUDA 12.1 下测试通过，支持 Python 3.8+，能兼容绝大多数的 CUDA 版本。

如果你觉得本项目对你有帮助，请给我们一个 star ⭐️，你的支持是我们最大的动力。

<details open>
<summary>主要特性</summary>

- 与 MMPretrain 高度保持一致的 API 接口及使用方法
- 开源了论文中不同版本大小的 RSMamba 模型
- 支持了多种数据集的训练和测试

</details>

## 更新日志

🌟 **2024.03.28** 发布了 RSMamba 项目，完全与 MMPretrain 保持一致的API接口及使用方法。

🌟 **2024.03.29** 开源了论文中不同版本大小的 RSMamba 模型的[权重文件](https://huggingface.co/KyanChen/RSMamba/tree/main)。


## TODO

- [X] 开源模型训练参数

## 目录

- [简介](#简介)
- [更新日志](#更新日志)
- [TODO](#TODO)
- [目录](#目录)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [图像预测](#图像预测)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [引用](#引用)
- [开源许可证](#开源许可证)
- [联系我们](#联系我们)

## 安装

### 依赖项

- Linux 系统， Windows 未测试，依赖于是否能安装 `causal-conv1d` 和 `mamba-ssm`
- Python 3.8+，推荐使用 3.11
- PyTorch 2.0 或更高版本，推荐使用 2.1
- CUDA 11.7 或更高版本，推荐使用 12.1
- MMCV 2.0 或更高版本，推荐使用 2.1

### 环境安装

推荐使用 Miniconda 来进行安装，以下命令将会创建一个名为 `rsmamba` 的虚拟环境，并安装 PyTorch 和 MMCV。下述安装步骤中，默认安装的 CUDA 版本为 **12.1**，如果你的 CUDA 版本不是 12.1，请根据实际情况进行修改。

注解：如果你对 PyTorch 有经验并且已经安装了它，你可以直接跳转到下一小节。否则，你可以按照下述步骤进行准备。

<details open>

**步骤 0**：安装 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)。

**步骤 1**：创建一个名为 `rsmamba` 的虚拟环境，并激活它。

```shell
conda create -n rsmamba python=3.11 -y
conda activate rsmamba
```

**步骤 2**：安装 [PyTorch2.2.x](https://pytorch.org/get-started/locally/)。

Linux/Windows:

```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 -y
```
或者
```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**步骤 3**：安装 [MMCV2.1.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

```shell
pip install -U openmim
mim install mmcv==2.1.0
#或者
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

**步骤 4**：安装其他依赖项。

```shell
pip install -U mat4py ipdb
pip install transformers==4.39.2
pip install causal-conv1d==1.2.0.post2
pip install mamba-ssm==1.2.0.post1
```

</details>


### 安装 RSMamba


下载或克隆 RSMamba 仓库即可。

```shell
git clone git@github.com:KyanChen/RSMamba.git
cd RSMamba
```

## 数据集准备

<details open>

### 遥感图像分类数据集

我们提供论文中使用的遥感图像分类数据集的准备方法。

#### UC Merced 数据集

- 图片及标注下载地址：[UC Merced 数据集](http://weegee.vision.ucmerced.edu/datasets/landuse.html)。


#### AID 数据集

- 图片及标注下载地址： [AID 数据集](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)。


#### NWPU RESISC45 数据集

- 图片及标注下载地址： [NWPU RESISC45 数据集](https://aistudio.baidu.com/datasetdetail/220767)。


**注解**：本项目的 `data` 文件夹提供了上述数据集的少量图片标签示例。

#### 组织方式

你也可以选择其他来源进行数据的下载，但是需要将数据集组织成如下的格式：

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/UC
├── airplane
│   ├── airplane01.tif
│   ├── airplane02.tif
│   └── ...
├── ...
├── ...
├── ...
└── ...
```
注解：在项目文件夹 `datainfo` 中，我们提供了数据集的划分文件。您也可以使用 [Python 脚本](tools/rsmamba/split_trainval.py) 来划分数据集。

### 其他数据集

如果你想使用其他数据集，可以参考 [MMPretrain 文档](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html) 来进行数据集的准备。

</details>

## 模型训练

### RSMamba 模型

#### Config 文件及主要参数解析

我们提供了论文中不同参数大小的 RSMamba 模型的配置文件，你可以在 [配置文件](configs/rsmamba) 文件夹中找到它们。Config 文件完全与 MMPretrain 保持一致的 API 接口及使用方法。下面我们提供了一些主要参数的解析。如果你想了解更多参数的含义，可以参考 [MMPretrain 文档](https://mmpretrain.readthedocs.io/zh-cn/latest/user_guides/config.html)。

<details>

**参数解析**：

- `work_dir`：模型训练的输出路径，一般不需要修改。
- `code_root`：代码根目录，**修改为本项目根目录的绝对路径**。
- `data_root`：数据集根目录，**修改为数据集根目录的绝对路径**。
- `batch_size`：单卡的 batch size，**需要根据显存大小进行修改**。
- `max_epochs`：最大训练轮数，一般不需要修改。
- `vis_backends/WandbVisBackend`：网络端可视化工具的配置，**打开注释后，需要在 `wandb` 官网上注册账号，可以在网络浏览器中查看训练过程中的可视化结果**。
- `model/backbone/arch`：模型的骨干网络类型，**需要根据选择的模型进行修改**，包括 `b`, `l`, `h`。
- `model/backbone/path_type`：模型的路径类型，**需要根据选择的模型进行修改**。
- `default_hooks-CheckpointHook`：模型训练过程中的检查点保存配置，一般不需要修改。
- `num_classes`：数据集的类别数，**需要根据数据集的类别数进行修改**。
- `dataset_type`：数据集的类型，**需要根据数据集的类型进行修改**。
- `resume`: 是否断点续训，一般不需要修改。
- `load_from`：模型的预训练的检查点路径，一般不需要修改。
- `data_preprocessor/mean/std`：数据预处理的均值和标准差，**需要根据数据集的均值和标准差进行修改**，一般不需要修改，参考 [Python 脚本](tools/rsmamba/get_dataset_img_meanstd.py)。

一些参数来源于 `_base_` 的继承值，你可以在 [基础配置文件](configs/rsmamba/_base_/) 文件夹中找到它们。

</details>


#### 单卡训练

```shell
python tools/train.py configs/rsmamba/name_to_config.py  # name_to_config.py 为你想要使用的配置文件
```

#### 多卡训练

```shell
sh ./tools/dist_train.sh configs/rsmamba/name_to_config.py ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，GPU_NUM 为使用的 GPU 数量
```

### 其他图像分类模型

<details open>

如果你想使用其他图像分类模型，可以参考 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 来进行模型的训练，也可以将其Config文件放入本项目的 `configs` 文件夹中，然后按照上述的方法进行训练。

</details>

## 模型测试

#### 单卡测试：

```shell
python tools/test.py configs/rsmamba/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件
```

#### 多卡测试：

```shell
sh ./tools/dist_test.sh configs/rsmamba/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，GPU_NUM 为使用的 GPU 数量
```


## 图像预测

#### 单张图像预测：

```shell
python demo/image_demo.py ${IMAGE_FILE}  configs/rsmamba/name_to_config.py --checkpoint ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}  # IMAGE_FILE 为你想要预测的图像文件，name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，OUTPUT_DIR 为预测结果的输出路径
```

#### 多张图像预测：

```shell
python demo/image_demo.py ${IMAGE_DIR}  configs/rsmamba/name_to_config.py --checkpoint ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}  # IMAGE_DIR 为你想要预测的图像文件夹，name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，OUTPUT_DIR 为预测结果的输出路径
```



## 常见问题

<details open>

我们在这里列出了使用时的一些常见问题及其相应的解决方案。如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。如果您无法在此获得帮助，请使用[issue](https://github.com/KyanChen/RSMamba/issues)来寻求帮助。请在模板中填写所有必填信息，这有助于我们更快定位问题。

### 1. 是否需要安装MMPretrain？

我们建议您不要安装MMPretrain，因为我们已经对MMPretrain的代码进行了部分修改，如果您安装了MMPretrain，可能会导致代码运行出错。如果你出现了模块尚未被注册的错误，请检查：

- 是否安装了MMPretrain，若有则卸载
- 是否在类名前加上了`@MODELS.register_module()`，若没有则加上
- 是否在`__init__.py`中加入了`from .xxx import xxx`，若没有则加上
- 是否在Config文件中加入了`custom_imports = dict(imports=['mmpretrain.rsmamba'], allow_failed_imports=False)`，若没有则加上


### 2. dist_train.sh: Bad substitution的解决

如果您在运行`dist_train.sh`时出现了`Bad substitution`的错误，请使用`bash dist_train.sh`来运行脚本。

### 3. 安装 causal-conv1d 和 mamba-ssm 失败

- 如果您在安装 causal-conv1d 和 mamba-ssm 时出现了错误，请检查您的CUDA版本是否与安装包的要求一致。
- 如果一致仍出现问题，请下载对应预编译包，然后使用`pip install xxx.whl`来安装。参考 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) 和 [mamba-ssm](https://github.com/state-spaces/mamba/releases)。

</details>

## 致谢

本项目基于 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 进行开发，感谢 MMPretrain 项目提供的代码基础。

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 RSMamba。

```
@article{chen2024rsmamba,
  title={RSMamba: Remote Sensing Image Classification with State Space Model},
  author={Chen, Keyan and Chen, Bowen and Liu, Chenyang and Li, Wenyuan and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2403.19654},
  year={2024}
}
```

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 联系我们

如果有其他问题❓，请及时与我们联系 👬
