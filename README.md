# ReIDCompetition初赛审核代码

# 这是“武大锅盔队”在行人重试别赛道的提交审核的代码。

由于在K-Lab项目中已经队项目进行了详细的介绍，因此这里我们只简单介绍项目的训练和测试流程，感谢您的审查。

KLab项目链接：https://www.kesci.com/home/project/share/926c665b7bd3a61a

## 1.准备

下载本项目：
`git clone https://github.com/ChronousZhang/ReIDCompetition.git`

切换到下面目录：
`cd ReIDCompetition/single_model`

修改torchreid.data.datamanager.py下的MyDataset类的

`self.train_dir`
`self.val_query_dir`
`self.val_gallery_dir`
`self.query_dir`
`self.gallery_dir`

为对应训练集，验证集query，gallery和测试集query，gallery目录。

## 2.训练
修改`run.py`中的`save_dir`为文件(模型，日志等)保存地址，`torchreid.data.datamanager.MyDataManager`中的`root`参数为数据集所在大目录(该目录下为包含数据集的完整目录，例如 `root` + `self.train_dir`即为训练集完整目录)。然后运行run.py，即
`python3 run.py`
以执行一次完整的模型训练和测试(但这个测试结果是单模的，我们不作为最终结果)

我们训练好的5个模型链接如下：
https://drive.google.com/open?id=1dVwq64QDgKg7DYqObzqjYWqDJJVDUTAR

## 2.测试
切换到多模型测试目录：`cd ../jicheng`
类似地修改torchreid.data.datamanager.py下的MyDataset类的下的相关路径，然后修改`jicheng5.py`下的`save_dir`为文件(json文件，日志等)保存地址，`torchreid.data.datamanager.MyDataManager`中的`root`参数为数据集所在大目录以及`fpath_1`~`fpath5`为训练好的5个模型所在目录，然后运行jicheng5.py即可进行测试：
`python3 jicheng5.py`

生成的json文件保存在`save_dir`下，我们生成的结果放在下面的链接中：
https://drive.google.com/open?id=1dVwq64QDgKg7DYqObzqjYWqDJJVDUTAR

所用到的预训练模型是在ImageNet上预训练的SeNet模型，链接为：
http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth

感谢您的审阅。


以上代码是参与初赛评审的最终代码，下面介绍一下我们使用的模型：
## 1. 模型
我们使用的模型基于strong baseline，可参考https://github.com/michuanhaohao/reid-strong-baseline
模型如下：
![ ](https://github.com/ChronousZhang/ReIDCompetition/blob/master/pipeline.jpg)
