# ReIDCompetition

# 这是“武大锅盔队”在行人重试别赛道的提交审核的代码。

由于在K-Lab项目中已经队项目进行了详细的介绍，因此这里我们只简单介绍项目的训练和测试流程，感谢您的审查。

KLab'项目链接：https://www.kesci.com/home/project/share/926c665b7bd3a61a

## 1.准备

下载本项目：
`git clone https://github.com/ChronousZhang/ReIDCompetition.git`
切换到下面目录：
`cd ReIDCompetition/single_model`
修改torchreid.data.datamanager.py下的MyDataset类的
self.train_dir = osp.join(root, "MyDataset_V2/bounding_box_train_remove_1_and_2_v2")  # root+该str路径为训练集具体路径
self.val_query_dir = osp.join(root, "MyDataset_V2/query_v2")  # root+该str路径为验证集query具体路径
self.val_gallery_dir = osp.join(root, "MyDataset_V2/bounding_box_test_distractor_v2")  # root+该str路径为验证集gallery具体路径
self.query_dir = osp.join(root, "chu_test_b/chu_test_b/query_b/query_b")  # root+该str路径为测试集A或B的query具体路径
self.gallery_dir = osp.join(root, "chu_test_b/chu_test_b/gallery_b/gallery_b")  # root+该str路径为测试集A或B的gallery具体路径
