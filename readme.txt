**********************************
运行环境：Anaconda2+py3.6+keras-tensorflow
文件说明：
1.train: 训练集及其标签 请将该目录下nodule下的candidate1换成所有训练集 或者更换train中文件路径
2.test：待预测测试集  请将该目录下candidate11换成所有测试集 或者更换test中文件路径
3.trainfiles: 一些调参、更换网络和数据增强操作的尝试性文件
4.train.py：最终训练脚本:
5.mylib:
   (1)model:包含训练模型文件 densenet resnet3D
   (2) dataloader:包含transform.py 数据增强的一些对数据的操作
   (3)util
6. test.py： 预测脚本，输出submission.csv供助教老师验证使用
7.weight.final.h5：最终权重文件
8.sampleSubmission.csv、submit.csv：用于candidate读取操作
9.run_test.sh,run_train.sh:交大云任务递交文件
*******************************************
