# STAR
Code for [Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction](https://arxiv.org/abs/2005.08514)

### Environment

```bash
pip install numpy==1.18.1
pip install torch==1.7.0 == 1.12.0
pip install pyyaml=5.3.1 == 6.0
pip install tqdm=4.45.0
```

### Train

The Default settings are to train on ETH-univ dataset. 

Data cache and models will be stored in the subdirectory "./output/eth/" by default. Notice that for this repo, we only provide implementation on GPU.

```
git clone https://github.com/Majiker/STAR.git
cd STAR
python trainval.py --test_set <dataset to evaluate> --start_test <epoch to start test>
```

Configuration files are also created after the first run, arguments could be modified through configuration files or command line. 
Priority: command line \> configuration files \> default values in script.

The datasets are selected on arguments '--test_set'. Five datasets in ETH/UCY including
[**eth, hotel, zara1, zara2, univ**].

### Example

This command is to train model for ETH-hotel and start test at epoch 10. For different dataset, change 'hotel' to other datasets named in the last section.

```
python trainval.py --test_set hotel --start_test 50
```

During training, the model for Best FDE on the corresponding test dataset would be record.

### Cite STAR

If you find this repo useful, please consider citing our paper
```bibtex
@inproceedings{
    YuMa2020Spatio,
    title={Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction},
    author={Cunjun Yu and Xiao Ma and Jiawei Ren and Haiyu Zhao and Shuai Yi},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    month = {August},
    year={2020}
}
```


### Reference

The code base heavily borrows from [SR-LSTM](https://github.com/zhangpur/SR-LSTM)

### New way to use eth,ucy数据集 
```angular2html
--dataset        普通单独数据集实验eth5 SDD 数据集迁移实验 eth5-SDD SDD-eth5
--test_set       sdd hotel eth univ zara1 zara2   //选取作为测试集的场景
--start-test     5    //何时开始在训练中运用测试数据进行测试
--phase          train test     // 阶段
--stage          origin  meta  MVDG   //origin表示最初模型，meta表示添加MLDG，
--meta_way       sequential1  parallel2   //两种MLDG的写法
--train_model    star  new_star      //  star 原始模型 new_star 加上CVAE并重构star
--SDD_if_filter  True  False   //True：过滤非行人数据 
--batch_around_ped   512        // 每批次包含行人的数目 主要为测试数据使用，或指标origin阶段
--batch_around_ped_meta 512     // 每批次包含行人的数目 meta写法中的task所包含的，

```
train-origin eth-ucy数据集 测试场景为hotel，训练阶段，新模型，训练范式原始，batch512
```
python trainval.py --dataset ETH_UCY --test_set hotel --start_test 5 --device 1  --phase train --stage origin --train_model new_star --batch_around_ped 512 
```
train-meta eth-ucy数据集 测试场景为hotel，训练阶段，新模型，训练范式MLDG，batch512,batchmeta512,mldg版本串行版 [效果最好版]
```
python trainval.py --dataset ETH_UCY --test_set hotel --start_test 5 --device 1  --phase train --stage meta --meta_way sequential1 --train_model new_star --batch_around_ped 512 --batch_around_ped_meta 512
```
train-meta SDD数据集 测试场景为sdd，训练阶段，新模型，训练范式MLDG，batch512,batchmeta512,mldg版本串行版,过滤行人
```angular2html

--dataset SDD --test_set sdd --start_test 0 --device 2 --meta_way sequential1 --SDD_if_filter True --phase train --stage meta --train_model new_star --batch_around_ped 512 --batch_around_ped_meta 512
```

