# STAR
Code for [Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction](https://arxiv.org/abs/2005.08514)

### Environment

```bash
pip install numpy==1.18.1
pip install torch==1.7.0
pip install pyyaml=5.3.1
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
