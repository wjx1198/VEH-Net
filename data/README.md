# Visual Enhanced Hierarchical Network for Sentence-Based Video Thumbnail Generation

## Introduction

Sentence specified dynamic video thumbnail generation aims to dynamically select and concatenate video clips from an original video to generate one video thumbnail, which not only provides a concise preview of the original video but also semantically corresponds to the given sentence description.

## Dataset

See the detail in the paper: Sentence Specified Dynamic Video Thumbnail Generation

## Model implementation
Code for data processing, model construction, and model training and testing.

#### ./src/data_prepare/generate_batch_data.py
* Data preprocessing for model training, testing and validation.  If run
```
python generate_batch_data.py
```
A folder './data' will be constructed. Three subdirs are in this folder, which contain the train, test, and validation h5py files, respectively. 

#### ./src/VEH-Net (VEH-Net_unsup)
* The VEH-Net model as well as the unsupervised version. Please see the paper for details.
* Model training:
```
python run_VEHNet.py --task train
```
* Model testing:
```
python run_VEHNet.py --task test
```

## Reference

https://github.com/yytzsy/GTP





