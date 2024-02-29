<!-- # Slide-SAM -->
# Slide-SAM: Medical SAM meets sliding window

We upload the SlideSAM-H checkpoint recently!
Please download by 

Slide-SAM-B: https://pan.baidu.com/s/1jvJ2W4MK24JdpZLwPqMIfA [code：7be9]

SlideSAM-H: https://pan.baidu.com/s/1jnOwyWd-M1fBIauNi3IA4w [code: 05dy] 
## Before Training
### install tutils
```
pip install trans-utils
```

### prepare datasets
We recommend you to convert the dataset into the nnUNet format.
```
00_custom_dataset
  imagesTr
    xxx_0000.nii.gz
    ...
  labelsTr
    xxx.nii.gz
    ...
```
try to use the function ```organize_in_nnunet_style``` or ```organize_by_names``` to prepare your custom datasets.

Then run
```
python -m  datasets.generate_txt
```

A ```[example]_train.txt``` will be generated in ```./datasets/dataset_list/```

The content should be like below
```
01_BCV-Abdomen/Training/img/img0001.nii.gz	01_BCV-Abdomen/Training/label/label0001.nii.gz
01_BCV-Abdomen/Training/img/img0002.nii.gz	01_BCV-Abdomen/Training/label/label0002.nii.gz
01_BCV-Abdomen/Training/img/img0003.nii.gz	01_BCV-Abdomen/Training/label/label0003.nii.gz
```

### cache 3d data into slices
After generating the ```[example]_train.txt``` file, check the config file ```configs/vit_b.yaml```.

Update the params in ```dataset``` by yours. And the ```dataset_list``` should be the name of the generated txt file ```[example]```.

Then run
```
python -m datasets.cache_dataset3d
```

## Training
run training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.ddp --tag debug
```

## Testing
```
python -m core.volume_predictor
```
