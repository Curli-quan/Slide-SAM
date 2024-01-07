<!-- # Slide-SAM -->
# Slide-SAM: Medical SAM meets sliding window

We upload the checkpoint recently!
Please download by 

https://pan.baidu.com/s/1jvJ2W4MK24JdpZLwPqMIfA 

code：7be9 

## Before Training
install tutils
```
pip install trans-utils
```

prepare datasets
```
python -m  datasets.generate_txt
```

cache 3d data into slices
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
