<!-- # Slide-SAM -->
# Slide-SAM: Medical SAM meets sliding window

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
python -m datasets.cache_datasets3d
```

## Training
run training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.ddp --tag debug
```
