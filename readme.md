<!-- # Slide-SAM -->
# Slide-SAM: Medical SAM meets sliding window


## Training
prepare datasets
```
python -m  datasets.generate_txt
```

cache 3d data into slices
```
python -m datasets.cache_datasets3d
```

run training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.ddp --tag debug
```