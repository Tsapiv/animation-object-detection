## AI Project. Animation-object-detection
### How to run
```shell
$ python train.py --config configs/<some config from here>
```

### How to run metrics evaluation + visualization
```shell
$ python classify.py --config configs/google-doodle-10.json --model_type ['vae|cnn'] --checkout_path path/to/models/weights 
```

### How to generate own dataset
```shell
$ python generate_dataset.py --raw-data [*.npy files of choosen classes] --name [give name to dataset]
```

### How to use VAE as image generator
```shell
$ python random_generation.py --checkout_path path/to/models/weights <VAE only>
```


### Where to get pretrained models - [here](https://drive.google.com/drive/folders/1P2zstNKqi1ut_D2emHOwIYmfynfTA-U9?usp=sharing)

### Where to get Sketch CIFAR-10 - [here](https://drive.google.com/file/d/1xEjIIR35wmcwdwiYjAmfVob1ParS2a3g/view?usp=sharing)

### Where to get *.npy files of Google doodle dataset - [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=true)
