# Unsupervised Discovery of Parts, Structure, and Dynamics  

This repository contains pre-trained models and testing code for PSD presented at ICLR 2019.

[http://psd.csail.mit.edu](http://psd.csail.mit.edu)

![Teaser Image](http://psd.csail.mit.edu/images/psd.png)


## Installation
Our current release has been tested on Ubuntu 18.04 LTS.

### Cloning the repository
```sh
git clone git@github.com:zhenjia-xu/psd.git
cd psd
```

### Set up Python environment:
```sh
pipenv install --dev
```

## Guide

### Generating shape dataset
```sh
pipenv run python data_generator.py
```
### Downloading the pretrained model (185 MB)
```sh
./download_model.sh
```

### Visualization
```sh
pipenv run python demo.py --gpu ID
```
- --gpu ID: use which gpu (starting from 0). Set to -1 to use CPU only.
  
The results will be presented in ```./demo/index.html```.



## Reference
```
@inproceedings{psd,
  title={Unsupervised Discovery of Parts, Structure, and Dynamics},
  author={Xu, Zhenjia and Liu, Zhijian and Sun, Chen and Murphy, Kevin and
        Freeman, William T and Tenenbaum, Joshua B and Wu, Jiajun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019},
}
```

For any questions, please contact Zhenjia Xu (xuzhenjia@cs.columbia.edu).
