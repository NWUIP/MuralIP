# MuralIP
This is the official PyTorch implementation of MuralIP.
## ✨Prerequisites
### Installation
- Python 3.9
- PyTorch 1.2
- NVIDIA GPU + CUDA cuDNN
- Install python requirements:
```bash
pip install -e .
```
### Datasets
**Image Dataset.** We construct a new multimodal mural inpainting dataset, named **[CNMural-MM](https://pan.baidu.com/s/1mIl-Y5z0YqC3UFG61gmMfA)**. 
In addition, we evaluate our method on three widely-used public benchmarks, including 
[CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), 
[Paris StreetView](https://github.com/pathak22/context-encoder), 
and [Places2](http://places2.csail.mit.edu/).


**Mask Dataset.** Irregular masks are obtained from Irregular Masks and classified based on their hole sizes relative to the entire image with an increment of 10%.

## 🚀Getting Started
Download the pre-trained models using the following links and copy them into the `./ckpts` directory.
[CNMural-MM](https://pan.baidu.com/s/1mIl-Y5z0YqC3UFG61gmMfA) | [CelebA-HQ](https://pan.baidu.com/s/1mIl-Y5z0YqC3UFG61gmMfA) | [Paris StreetView](https://pan.baidu.com/s/1mIl-Y5z0YqC3UFG61gmMfA) | [Places2](https://pan.baidu.com/s/1mIl-Y5z0YqC3UFG61gmMfA)
### Testing
To test the model, you run the following code.
```bash
cd scripts
python test.py
```
### Training
To train the model, you run the following code.
```bash
cd scripts
python train.py
```

## 🥰Acknowledgements
Some of the code of this repo is borrowed from: 
- [Guided Diffusion](https://github.com/openai/guided-diffusion)
- [MAE-FAR](https://github.com/ewrfcas/MAE-FAR)

## 🤗To do list
- [x] Release source code
- [x] Provide detailed documentation
- [x] Release dataset
