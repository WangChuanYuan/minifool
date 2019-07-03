# minifool
Generate adversarial samples for FASHION-MNIST dataset

## nets
Keras models trained on FASHION-MNIST dataset

## models
Keras models exported

## train
Train code for the models in package nets

## util
Include SSIM calculator and different evolution implementation

## adv
The attack methods implemented  
Include:  
* **Iterative FGSM with momentum** references [动量迭代式对抗噪声生成方法](https://www.jiqizhixin.com/articles/2019-05-21-10)
* **One pixel attack** references [one-pixel-attack-keras](https://github.com/Hyperparticle/one-pixel-attack-keras)