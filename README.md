## PyTorch GAN
It is the raw code to **generate a quadratic girl image** from **a random noise array** using a GAN(Generative Adversarial Networks ) network. It is based on ([GAN二次元头像生成Pytorch实现（附完整代码)(https://blog.csdn.net/qq_36937684/article/details/106215485)]

This article is Pytorch version of [Li Hongyi GAN course assignments, HW3_1 (quadratic image generation, Keras implementation)](https://blog.csdn.net/space_walk/article/details/102658047). The reason for writing this article is that on the one hand, I want to understand GAN, and on the other hand, I am used to using Pytorch, so I changed keras into Pytorch version.

## Requirements
* PyTorch
* torchvision
* visdom
* matplotlib

## Training
Resources required for  implementation:

> link: [https://pan.baidu.com/s/1cLmFNQpJe1DOI96IVuvVyQ](https://pan.baidu.com/s/1cLmFNQpJe1DOI96IVuvVyQ)
 extract code: nha2

> usage: [--train][--GPU]  <br> 
> &#12288;&#12288;&#12288;&#32;[--continue_training] [--cuda]	<br>  
> &#12288;&#12288;&#12288;&#32;[--datapath]   [--latent_dim]  <br>
> &#12288;&#12288;&#12288;&#32;[--num_epoch] [--batch_size] <br>

Example: 

```python
python shizuo_gan_new.py --cuda --GPU 1 --batch_size 64 --train 1 --num_epoch 300
```
This will start a training session in the GPU.
 
## Testing
 > usage: [--test] [--GPU]  <br> 
> &#12288;&#12288;&#12288;&#32; [--cuda]	[--testmodelpath]     <br>
> &#12288;&#12288;&#12288;&#32;[--datapath]   [--latent_dim]  <br>


## Results

The result is no different from the original keras. After all, the network is similar and does not need too high expectations. Moreover, the network itself is relatively small.
Partial results are shown here.
<div align='center'>
<img src='https://img-blog.csdnimg.cn/20200519154631702.png'>
</div>
<div align='center'>
<img src='https://img-blog.csdnimg.cn/20200519155348525.png'>
</div>
<div align='center'>
<img src='https://img-blog.csdnimg.cn/2020051915471776.png'>
</div>

<div align='center'>
<img src='https://img-blog.csdnimg.cn/20200519155526940.png'>
</div>
