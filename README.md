<h2 align="center">Pippo: High-Resolution Multi-View Humans from a Single Image</h1>

<p align="center">
   <a href='https://yashkant.github.io/pippo/'>
      <img src='https://img.shields.io/badge/Pippo-Page-azure?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=000080&color=007FFF' alt='Project Page'>
   </a>

   <a href="https://yashkant.github.io/pippo/pippo.pdf">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
   </a>

   <a href='https://yashkant.github.io/pippo/#visuals'>
      <img src='https://img.shields.io/badge/Webpage-Visuals-orange?style=for-the-badge&&labelColor=FF5500&color=orange' alt='Spaces'>
   </a>

   <a href='https://drive.google.com/drive/folders/1UbAbfhjZxAFwHiQ1jXKDf_puIhTz-0Et'>
      <img src='https://img.shields.io/badge/More-Results-ffffff?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xOSAzSDVjLTEuMSAwLTIgLjktMiAydjE0YzAgMS4xLjkgMiAyIDJoMTRjMS4xIDAgMi0uOSAyLTJWNWMwLTEuMS0uOS0yLTItMnpNOSAxN0g3di01aDJ2NXptNCAwaC0ydi03aDJ2N3ptNCAwaC0yVjhoMnY5eiIvPjwvc3ZnPg==&logoColor=white&labelColor=8A2BE2&color=9370DB' alt='Visuals (Drive)'>
   </a>
</p>

<p align="center">
  <img src="./assets/pippo_short_github.gif" alt="Pippo" title="Pippo" width="1080"/>
</p>

<p align="center">
  <a href="https://yashkant.github.io/"><strong>Yash Kant</strong></a><sup>1,2,3</sup>
  ·
  <a href="https://ethanweber.me/"><strong>Ethan Weber</strong></a><sup>1,4</sup>
  ·
  <a href="https://scholar.google.com/citations?user=ki5hheQAAAAJ&amp;hl=en"><strong>Jin Kyu Kim</strong></a><sup>1</sup>
  ·
  <a href="https://rawalkhirodkar.github.io/"><strong>Rawal Khirodkar</strong></a><sup>1</sup>
  ·
  <a href="https://www.linkedin.com/in/suzhaoen/"><strong>Su Zhaoen</strong></a><sup>1</sup>
  ·
  <a href="https://una-dinosauria.github.io/"><strong>Julieta Martinez</strong></a><sup>1</sup>
  <br>
  <a href="https://www.gilitschenski.org/igor/"><strong>Igor Gilitschenski*</strong></a><sup>2,3</sup>
  ·
  <a href="https://shunsukesaito.github.io/"><strong>Shunsuke Saito*</strong></a><sup>1</sup>
  ·
  <a href="https://scholar.google.ch/citations?user=oLi7xJ0AAAAJ&amp;hl=en"><strong>Timur Bagautdinov*</strong></a><sup>1</sup>
  <p align="center">* Joint Advising</p>
  <p align="center">
      <sup>1</sup> Meta Reality Labs · 
      <sup>2</sup> University of Toronto · 
      <sup>3</sup> Vector Institute · 
      <sup>4</sup> UC Berkeley
  </p>
</p>



We present Pippo, a generative model capable of producing 1K resolution dense turnaround videos of a person from a single casually clicked photo.
Pippo is a multi-view diffusion transformer and does not require any additional inputs — e.g., a fitted parametric model or camera parameters of the input image.



#### This is a code-only release without pre-trained weights. We provide models, configs, inference, and sample training code on Ava-256.

## Setup code
Clone and add repository to your path:
```
git clone git@github.com:facebookresearch/pippo.git
cd pippo
export PATH=$PATH:$PWD
```

## Prerequisites and Dependencies
```
conda create -n pippo python=3.10.1 -c conda-forge
conda activate pippo

# can adjust as required (we tested on the below configuration)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.0 -c pytorch -c nvidia

pip install -r requirements.txt

```

## Download and Sample Training
You can launch a sample training run on a few samples of [Ava-256 dataset](https://github.com/facebookresearch/ava-256). We provide pre-packaged samples for this training stored as npy files [here](https://huggingface.co/datasets/yashkant/pippo/tree/main). Ensure you are authenticated to huggingface with a login token to download the samples.
```
# download packaged Ava-256 samples
python scripts/pippo/download_samples.py
```

We provide exact model configs to train Pippo models at different resolutions of 128, 512, and 1024 placed in `config/full/` directory.
```
# launch training (tested on single A100 GPU 80GB): full sized model
 python train.py config/full/128_4v.yml
```

Additionally, we provide a tiny model config to train on a smaller gpu:
```
# launch training (tested on single T4 GPU 16GB): tiny model
python train.py config/tiny/128_4v_tiny.yml
```

## Training on custom dataset (see https://github.com/facebookresearch/pippo/issues/9): 
You will have to prepare your dataset similar to the provided [Ava-256 samples stored in numpy files](https://huggingface.co/datasets/yashkant/pippo/tree/main/ava_samples) on your custom dataset.

The difficult bits could be to create the Plucker Rays and Spatial Anchor images, and we have provided our implementations for those methods (using Ava-256 and Goliath data) [in this gist here](https://gist.github.com/yashkant/971e205d85b15e17d20d33edd29d6016). You can refer these methods to create these fields on your own custom datasets!


## Re-projection Error
To compute the re-projection error between generated images and ground truth images, run the following command:
```
python scripts/pippo/reprojection_error.py
```


## Useful Pointers
Here is a list of useful things to borrow from this codebase:
- ControlMLP to inject spatial control in Diffusion Transformers: [see here](https://github.com/facebookresearch/pippo/blob/main/latent_diffusion/models/control_mlp.py#L161)
- Attention Biasing to run inference on 5x longer sequences: [see here](https://github.com/facebookresearch/pippo/blob/main/latent_diffusion/models/dit.py#L165)
- Re-projection Error Metric: [see here](https://github.com/facebookresearch/pippo/blob/main/scripts/pippo/reprojection_error.py#L150)


## Todos
We plan to add and update the following in the future:
- Cleaning up fluff in pippo.py and dit.py
- Inference script for pretrained models.

## License

See LICENSE file for details.

## Citation
If you benefit from this codebase, consider citing our work:
```
@article{Kant2024Pippo,
  title={Pippo: High-Resolution Multi-View Humans from a Single Image},
  author={Yash Kant and Ethan Weber and Jin Kyu Kim and Rawal Khirodkar and Su Zhaoen and Julieta Martinez and Igor Gilitschenski and Shunsuke Saito and Timur Bagautdinov},
  year={2025},
}
```
