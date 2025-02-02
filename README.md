##  Pippo: High-Resolution Multi-View Humans from a Single Image


We present Pippo, a generative model capable of producing 1K resolution dense turnaround videos of a person from a single casually clicked photo.
Pippo is a multi-view diffusion transformer and does not require any additional inputs — e.g., a fitted parametric model or camera parameters of the input image.



#### This is a code-only release without pre-trained weights. We provide models, configs, inference, and sample training code on Ava-256.


## Prerequisites and Dependencies
```
conda create -n pippo python=3.10.1
conda activate pippo

# can adjust as required (we tested on below configuration)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

```

## Setup code
Clone and add repository to your path:
```
git clone git@github.com:facebookresearch/pippo.git
cd pippo
export PATH=$PATH:$PWD
```
## Download and Sample Training
You can launch a sample training run on few samples of [Ava-256 dataset](https://github.com/facebookresearch/ava-256). We provide pre-packaged samples for this training stored as npy files [here](https://huggingface.co/datasets/yashkant/pippo/tree/main).
```
# download packaged Ava-256 samples
python scripts/pippo/download_samples.py

# launch training (on single A100 GPU)
 python latent_diffusion/overfit.py \
  config/pippo/head_only/128_4v_overfit.yml
```

## Useful Pointers
Here is a list of useful things to borrow from this codebase:
- ControlMLP to inject spatial control in Diffusion Transformers: `latent_diffusion/models/control_mlp.py`
- Attention Biasing to run inference on 5x longer sequences: `latent_diffusion/models/dit.py#161`
- Re-projection Error Metric: coming very soon!


## Todos
We plan to add and update the following in the future:
- Standalone module for computing Re-projection Error.
- Cleaning up fluff in pippo.py and dit.py
- Inference script for pretrained models.

## Citation
If you benefit from this codebase, consider citing our work:
```
@article{Kant2024Pippo,
  title={Pippo: High-Resolution Multi-View Humans from a Single Image},
  author={Yash Kant and Ethan Weber and Jin Kyu Kim and Rawal Khirodkar and Su Zhaoen and Julieta Martinez and Igor Gilitschenski and Shunsuke Saito and Timur Bagautdinov},
  year={2025},
}
```
