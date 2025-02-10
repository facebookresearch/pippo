##  Pippo: High-Resolution Multi-View Humans from a Single Image

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

<div class="row">
            <div class="col-md-8 col-md-offset-2">
                <ul class="nav nav-pills nav-justified">
                    <li>
                        <a href="pippo.pdf">
                            <img src="img/pippo-ss.png" height="60px" alt="Paper">
                            <h4><strong>Paper</strong></h4>
                        </a>
                    </li>
                    <li>
                        <a href="https://github.com/facebookresearch/pippo">
                            <img src="img/github.png" height="60px" alt="GitHub">
                            <h4><strong>Code</strong></h4>
                        </a>
                    </li>
                    <li>
                        <a href="#visuals">
                            <img src="img/webpage.svg" height="60px" alt="Drive">
                            <h4><strong>Jump to Visuals</strong></h4>
                        </a>
                    </li>
                    <li>
                        <a href="https://drive.google.com/drive/folders/1UbAbfhjZxAFwHiQ1jXKDf_puIhTz-0Et?usp=sharing">
                            <img src="img/drive_icon.png" height="60px" alt="Drive">
                            <h4><strong>Visuals (Drive)</strong></h4>
                        </a>
                    </li>
                </ul>
            </div>
        </div>


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

# can adjust as required (we tested on below configuration)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.0 -c pytorch -c nvidia

pip install -r requirements.txt

```

## Download and Sample Training
You can launch a sample training run on few samples of [Ava-256 dataset](https://github.com/facebookresearch/ava-256). We provide pre-packaged samples for this training stored as npy files [here](https://huggingface.co/datasets/yashkant/pippo/tree/main). Ensure you are authenticated to huggingface with login token to download the samples.
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
