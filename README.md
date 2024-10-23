## VideoAgent

The official codebase for training video policies in VideoAgent

NEWS: We have released another repository for running our Meta-World and iTHOR experiments [here](https://github.com/Video-as-Agent/VideoAgent_exp)!

This repository contains the code for training video policies presented in our work   
[VideoAgent: Self improving video generation](https://arxiv.org/pdf/2410.10076)  
[Achint Soni](https://trickyjustice.github.io),
[Sreyas Venkataraman](https://github.com/vsreyas),
[Abhranil Chandra](https://abhranilchandra.github.io),
[Sebastian Fischmeister](https://uwaterloo.ca/embedded-software-group/profiles/sebastian-fischmeister),
[Percy Liang](https://cs.stanford.edu/~pliang/),
[Bo Dai](https://bo-dai.github.io),
[Sherry Yang](https://sherryy.github.io)
[website](https://video-as-agent.github.io) | [paper](https://arxiv.org/pdf/2410.10076) | [arXiv](https://arxiv.org/abs/2410.10076) | [experiment repo](https://github.com/Video-as-Agent/VideoAgent_exp)

```bib
@misc{soni2024videoagentselfimprovingvideogeneration,
      title={VideoAgent: Self-Improving Video Generation}, 
      author={Achint Soni and Sreyas Venkataraman and Abhranil Chandra and Sebastian Fischmeister and Percy Liang and Bo Dai and Sherry Yang},
      year={2024},
      eprint={2410.10076},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.10076}, 
}
```

## Getting started  

We recommend to create a new environment with pytorch installed using conda.   

```bash  
conda create -n videoagent python=3.9
conda activate videoagent
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```  

Next, clone the repository and install the requirements  

```bash
git clone https://github.com/Video-as-Agent/VideoAgent
cd VideoAgent
pip install -r requirements.txt
```


## Dataset structure

The pytorch dataset classes are defined in `flowdiffusion/datasets.py`


## Training models

For Meta-World experiments, run
```bash
cd flowdiffusion
python train_mw.py --mode train
# or python train_mw.py -m train
```

or run with `accelerate`
```bash
accelerate launch train_mw.py
```

For iTHOR experiments, run `train_thor.py` instead of `train_mw.py`  
For bridge experiments, run `train_bridge.py` instead of `train_mw.py`  

The trained model should be saved in `../results` folder  

To resume training, you can use `-c` `--checkpoint_num` argument.  
```bash
# This will resume training with 1st checkpoint (should be named as model-1.pt)
python train_mw.py --mode train -c 1
```

## Inferencing

Use the following arguments for inference  
`-p` `--inference_path`: specify input image path  
`-t` `--text`: specify the text discription of task   
`-n` `sample_steps` Optional, the number of steps used in test time sampling. If the specified value less than 100, DDIM sampling will be used.  
`-g` `guidance_weight` Optional, The weight used for classifier free guidance. Set to positive to turn on classifier free guidance.   

For example:  
```bash
python train_mw.py --mode inference -c 1 -p ../examples/assembly.png -t assembly -g 2 -n 20
```

## Pretrained models 

We also provide checkpoints of the models described in our experiments as following.   
### Meta-World
[VideoAgent](https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-305.pt) |  [VideoAgent-Online](https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-3053083.pt) | [VideoAgent-Suggestive](https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/metaworld/model-4307.pt)   

### iThor
[VideoAgent](https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/ithor/thor-402.pt)

### Bridge
[VideoAgent](https://huggingface.co/Trickyjustice/VideoAgent/resolve/main/bridge/model-44.pt)

Download and put the .pt file in `results/[environment]` folder. The resulting directory structure should be `results/{mw, thor, bridge}/model-[x].pt`, for example `results/mw/model-305.pt`

Or use `download.sh`
```bash
./download.sh metaworld
# ./download.sh ithor
# ./download.sh bridge
```

After this, you can use argument `-c [x]` to resume training or inference with our checkpoint. For example:  
```bash
python train_mw.py --mode train -c 305
```
Or  
```bash
python train_mw.py --mode inference -c 305 -p ../examples/assembly.png -t assembly
```

## Acknowledgements

This codebase is modified from the following repositories:  
[avdc](https://github.com/flow-diffusion/AVDC)
