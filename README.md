# The use of music for the generation of AI visuals

This project is meant to generate an AI music video using one (or more) artwork image and a music sample as inputs.

## 1. Generate images

This first section generates AI images using the Kadinsky 2.2 model by using an LLM music caption model, to generate captions for each 10 seconds of the audio sample, and an initial image artwork. 
The "Generate_Images_artwork_LLM_music_caption.ipynb" file presents this stage of the project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeomPina/The-use-of-music-for-the-generation-of-AI-visuals/blob/main/Generate_Images_artwork_LLM_music_caption.ipynb)

### Installation Setup

* Install required packages by cloning the LLM music caption model GitHub repository (more info about this model can be found at https://github.com/seungheondoh/lp-music-caps/tree/main)
  
```python
!git clone https://github.com/seungheondoh/lp-music-caps.git
%cd lp-music-caps/
!pip install -e .
```

* Download and run the pre-trained model from Huggingface to generate captions

```python
%cd lpmc/music_captioning/
!wget https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth -O exp/transfer/lp_music_caps/last.pth
!python3 captioning.py --gpu 0 --audio_path 'song_path' > .../music_desc.txt
```

* Install the image generation Kadinsky 2.2 model (more info about this model can be found at https://huggingface.co/kandinsky-community/kandinsky-2-2-prior) and its dependencies

```python
!pip install diffusers
!pip install transformers scipy ftfy accelerate

from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from diffusers.utils import load_image, make_image_grid
import PIL
import torch
from torchvision import transforms

pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
)
pipe_prior.to("cuda")
```

## 2. Generate the music visualiser

This section generates an AI music visualiser using the Stable Diffusion Image Variations model to generate "image variations" capable of creating the transition frames between the AI images generated in the first stage of this project to produce the music video.
The audio sample previously chosen will be now used to guide the flow of the transitions (using the librosa package to extract musical elements such as the harmonic and percussive).
The "Generate_music_visualisers.ipynb" file presents this stage of the project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeomPina/The-use-of-music-for-the-generation-of-AI-visuals/blob/main/Generate_music_visualisers.ipynb)

### Installation Setup

* Install required packages by cloning the LLM music caption model GitHub repository (more info about this model can be found at https://huggingface.co/lambdalabs/sd-image-variations-diffusers) and its dependencies
  
```python
%%capture
! pip install diffusers transformers

import torch
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from pathlib import Path

# Device setup
device = "cuda"

output_dir = Path('images_walk_with_audio') # change output path
output_dir.mkdir(exist_ok=True, parents=True)

# Load the pre-trained model
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
)
sd_pipe = sd_pipe.to(device)
```

## 2. Music visualiser Evaluation

The evaluation of the outputted music video is performed by measuring the audio-visual synchrony/rhythm consistency between the music sample and the video.
The dynamic time warping (DTW) is measured between the peak values of the frequencies and abrupt pixel differences presented in the music sample and video, respectively.

### Installation Setup

```python
!pip install fastdtw
```

## Project based on

* https://github.com/seungheondoh/lp-music-caps/tree/main
* https://github.com/alexjameswilliams/Music-Text-To-Image-Generation/tree/main
* https://aiart.dev/posts/sd-music-videos/sd_music_videos.html
