# The use of music for the generation of AI visuals

This project is meant to generate an AI music video using one (or more) artwork image and a music sample as inputs.

## 1. Generate images

This section generates AI images using the Kadinsky 2.2 model by using an LLM music caption model, to generate captions for each 10 seconds of the audio sample, and an initial image artwork. 
The "Generate_Images_artwork_LLM_music_caption.ipynb" file presents this stage of the project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeomPina/The-use-of-music-for-the-generation-of-AI-visuals/blob/main/Generate_Images_artwork_LLM_music_caption.ipynb)

### Installation Setup

* Install required packages by cloning the LLM music caption model GitHub repository (more info about this model can be found at https://github.com/seungheondoh/lp-music-caps/tree/main)
  
```python
!git clone https://github.com/seungheondoh/lp-music-caps.git
%cd lp-music-caps/
!pip install -e .

!pip install librosa # to get the most recent librosa package
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

This section generates AI images using the Kadinsky 2.2 model by using an LLM music caption model, to generate captions for each 10 seconds of the audio sample, and an initial image artwork. 
The "Generate_music_visualisers.ipynb" file presents this stage of the project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeomPina/The-use-of-music-for-the-generation-of-AI-visuals/blob/main/Generate_music_visualisers.ipynb)

## Project based on

* https://github.com/seungheondoh/lp-music-caps/tree/main
