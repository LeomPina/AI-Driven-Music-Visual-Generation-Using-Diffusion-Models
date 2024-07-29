# The use of music for the generation of AI visuals

This project is meant to generate an AI music video using one (or more) artwork image and a music sample as inputs.

## 1. Generate images

This section generates AI images using the Kadinsky 2.2 model by using an LLM music caption model, to generate captions for each 10 seconds of the audio sample, and an initial image artwork.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeomPina/The-use-of-music-for-the-generation-of-AI-visuals/blob/main/Generate_Images_artwork_LLM_music_caption.ipynb]

### 1.1. Installation Setup

* Install required packages by cloning the LLM music caption model GitHub repository (more info about this model can be found at https://github.com/seungheondoh/lp-music-caps/tree/main)
  
```python
!git clone https://github.com/seungheondoh/lp-music-caps.git
%cd lp-music-caps/
!pip install -e .

!pip install librosa # to get the most recent librosa package
```

* Download the pretrained model from Huggingface

```python
%cd lpmc/music_captioning/
!wget https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth -O exp/transfer/lp_music_caps/last.pth
```
