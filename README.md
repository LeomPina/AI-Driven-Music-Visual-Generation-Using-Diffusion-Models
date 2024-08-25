# Leveraging Diffusion Models for AI-Driven Music-Visual Generation

This project is meant to generate an AI music video using one (or more) artwork image and a music sample as inputs.

## 1. Generate images

This first section generates AI images using the Kandinsky 2.2 model by using an LLM music caption model, to generate captions for each 10 seconds of the audio sample, and an initial image artwork. 
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

* Install the image generation Kandinsky 2.2 model (more info about this model can be found at https://huggingface.co/kandinsky-community/kandinsky-2-2-prior and https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) and its dependencies

```python
!pip install diffusers
!pip install transformers scipy ftfy accelerate

from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline

pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
)
pipe_prior.to("cuda")

pipe = KandinskyV22Pipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.to("cuda")
```

## 2. Generate the music visualiser

This section generates an AI music visualiser using the Stable Diffusion Image Variations model to generate "image variations" capable of creating the transition frames between the AI images generated in the first stage of this project to produce the music video.
The audio sample previously chosen will be now used to guide the flow of the transitions (using the librosa package to extract musical elements such as the harmonic and percussive).
The "Generate_music_visualisers.ipynb" file presents this stage of the project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeomPina/The-use-of-music-for-the-generation-of-AI-visuals/blob/main/Generate_music_visualisers.ipynb)

### Installation Setup

* Install the image generation the Stable Diffusion Image Variations model (more info about this model can be found at https://huggingface.co/lambdalabs/sd-image-variations-diffusers) and its dependencies
  
```python
%%capture
! pip install diffusers transformers

from diffusers import StableDiffusionImageVariationPipeline

device = "cuda"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
)
sd_pipe = sd_pipe.to(device)
```

## 3. Music visualiser Evaluation

The evaluation of the outputted music video is performed by measuring the audio-visual synchrony/rhythm consistency between the music sample and the video.
The dynamic time warping (DTW) is measured between the peak values of the frequencies and abrupt pixel differences presented in the music sample and video, respectively.

### Installation Setup

```python
!pip install fastdtw
```

## 4. Results examples

This segment shows some example music visualisers that were generated by the developers. These examples are organised according to different music genres.

### Classical themed

[![Tchaikovsky Swan Lake 30sec AI visualiser](https://img.youtube.com/vi/8c93ZiLe1Rk/0.jpg)](https://youtube.com/shorts/8c93ZiLe1Rk?si=ic0DhHoKj5r-db_R)


link: https://youtube.com/shorts/8c93ZiLe1Rk?si=ic0DhHoKj5r-db_R

### Electronic themed

[![Disclosure Happening 30sec AI visualiser](https://img.youtube.com/vi/dPgbyScW6iw/0.jpg)](https://youtube.com/shorts/dPgbyScW6iw?si=69ggdgnjaZN6NkcQ)


link: https://youtube.com/shorts/dPgbyScW6iw?si=69ggdgnjaZN6NkcQ

### Country themed

[![Born and Raised John Mayer 30sec AI visualiser](https://img.youtube.com/vi/KBh8yegxHO8/0.jpg)](https://youtube.com/shorts/KBh8yegxHO8?si=uH5TEZNGRusSMS2Q)


link: https://youtube.com/shorts/KBh8yegxHO8?si=uH5TEZNGRusSMS2Q

### Rock themed

[![Roadhouse Blues The Doors 30sec AI visualiser](https://img.youtube.com/vi/izeLCRBKl8I/0.jpg)](https://youtube.com/shorts/izeLCRBKl8I?si=Y4kBEHpcOiq3DMSx)


link: https://youtube.com/shorts/izeLCRBKl8I?si=Y4kBEHpcOiq3DMSx

### Jazz themed

[![John Coltrane Giant Steps 30sec AI visualiser](https://img.youtube.com/vi/PvG95JKqiPA/0.jpg)](https://youtube.com/shorts/PvG95JKqiPA?si=0YQUZz89FYo5E4fa)


link: https://youtube.com/shorts/PvG95JKqiPA?si=0YQUZz89FYo5E4fa

## Reference

Please use this bibtex if you would like to cite it:

```
@misc{
      author = {LeomPina},
      title = {Leveraging Diffusion Models for AI-Driven Music-Visual Generation},
      year = {2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/LeomPina/Leveraging-Diffusion-Models-for-AI-Driven-Music-Visual-Generation.git}},
    }
```

## Project based on

* https://github.com/seungheondoh/lp-music-caps/tree/main
* https://github.com/alexjameswilliams/Music-Text-To-Image-Generation/tree/main
* https://aiart.dev/posts/sd-music-videos/sd_music_videos.html
