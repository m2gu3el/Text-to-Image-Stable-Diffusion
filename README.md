
#   Text-to-Image Using Stable Diffusion


This project focuses on generating images from textual descriptions using AI techniques, specifically a stable diffusion model. The key components involved include a VAE encoder, U-Net, VAE decoder, CLIP encoder, and a DDPM(Denoising Diffusion Probabilistic Model) time scheduler.

 1. VAE Encoder: The Variational Autoencoder encoder maps input data into a latent space, allowing for the efficient encoding of complex information 
 while ensuring smooth interpolation between data points.

 2. U-Net: A convolutional neural network architecture adapted here to refine the latent representations and guide the diffusion process.

 3. CLIP(Contrastive Language–Image Pretraining) Encoder: CLIP bridges the gap between visual and textual data, providing a powerful embedding space 
 where images and text can be compared directly.

 4. Decoder: This component reconstructs the image from the processed latent space, ensuring that the final output maintains high fidelity to the 
 original input description.

 5. DDPM Time Scheduler: This scheduler introduces controlled noise to the latent representations, which is subsequently reduced through the denoising 
 process, facilitating the generation of coherent and high-quality images.
<img width="1437" alt="Screenshot 2024-06-19 at 7 55 15 PM" src="https://github.com/m2gu3el/Text-to-Image-Stable-Diffusion/assets/152903210/fc0e4f25-ef30-4b1f-9322-61d709f331ac">


## Dataset and Weights

Flickr30K_Dataset- https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Download vocab.json and merges.txt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer 

Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main 



### Clone repo

```
git clone https://github.com/m2gu3el/Text-to-Image-Stable-Diffusion.git
```

```
cd Text-to-Image-Stable-Diffusion
```
### Create environment

```
python -m venv venv
```

### Activate environment

Linux/macOS:
```
source venv/bin/activate
```

Windows: 
```
.\venv\Scripts\activate
```
### Install requirements

```
pip install -r requirements.txt
```
### Run the App

```
python app.py
```
## What is Stable Diffusion?
Stable Diffusion refers to a type of generative model used for creating high-quality images based on text prompts. It's part of the family of diffusion models, which are a type of deep generative model designed to generate data samples by iteratively denoising a noisy signal.

## Why Stable Diffusion?
Stable Diffusion has garnered significant attention and popularity in the field of generative AI for several reasons. Here are some key advantages and reasons why it stands out:

  1.High-Quality Image Generation:Stable Diffusion models are known for producing high-quality and detailed images. The iterative denoising process 
  helps in refining images to a high degree of visual fidelity.
  
  2.Flexibility and Diversity:These models can generate a wide variety of images from diverse text prompts. This versatility makes them useful for many 
  creative applications, from art generation to advertising and beyond.
  
  3.Stable Training Process: Diffusion models tend to have a more stable training process compared to other generative models like GANs (Generative 
  Adversarial Networks). This stability often results in better performance and more reliable outputs.
