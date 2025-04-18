# Sketch-to-Real Image CycleGAN

This repository contains an implementation of a CycleGAN model that transforms sketches to realistic images and vice versa. The project includes both the training code and a Flask web application for deploying the model.

## Overview

This project uses a Cycle-Consistent Generative Adversarial Network (CycleGAN) architecture to learn the mapping between two domains:
- Sketches (Domain X)
- Real images (Domain Y)

The implementation features:
- Two generators: G (Sketch→Real) and F (Real→Sketch)
- Two discriminators: D_X (for Real images) and D_Y (for Sketches)
- Cycle consistency loss to ensure that the transformations are invertible
- Identity mapping loss to preserve content

## Model Architecture

### Generator
- U-Net-like architecture with encoder-bottleneck-decoder structure
- Uses convolutional layers, batch normalization, and ReLU/LeakyReLU activations
- Tanh activation in the output layer

### Discriminator
- PatchGAN discriminator architecture
- Uses convolutional layers with batch normalization and LeakyReLU
- Sigmoid activation in the output layer

## Training Process

The model is trained using multiple loss components:
1. **Identity Loss**: Encourages the generator to preserve content when given an image from its target domain
2. **GAN Loss**: Makes the generated images look realistic
3. **Cycle Consistency Loss**: Ensures that transforming an image to the other domain and back results in the original image

## Pretrained model weights
│ ├── generator_G.pth # Sketch to Real generator
│ ├── generator_F.pth # Real to Sketch generator
│ ├── discriminator_D_X.pth # Real image discriminator
│ ├── discriminator_D_Y.pth # Sketch discriminator

The training script includes:
- DataLoader configuration for batching
- Loss function setup
- Training loop with regular progress updates
- Model checkpointing
- Loss visualization

## Usage

### Running Web App

```python
python app.py

After starting the server, navigate to http://localhost:5000 in your web browser.


The Flask application allows users to:

Upload sketches to generate realistic images
Upload real images to generate sketches
View the transformation results in real-time