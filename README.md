---
title: Face Editor
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🎨 Face Attribute Editor

A Gradio web app for editing facial attributes using a Variational Autoencoder (VAE) trained on the CelebA dataset.

Upload a face photo and use the sliders to manipulate attributes like smiling, eyeglasses, hair darkness, age, and more — all by navigating the VAE's latent space.

---

## How It Works

1. **Encode** — your photo is mapped to a point in a 400-dimensional latent space
2. **Shift** — each slider nudges that point along a learned attribute direction
3. **Decode** — the decoder renders the shifted point back into a face image

---

## Requirements

- Docker
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## Running the App

**1. Clone the repository**
```bash
git clone https://github.com/YuvalKoren07/face-transformation-vae
cd face-transformation-vae
```

**2. Build and start the container**
```bash
docker compose -f docker-compose.gpu.yml up --build
```

**3. Open in browser**
```
http://localhost:7860
```

---

## Attributes

| Slider | Effect |
|---|---|
| Eyeglasses | Add or remove glasses |
| Smiling | Add or remove a smile |
| Young | Age the face up or down |
| Bald | Add or remove hair |
| Pale Skin | Lighten or darken skin tone |
| Hair Darkness | Lighter to darker hair |
| Bangs | Add or remove bangs |
| Gender | Shift toward masculine or feminine features |
| Bushy Eyebrows | Thicken or thin the eyebrows |

---

## Model

The VAE was trained on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (200k face images) with the following parameters:

| Parameter | Value |
|---|---|
| Image size | 32×32 |
| Latent dimensions | 400 |
| Beta | 20,000 |
| Epochs | 40 |

---

## Project Structure

```
├── gui_app.py              # Gradio web app
├── utils.py                # Helper functions
├── attribute_vectors.pkl   # Precomputed attribute directions in latent space
├── models/
│   ├── encoder/            # Trained encoder model
│   └── decoder/            # Trained decoder model
├── docker-compose.gpu.yml
└── requirements.txt
```

---

## Credits

VAE architecture and training framework adapted from *Generative Deep Learning, 2nd Edition* by David Foster (O'Reilly). Gradio frontend, attribute vector computation, and deployment are original work built on top of that foundation.
