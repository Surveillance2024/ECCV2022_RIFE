# Project Overview: ECCV2022-RIFE

This project implements "Real-Time Intermediate Flow Estimation for Video Frame Interpolation (RIFE)", a deep learning model designed to generate intermediate frames between two given images. It supports real-time performance and arbitrary-timestep interpolation.

**Key Features:**
*   Real-time video frame interpolation.
*   Supports arbitrary-timestep interpolation.
*   Optimized versions available for specific use cases (e.g., anime scenes, post-processing diffusion model generated videos).

**Main Technologies:**
*   Python
*   PyTorch (Deep Learning Framework)

## Building and Running

### Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:megvii-research/ECCV2022-RIFE.git
    cd ECCV2022-RIFE
    ```
2.  Install dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```
3.  Download pretrained HD models from [Google Drive](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing) and place them in the `train_log/` directory.

### Running Inference

**Video Frame Interpolation:**
```bash
# 2X interpolation
python3 inference_video.py --exp=1 --video=video.mp4

# 4X interpolation
python3 inference_video.py --exp=2 --video=video.mp4

# With scaling for high-resolution videos
python3 inference_video.py --exp=1 --video=video.mp4 --scale=0.5

# Read video from PNGs (e.g., input/0.png ... input/612.png)
python3 inference_video.py --exp=2 --img=input/

# Add slow-motion effect (audio will be removed)
python3 inference_video.py --exp=2 --video=video.mp4 --fps=60

# Montage original video and save PNG output
python3 inference_video.py --video=video.mp4 --montage --png
```

**Image Interpolation:**
```bash
# 16X interpolation (2^4) between img0.png and img1.png
python3 inference_img.py --img img0.png img1.png --exp=4
```

### Running with Docker

1.  Place pre-trained models in `train_log/*.pkl`.
2.  Build the container:
    ```bash
    docker build -t rife -f docker/Dockerfile .
    ```
3.  Run for video interpolation:
    ```bash
    docker run --rm -it -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
    ```
4.  Run for image interpolation:
    ```bash
    docker run --rm -it -v $PWD:/host rife:latest inference_img --img img0.png img1.png --exp=4
    ```
5.  Using GPU acceleration:
    ```bash
    docker run --rm -it --gpus all -v /dev/dri:/dev/dri -v $PWD:/host rife:latest inference_video --exp=1 --video=untitled.mp4 --output=untitled_rife.mp4
    ```

### Training

1.  Download the [Vimeo90K dataset](http://toflow.csail.mit.edu/).
2.  Run the training script (example for 4 GPUs):
    ```bash
    python3 -m torch.distributed.launch --nproc_per_node=4 train.py --world_size=4
    ```

### Evaluation

Commands for evaluating the model on various datasets are available in the `benchmark/` directory. Examples:
```bash
python3 benchmark/UCF101.py
python3 benchmark/Vimeo90K.py
python3 benchmark/MiddleBury_Other.py
python3 benchmark/HD.py
```

## Development Conventions

This project is primarily a research-oriented deep learning codebase. While no explicit coding style guides are provided, the codebase follows common Python and PyTorch practices. Contributions should aim to maintain consistency with existing code.

## Core Modules

### `InterpolatorInterface.py`

This file defines the `InterpolatorInterface` class, which acts as a wrapper for the RIFE model, facilitating video frame interpolation.

**`InterpolatorInterface` Class:**
*   **Purpose**: Loads a pre-trained RIFE model and provides a unified interface for performing frame interpolation.
*   **Initialization (`__init__`)**:
    *   Detects and utilizes CUDA (GPU) if available, otherwise falls back to CPU.
    *   Loads one of the RIFE model versions (HDv2, HDv3, HD, or ArXiv-RIFE) from the `train_log/` directory, attempting them in a specific fallback order.
    *   Sets the model to evaluation mode.
*   **`generate` Method**:
    *   **Functionality**: The primary method for interpolating frames between two input images.
    *   **Inputs (`imgs`)**: Accepts a tuple of two images, which can be file paths (strings), NumPy arrays, or raw image bytes.
    *   **`exp`**: An integer controlling the interpolation factor. If `exp=N`, it performs 2^N interpolation (e.g., `exp=4` for 16X interpolation).
    *   **`ratio`**: A float (0.0 to 1.0) for arbitrary-timestep interpolation, generating a single frame at a specific point between the two input images.
    *   **`rthreshold`, `rmaxcycles`**: Parameters for `ratio`-based interpolation, controlling precision and iteration limits.
    *   **`outputdir`**: An optional string specifying a directory to save the generated frames as PNG or EXR files. If `None`, the method returns a list of NumPy arrays.
    *   **Internal Logic**: Handles image loading, padding to model-compatible dimensions, and orchestrates the RIFE model's inference calls based on `exp` or `ratio` parameters.
