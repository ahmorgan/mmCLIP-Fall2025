## Requirements

All of the experiment results are acquired on a single A100 GPU.

conda create -n mmCLIP-ae python=3.8.17

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

pip install matplotlib

pip install transformers==4.36.0

pip install git+https://github.com/openai/CLIP.git

pip install timm==0.9.12

pip install opencv-python

pip install pandas

pip install scikit-learn

## Steps

While we strive to ensure reproducibility by setting a fixed seed on our server, some results still exhibit a variance of 3-5%.
To address this, we have included a log file for each experiment as a reference.


1. To reproduce zero-shot Tent baseline, python src/train_seen_unseen_tent.py
2. To pretrain model on synthetic dataset, python src/train_babel_gpt_v2.py. This step could take a long time, and the saved model file is too large, thus we prepared a saved epoch which can be directly used for the following experiments.
2. To reproduce Zero-shot(Syn+attr, Real+attr, mmCLIP) and few-shot(mmCLIP-one-shot) experiment, python src/train_seen_unseen.py