# BLIP
This repository includes my experiments on Vision Language Understanding 

## Install Dependencies

```json
certifi==2022.6.15
charset-normalizer==2.1.1
fairscale==0.4.8
filelock==3.8.0
huggingface-hub==0.9.1
idna==3.3
numpy==1.23.2
packaging==21.3
Pillow==9.2.0
pyparsing==3.0.9
PyYAML==6.0
regex==2022.8.17
requests==2.28.1
timm==0.6.7
tokenizers==0.12.1
torch==1.12.1+cu116
torchvision==0.13.1+cu116
tqdm==4.64.0
transformers==4.21.2
typing_extensions==4.3.0
urllib3==1.26.12
```

Use `requirements.txt` to install dependencies

```sh
pip3 install -r requirements.txt
```

## Run Visual Question-Answering(VQA) Demo

```sh
python3 vqa.py <'sample_img_path'> <'Question?'>

# Sample: python3 vqa.py sample.jpg "What is the color of the horse?"
```

## Todo

- [x] Visual Question Answering
- [ ] Image Captioning
- [ ] Image-Text Matching using cosine similarity

## Citations

```txt
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}
```