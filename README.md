# MM-MKP
This repository includes codes for the paper ''Towards Better Multi-modal Keyphrase Generation via Visual Entity Enhancement and Multi-granularity Image Noise Filtering".


## Installation
```
# Create environment
conda create -n CMKP  python==3.6

# Install pytorch 
conda install -n CMKP  -c pytorch pytorch==1.2 torchvision

# Other packages
pip install nltk h5py
```

## Data
For raw tweet image data, please find it from (https://github.com/yuewang-cuhk/CMKP).
For TRC dataset, please find it from ( https://github.com/danielpreotiuc/text-image-relationship/).
For the data we processed, please find it from (../data/tw_mm_s1_ocr/)

## Prepare image and text features
Please find it from `ext_feats`, where we provide codes to extract VGG or BUTD visual features (`infer_visual_feat`), image attribute features (`infer_attribute`), and OCR texts (`infer_OCR`) from the image, and extract Glove embedding (`prepare_tw_glove_emb.py`) and BERT features (`infer_bert`) from the text.


### Preprocessing
Before training, you need to run `python preprocess_type.py` to preprocess the raw text data and run `python preprocess_itm.py` to preprocess the TRC data. It will transform the raw texts into token ids and embeddings into `processed_data`.
For the data we processed, please find it from (https://drive.google.com/drive/folders/159Xy9oiBkE6zLwLmUudryXwXO3tjIOx0?usp=sharing)


### Training, predicting, and evaluating
Use the scripts (`bash train.sh`) to run the models. 


We provide our pretrained models in ([`best.ckpt`](https://drive.google.com/file/d/1yR0z-0Xc9F82-g-_8Z4STxnVk4kqIKs6/view?usp=sharing)https://drive.google.com/file/d/1yR0z-0Xc9F82-g-_8Z4STxnVk4kqIKs6/view?usp=sharing).

  
# License
This project is licensed under the terms of the MIT license. 
