# COLUMBO

This is the source code for the paper `COLUMBO: Combating Obfuscated Payload with Generative Adversarial Training to Prevent Web Attacks Bypassing WAFs`'s model . This project is build on [DeBERTa-V3](https://github.com/microsoft/DeBERTa) and has tested on `Ubuntu 20.04.5 LTS` with single GPU (V100 32GB).

### Prepare Environment and Datasets

1. Create environment and install requirement packages using provided `environment.yml`:

```
conda env create -f environment.yml
conda activate Columbo
```

2. Download pre-trained model
   * Download `pytorch_model.bin` and `pytorch_model.generator.bin` from [huggingface](https://huggingface.co/microsoft/deberta-v3-large/tree/main) and put it in `./deberta-v3-large` . 
3. Download data
   * Download HPD dataset from link https://github.com/Morzeux/HttpParamsDataset and SIK dataset from link https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset.
```
python download_glue_data.py
```



### Train

Run the following bash scripts, it will train the model on corresponding.

```
bash ./adv_glue/wad.sh
```

