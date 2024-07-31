# COLUMBO

This is the source code for the paper `COLUMBO: Combating Obfuscated Payload with Generative Adversarial Training to Prevent Web Attacks Bypassing WAFs`'s model . This project is build on [DeBERTa-V3](https://github.com/microsoft/DeBERTa) and [GenerAT](https://github.com/Opdoop/GenerAT)  and has tested on `Ubuntu 20.04.5 LTS` with single GPU (A100 40GB).

### Prepare Environment and Datasets

1. Create environment and install requirement packages using provided `environment.yml`:

```
conda env create -f environment.yml
conda activate Columbo
```

2. Download pre-trained model
   * Download `pytorch_model.bin` and `pytorch_model.generator.bin` from [huggingface](https://huggingface.co/microsoft/deberta-v3-large/tree/main) and put it in `./deberta-v3-large` . 
3. Download data
   * Download [HPD dataset](https://github.com/Morzeux/HttpParamsDataset) and [SIK dataset](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset).

### Train

Run the following bash scripts, it will train the model on corresponding.

```
bash ./model/adv_glue/wad.sh
```

### Run attack
We prototype COLUMBO by combating three state-of-the-art adversarial attack methods, [WAF-A-MoLE](https://github.com/AvalZ/waf-a-mole), GPTfuzzer(https://github.com/hongliangliang/gptfuzzer) and [AdvSQLi](https://github.com/u21h2/AutoSpear).
1.WAF-A-MoLE
```
python ./attack/wafamole/mutate.py
```

2.GPTfuzzer
```
python ./attack/gptfuzzer/detect.py
```

3.AdvSQLi
```
python ./attack/advsqli/mutate/main.py
```
