# BERT-JRE

The final project of EMNLP course in PKU, 2021 Spring.

### Abstract

Entity and relation extraction aims to identify named entities from plain texts and relations among them. These two task are typically modeled jointly with a neural network in an end-to-end manner. Recent work [11] models the two tasks separately, and propose a pipeline where the entity information is fused to relation extractor at the input layer. Surprisingly, that model outperforms a lot of strong joint models. In this work, we follow their separate modeling style, and propose an entity-specic encoding with multi-head attention mechanism. The model (BERT-JRE) reaches results comparable with input-level fusion in [11] on relation extraction, but are much more time-efficient.

### Model

Please refer to the report

### Experiment

To run the code, please check `run.sh` and install required packages before you run this script. 