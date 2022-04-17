# SIGNATE-医学論文の自動仕分けチャレンジ
SIGNATEの「医学論文の自動仕分けチャレンジ」

## Dependency
todo

## Setup
```bash
cd data
signate download -c 595
```

## Usage
### Argument
### fold
default=10, type=int
### model_name
default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"\
type=str
### epochs
default=5, type=int
### data_dir
default="data", type=str
### output_dir
default="outputs", type=str
### seed
default=472, type=int

## References
https://signate.jp/competitions/595