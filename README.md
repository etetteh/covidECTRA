# covidECTRA
Pretrained ELECTRA model for biomedical and covid text understanding

## Introduction
This is the implementation of covidECTRA (and coviBERT), a biomedical language understanding model pretrained mainly on [CORD-19](https://www.semanticscholar.org/cord19) metadata (abstracts only), and [PubMed](https://pubmed.ncbi.nlm.nih.gov/) abstracts using the recent self-supervised language model [ELECTRA](https://github.com/google-research/electra).

#### Some topics in the CORD-19 dataset literature:
COVID-19
![COVID-19](https://github.com/etetteh/covidECTRA/blob/main/CORD-19%20Topic%20Examples/coro.png)

Common symptoms
![Common symptoms](https://github.com/etetteh/covidECTRA/blob/main/CORD-19%20Topic%20Examples/coro_most_com_symptom.png)

Prevention
![Prevention](https://github.com/etetteh/covidECTRA/blob/main/CORD-19%20Topic%20Examples/coro_prev_mask.png)


## Requirement
* Python 3
* TensorFlow 1.15.4
* NumPy
* scikit-learn and SciPy (for computing some evaluation metrics).

## Pretraining
###
We created a 50K uncased custom WordPiece vocabulary file for all our implementation.

### Create Whole-Word Masking (WWM) pretaining data
The following code (adapted from [BERT](https://github.com/google-research/bert)) creates the whole-word masking pretraining data. The code below creates the data for the small-model. To create the data for the base-model, change the values of `max_seq_length` to 512, and `max_predictions_per_seq` to 79.  
```
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$covidECTRA_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --do_whole_word_mask=True \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
 ```
### Create Wordpiece pretaining data
The following code (adapted from [ELECTRA](https://github.com/google-research/electra)) creates pretraining data for the small, and base models.

Small model
```
python electra_small/build_pretraining_dataset.py --corpus-dir $covidECTRA_DIR --vocab-file $covidECTRA_DIR/vocab.txt --output-dir pretrain_small --max-seq-length 128 --blanks-separate-docs False --do-lower-case
```

Base model
```
python electra_base/build_pretraining_dataset.py --corpus-dir $covidECTRA_DIR --vocab-file $covidECTRA_DIR/vocab.txt --output-dir pretrain_base --max-seq-length 512 --blanks-separate-docs False --do-lower-case
```
### Run pretraining
We pretrain three models (two small models and one base model), each on the WWM and non-WWM pretrained data. We follow the original [ELECTRA](https://github.com/google-research/electra) implementation

####### covidECTRA-BASE #######

`python3 electra_base/run_pretraining.py --data-dir gs://covidectrap --model-name "covidECTRA-Base"`
`python3 electra_base/run_pretraining.py --data-dir gs://covidectrap --model-name "covidECTRA-Base-WWM"`

####### covidECTRA-Small #######

`python3 electra_small/run_pretraining.py --data-dir gs://covidectrap --model-name "covidECTRA-Small"`
`python electra_small/run_pretraining.py --data-dir gs://covidectrap --model-name "covidECTRA-Small-WWM"`

####### covibert #######

`python3 electra_small/run_pretraining.py --data-dir gs://covidectrap --model-name "coviBERT"`
`python3 electra_small/run_pretraining.py --data-dir gs://covidectrap --model-name "coviBERT-WWM"`


## Fine-tuning
We fine-tune on three Named Entity Recognition (NER) datasets, and three Relation Extraction (RE) datasets. The NER datasets are BC5CDR-chemical, BC5CDR-disease and NCBI-disease, while the RE datasets are GAD, DDI, and Chemprot. For the NER fine-tuning, we adapted the code of [BioBERT](https://github.com/dmis-lab/biobert), while for the RE fine-tuning, we adapted the code of [ELECTRA](https://github.com/google-research/electra) for [sequence classification](https://github.com/google-research/electra/tree/master/finetune/classification). 


### NER
The following code shows an example of running inference on a NER task:
Example to fine-tune on BC5CDR-chemical, run

```
python biobert/electra_run_ner.py --do_train=true --do_eval=true --do_predict=true --max_seq_length=384 --vocab_file=vocab.txt --bert_config_file=coviBERT-Baseline/covibert_small_config.json --init_checkpoint=coviBERT-Baseline/model.ckpt-1000000 --num_train_epochs=2.0 --learning rate=1e-5 --data_dir=finetuning_data/bert_bc5c --output_dir=coviBERT-Baseline/results/bc5c
```

The code below converts the token-level output results from the code above, to entity-level (the official evaluation format expected)

```
python biobert/biocodes/ner_detokenize.py --token_test_path=coviBERT-Baseline/results/bc5c/token_test.txt --label_test_path=coviBERT-Baseline/results/bc5c/label_test.txt --answer_path=finetuning_data/bert_bc5c/test.tsv --output_dir=coviBERT-Baseline/results/bc5c
```

The code below executes the entity-level exact match evaluation.

```
perl biobert/biocodes/conlleval.pl < coviBERT-Baseline/results/bc5c/NER_result_conll.txt
```


### RE
The following code shows an example of running inference on a RE task:
Example to fine-tune on DDI, run:

```
python electra_small/run_finetuning.py --data-dir "gs://covidectrap" --model-name "covidECTRA-Small" --hparams '{"model_size": "small", "task_names": ["ddi"]}'
```

### Results
#### Named Entity Recognition Results ( F1-score entity-level)
| Model | BC5-Chemical | BC5-Disease | NCBI-Disease |
| :---- | :----------: | :---------: | :----------: |
|BioBERT      | 93.41 | 85.31 | 89.47 |
|SciBERT      | 93.12 | 85.34 | 88.61 |
|ClinicalBERT | 91.51 | 84.18 | 87.31 |
|BlueBERT     | 91.98 | 84.63 | 89.20 |
|PubMedBERT   | 94.06 | 86.63 | 88.81 |
|**coviBERT-Baseline(ours)** | 88.69 | 79.63 | 81.05 |
|**covidECTRA-Small(ours)** | 88.10 | 78.46 | 78.28 |
|**covidECTRA-Base(ours)** | 91.95 | 81.18 | 85.18 |

#### Relation Extraction Results (F1-score)
| Model | GAD   | DDI   | ChemProt | 
| :---- | :---: | :---: | :------: |
|BioBERT      | 80.94 | 80.88 | 76.14 | 
|SciBERT      | 80.90 | 81.06 | 75.24 | 
|ClinicalBERT | 78.40 | 78.20 | 72.04 | 
|BlueBERT     | 77.24 | 77.78 | 71.46 | 
|PubMedBERT   | 82.34 | 82.36 | 77.24 |
|**coviBERT-Baseline(ours)** | 76.99 | 83.90 | 79.53 |
|**covidECTRA-Small(ours)**  | 79.01 | 85.69 | 81.09 |
|**covidECTRA-Base(ours)**   | 80.60 | 86.65 | 83.15 |

## Acknowledgement

A big thank you to the [BlueBERT](https://github.com/ncbi-nlp/bluebert) team for making their preprocessed data available. Another big thank you to the CORD-19 Dataset Research Team.
Finally, a huge thank you to the [Google TensorFlow Research Cloud (TFRC)](https://sites.research.google/trc/) team for supporting this work with Cloud TPUs.
