# Evolved CTRL MLProject 5

## Table of Contents

1. [Group Composition](#group-composition)
2. [Obtain models](#obtain-models)
3. [Dataset](#dataset)
4. [Candidates](#candidates)
5. [Metrics](#metrics)
6. [EvolvedCTRL](#evolvedCTRL)
7. [Google Drive Connections](#gdrive-connection)

## Group Composition:

* Giovanni Tangredi s276086
* Yuri Sala s273339
* Luca Viscanti s277981

## Obtain models

For obtaining the pretrained model with only 10 layer follow the following steps:
First download the original model with 48 layers
```bash
gsutil -m cp -r gs://sf-ctrl/seqlen256_v1.ckpt .
```
then use the code in this github to convert the checkpoint
```bash
python convert_tf_to_huggingface_pytorch.py --tf_checkpoint tf_model --pytorch_checkpoint converted_model.bin --num_layers 10
```
For obtain any other trained models we have produce during the project you can [download them here](https://drive.google.com/drive/folders/14_t4bxw4y6m82FG_EpbP7aM9BLCqtnBd?usp=sharing).

## Dataset

All the dataset can be found already converted for use in the data_shuffle folder.
the structure of the sentences is: *Control Code* *Sentence*. Each sentence is divided bu a new line.

## Candidates

All the generated sentences used by us for calculate the metrics can be found in the candidates folder. The file reference.txt contains the set of sentences that we use to obtain the prompts.

## Metrics 
All the Metrics implementation can be found at Metrics.py.
If you want to recreate the results you can use the Metrics.ipynb file with all the files path changed as needed.
if you want to calculate more N-gram cumulative scores just change the weights, for example for the BLEU-5 set as weights (0.2,0.2,0.2,0.2,0.2).

## EvolvedCTRL
in new_classes.py you can found all the implementation needed for the Evolved Transformer.
For running everything in your local environment just use the EvolvedCTRL.ipynb file.
Remember to change the files path as needed, also this file contain anything needed for training of the model and also for generate sentences given a prompt from keyboard.

## Google Drive Connections
In some .ipynb file a cell with the connectio to a gdrive can be found, this cell can be ignored in case you  don't want to establish any connection with your drive , just remember that all the files needed for the code to work must be present in the folder you use.
