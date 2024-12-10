# Examining Downstream Bias Reemergence after Debiasing via Selective Parameter Updating

The core of the relevant code is contained in the PCGU-UnlearningBias folder, with code adapte from [the GitHub](https://github.com/CharlesYu2000/PCGU-UnlearningBias) for the PCGU debiasing method.

## Setup

First, initialize an environment from the _pcgu.yml_ file in the PCGU-UnlearningBias directory

## Datasets

The datasets used for fine-tuning are the following (all found on Hugging Face):
  - **BiasBios:** https://huggingface.co/datasets/LabHC/bias_in_bios
  - **WikiText:** https://huggingface.co/datasets/Salesforce/wikitext
  - **IMDb:** https://huggingface.co/datasets/Salesforce/wikitext

## Fine-tuning

The clean_wikitext_mlm_finetuning.ipynb and clean_biasbios_mlm_finetuning.ipynb notebooks in the root directory can be run to generate the fine-tuned models on WikiText and BiasBios.  
These notebooks should output a folder with subfolders containing the saved, fine-tuned model at each checkpoint.

## Additional fine-tuning

To evaluate after fine-tuning on additional datasets on Hugging Face (fine-tuning on a masked language modeling task), using the the finetune_mlm file. Change 1) the dataset loaded 
2) the column used in the tokenizer function (so it returns whichever column has the text) and 3) change the "remove_columns" when calling the tokenize function to define 
tokenized_datasets functions to include all other columns in the Hugging Face dataset. This should produce output in a similar format, giving a folder containing the model results at each checkpoint.

## Evaluation

Once fine-tuned (or on the original 'bert-base-uncased' model), we can attain StereoSet evaluation results as described in the PCGU-UnlearningBias ReadMe (the same as in the subdirectory).
