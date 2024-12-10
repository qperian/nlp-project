import transformers
from transformers import EarlyStoppingCallback

print(transformers.__version__)


##########


from transformers.utils import send_example_telemetry

send_example_telemetry("language_modeling_notebook", framework="pytorch")

############

from datasets import load_dataset
social_frames=load_dataset("allenai/social_bias_frames")

##########

model_checkpoint = "bert-base-uncased"#/home/perian/nlp-project/PCGU-UnlearningBias/src/models/model_9"

#########

def tokenize_function(examples):
    result = tokenizer(examples["post"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# block_size = tokenizer.model_max_length
block_size = 12

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

########

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"]) # for wikitext
# tokenized_datasets = imdb_dataset.map(
#     tokenize_function, batched=True, remove_columns=["text", "label"]
# )

tokenized_datasets = social_frames.map(
    tokenize_function, batched=True, remove_columns=["whoTarget","intentYN","sexYN","sexReason","offensiveYN","annotatorGender","annotatorMinority",
                                                     "sexPhrase","speakerMinorityYN","WorkerId","HITId","annotatorPolitics","annotatorRace","annotatorAge",
                                                     "post","targetMinority","targetCategory","targetStereotype","dataSource"])
print(tokenized_datasets)
###########

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

#######

from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

######

from transformers import Trainer, TrainingArguments

model_name = model_checkpoint.split("/")[-1]

batch_size = 64
# Show the training loss with every epoch
# logging_steps = len(downsampled_dataset["train"]) // batch_size
# model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-block12-batch8-socialframes-mlm",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end = True,
    save_steps=500
)

##########

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

########

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

#######

print("about to train!!")
trainer.train()

######

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

#######

