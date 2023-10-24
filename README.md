# DPO pipeline for finetuning LlaMa 2

Special thanks to Hugging face for helpful code and examples. See [here](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts) and [here](https://huggingface.co/blog/dpo-trl). 

## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```

Since we will use `accelerate` for training, make sure to run:
```
$ accelerate config
```

## Training

There were three main steps to the experiment:
1. Supervised fine-tuning of the base llama-v2 model. See pre_training_script.py. This makes the language model "on-policy".
2. DPO fine-tuning. See training_script.py. This finetunes the model to align with the provided preferences.
3. Testing. See either testing_script.py to run one question through the model, or checkpoints_to_csv.py to run many questions through many models.

See **commands.txt** for the command line arguments used for these experiments. 
