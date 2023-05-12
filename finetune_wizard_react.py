import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

import json

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


# optimized for RTX 4090. for larger GPUs, increase some of these?

MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 2048  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 1000


dataset_list = []
# dir_ = "easy_task_mini_dataset_cleaned"
dir_ = "medium_size_generated_tasks"
files_ = os.listdir(dir_)
for f in files_:
    filename = os.path.join(dir_, f)
    print(filename)
    with open(filename, "r") as fp:
        txt = fp.read()
    prompt = txt.split("#####PROMPT:")[1].split("#####OUTPUT:")[0].strip()
    output = txt.split("#####OUTPUT:")[1].strip()
    dataset_list.append({
         "prompt":prompt,
         "output": output,
    })

with open("data.json", "w") as fp:
    json.dump(dataset_list, fp, indent=4)

# Wizard
model_path = "TheBloke/wizardLM-7B-HF"

model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(
    model_path,
    add_eos_token=True
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files="data.json")

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]


react_prompt_prelude = """
Received prompt:  Answer the following questions as best you can. You have access to the following tools:

Python REPL: A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.
Search: useful for when you need to ask with search

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Python REPL, Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

"""
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
        return react_prompt_prelude + data_point["prompt"] + data_point["output"] + "\n\nObservation:"


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        output_dir="lora-wizard",
        save_total_limit=3,
        load_best_model_at_end=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

trainer.train()

model.save_pretrained("lora-wizard-react")
