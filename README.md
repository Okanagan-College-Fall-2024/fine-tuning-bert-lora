# fine-tuning-bert-lora
Similar to the previous assignment, please create a notebook on Google Colab and paste the provided code into the notebook. If you have a GPU and CUDA installed on your laptop, you can create the notebook on your local system instead.


## Lab Overview

In this lab, you will fine-tune a pre-trained BERT model to perform sentiment analysis using Parameter-Efficient Fine-Tuning (PEFT) methods. Specifically, you’ll use Low-Rank Adaptation (LoRA) to adapt the BERT model for the sentiment analysis task. You’ll compare the performance of the BERT model before and after fine-tuning to observe the improvements achieved through PEFT.

**Objectives**

-	Understand how to use PEFT methods to fine-tune large language models efficiently.
-	Implement LoRA for fine-tuning BERT on a sentiment analysis dataset.
-	Evaluate and compare the model’s performance before and after fine-tuning.

**Instruction**
Install the required libraries using:
```bash
pip install torch transformers datasets peft scikit-learn

```
**Steps**

1. Setup the Environment

Ensure you have all the necessary libraries installed and import them:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```
2. Load and Prepare the Dataset

We’ll use the IMDB movie reviews dataset for binary sentiment classification.

```python
dataset = load_dataset('imdb')
```
Split the dataset into training and test sets:

```python
train_dataset = dataset['train']
test_dataset = dataset['test']
```

3. Tokenize the Data

Initialize the tokenizer:

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```
Define a tokenization function:

```python
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )
```
Apply the tokenizer to the datasets:

```python
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
```

**To Use GPU:**
```python
import torch

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import DataLoader

def collate_fn(batch):
    # Stack the inputs and move them to GPU
    input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
    attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
    labels = torch.tensor([item['label'] for item in batch]).to(device)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
```
```python
from torch.utils.data import DataLoader

# For the training dataset
train_loader = DataLoader(
    tokenized_train, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_fn
)

# For the test dataset
test_loader = DataLoader(
    tokenized_test, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=collate_fn
)
```

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
```
```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)

# Define training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Adjust as needed
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize the Trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
```

Set the format for PyTorch:

```python
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```
4. Evaluate the Pre-trained BERT Model (Before Fine-Tuning)

4.1 Load the Pre-trained Model

```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```
4.2 Define Evaluation Metrics

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
```
4.3 Setup Training Arguments

```python
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=16,
    logging_steps=500,
)
```
4.4 Initialize the Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)
```
4.5 Evaluate the Model

```python
eval_result = trainer.evaluate()
print("Pre-trained Model Evaluation:", eval_result)
```
5. Fine-Tune the BERT Model Using PEFT (LoRA)

5.1 Configure LoRA

Create a LoRA configuration:
```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence Classification
    inference_mode=False,
    r=8,  # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
)
```
5.2 Apply LoRA to the Model

Wrap the pre-trained model with PEFT:

```python
model = get_peft_model(model, peft_config)
```
5.3 Update Training Arguments for Fine-Tuning

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # For demonstration; adjust as needed
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```
5.4 Initialize the Trainer for Fine-Tuning

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)
```
5.5 Fine-Tune the Model

```python
trainer.train()
```
6. Evaluate the Fine-Tuned Model

After training, evaluate the fine-tuned model:

```python
eval_result = trainer.evaluate()
print("Fine-Tuned Model Evaluation:", eval_result)
```
7. Compare the Results

Compare the evaluation metrics before and after fine-tuning:

	•	Accuracy
	•	F1 Score
	•	Precision
	•	Recall

Example Output:

```text
Pre-trained Model Evaluation: {'eval_loss': 0.6931, 'eval_accuracy': 0.5, ...}
Fine-Tuned Model Evaluation: {'eval_loss': 0.2456, 'eval_accuracy': 0.90, ...}
```
