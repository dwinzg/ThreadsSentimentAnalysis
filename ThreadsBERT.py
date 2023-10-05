"""
Continuing on from the Sentiment Analysis using XGBoost, this portion will feature PyTorch
An API will also be included as to help users test strings for their own sentiment analysis
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

"""
Hyperparameters will be defined here as BERT will take a while to process the entire dataset, I will change it where
the computational time is less at the expense of a less accurate model 
"""
batch_size = 32
max_length = 128
learning_rate = 2e-5
num_epochs = 2
warmup_proportion = 0.1

"""
Loading dataset, preprocessing, and splits from the previous Sentiment Analysis
"""
# Loading dataset
df = pd.read_csv('data/threads_review.csv')

# To account for a positive or negative rating, [0,1,2,3] as negative, [4,5] as positive
df['sentiment'] = df['rating'].apply(lambda x: 1 if x in [4, 5] else 0)

# Splitting dataset, train: 80%, test 20%
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encoding tokenize data
train_texts = train_df['review_description'].tolist()
test_texts = test_df['review_description'].tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(train_df['sentiment'].tolist())
test_labels = torch.tensor(test_df['sentiment'].tolist())

# Create data loaders
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 classes

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(warmup_proportion * total_steps),
    num_training_steps=total_steps
)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Testing
model.eval()

# Testing
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.tolist())
        true_labels.extend(labels.tolist())

test_accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Saving model
model.save_pretrained('fine_tuned_bert_model')
tokenizer.save_pretrained('fine_tuned_bert_model')