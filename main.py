from multiprocessing import freeze_support

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from torch.utils.throughput_benchmark import format_time
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import random
import numpy as np
import pandas as pd

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the English version of the STSB dataset
dataset = load_dataset("stsb_multi_mt", "ru")
# print(dataset)
# print("A sample from the STSB dataset's training split:")
# print(dataset['train'][98])

# You can use larger variants of the model, here we're using the base model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class STSBDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        # Normalize the similarity scores in the dataset
        similarity_scores = [i['similarity_score'] for i in dataset]
        self.normalized_similarity_scores = [i / 5.0 for i in similarity_scores]
        self.first_sentences = [i['sentence1'] for i in dataset]
        self.second_sentences = [i['sentence2'] for i in dataset]
        self.concatenated_sentences = [[str(x), str(y)] for x, y in zip(self.first_sentences, self.second_sentences)]

    def __len__(self):
        return len(self.concatenated_sentences)

    def get_batch_labels(self, idx):
        return torch.tensor(self.normalized_similarity_scores[idx])

    def get_batch_texts(self, idx):
        return tokenizer(self.concatenated_sentences[idx], padding='max_length', max_length=128, truncation=True,
                         return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def collate_fn(texts):
    input_ids = texts['input_ids']
    attention_masks = texts['attention_mask']
    features = [{'input_ids': input_id, 'attention_mask': attention_mask}
                for input_id, attention_mask in zip(input_ids, attention_masks)]
    return features


class BertForSTS(torch.nn.Module):

    def __init__(self):
        super(BertForSTS, self).__init__()
        self.bert = models.Transformer('bert-base-uncased', max_seq_length=128)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_bert(input_data)['sentence_embedding']
        return output


# Instantiate the model and move it to GPU
model = BertForSTS()
model.to(device)


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self, loss_fn=torch.nn.MSELoss(), transform_fn=torch.nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fn = loss_fn
        self.transform_fn = transform_fn
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, inputs, labels):
        emb_1 = torch.stack([inp[0] for inp in inputs])
        emb_2 = torch.stack([inp[1] for inp in inputs])
        outputs = self.transform_fn(self.cos_similarity(emb_1, emb_2))
        return self.loss_fn(outputs, labels.squeeze())


train_ds = STSBDataset(dataset['train'])
val_ds = STSBDataset(dataset['dev'])

# Create a 90-10 train-validation split.
train_size = len(train_ds)
val_size = len(val_ds)

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


def predict_similarity(sentence_pair):
    test_input = tokenizer(sentence_pair, padding='max_length', max_length=128, truncation=True,
                           return_tensors="pt").to(device)
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    del test_input['token_type_ids']
    output = model(test_input)
    sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()
    return sim


if __name__ == '__main__':
    freeze_support()

    batch_size = 8

    train_dataloader = DataLoader(
        train_ds,  # The training samples.
        num_workers=2,
        batch_size=batch_size,  # Use this batch size.
        shuffle=True  # Select samples randomly for each batch
    )

    validation_dataloader = DataLoader(
        val_ds,
        num_workers=2,
        batch_size=batch_size  # Use the same batch size
    )

    optimizer = AdamW(model.parameters(),
                      lr=1e-6)
    epochs = 8
    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)


    def train():
        seed_val = 42
        criterion = CosineSimilarityLoss()
        criterion = criterion.cuda()
        random.seed(seed_val)
        torch.manual_seed(seed_val)
        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(0, epochs):
            t0 = time.time()
            total_train_loss = 0
            model.train()
            # For each batch of training data...
            for train_data, train_label in tqdm(train_dataloader):
                train_data['input_ids'] = train_data['input_ids'].to(device)
                train_data['attention_mask'] = train_data['attention_mask'].to(device)
                train_data = collate_fn(train_data)
                model.zero_grad()
                output = [model(feature) for feature in train_data]
                loss = criterion(output, train_label.to(device))
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            t0 = time.time()
            model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            # Evaluate data for one epoch
            for val_data, val_label in tqdm(validation_dataloader):
                val_data['input_ids'] = val_data['input_ids'].to(device)
                val_data['attention_mask'] = val_data['attention_mask'].to(device)
                val_data = collate_fn(val_data)
                with torch.no_grad():
                    output = [model(feature) for feature in val_data]
                loss = criterion(output, val_label.to(device))
                total_eval_loss += loss.item()
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        return model, training_stats


    # Launch the training
    model, training_stats = train()

    # Create a DataFrame from our training statistics
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index
    df_stats = df_stats.set_index('epoch')

    print(df_stats)

    # load the test set
    test_dataset = load_dataset("stsb_multi_mt", name="en", split="test")

    # Prepare the data
    first_sent = [i['sentence1'] for i in test_dataset]
    second_sent = [i['sentence2'] for i in test_dataset]
    full_text = [[str(x), str(y)] for x, y in zip(first_sent, second_sent)]

    model.eval()

    PATH = 'bert-sts.pt'
    torch.save(model.state_dict(), PATH)
    example_1 = full_text[100]
    print(f"Sentence 1: {example_1[0]}")
    print(f"Sentence 2: {example_1[1]}")
    print(f"Predicted similarity score: {round(predict_similarity(example_1), 2)}")
