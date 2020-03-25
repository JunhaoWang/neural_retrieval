import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hashlib
import torch
from transformers import (
    BertModel,
    BertTokenizer,
)
from typing import List
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

## Berts
model_str = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_str)
bert_question = BertModel.from_pretrained(model_str)
bert_paragraph = BertModel.from_pretrained(model_str)

## Dataloaders
num_dat = 10
batch_size = 2
max_question_len_global = 30
max_paragraph_len_global = 30
default_bert_emb_dim_global = 768

def process_natq_clean(file_path):
    ## if not exist save processed file
    pass


def generate_natq_clean_dataloaders():
    ## convert things to data loaders
    txt = 'I am a question'
    input_question = tokenizer.encode_plus(txt, add_special_tokens=True,
                                           max_length=max_question_len_global, pad_to_max_length=True,
                                           return_tensors='pt')
    inputs_paragraph = tokenizer.batch_encode_plus(['I am positve' * 3, 'I am negative' * 4, 'I am negative'],
                                                   add_special_tokens=True,
                                                   pad_to_max_length=True,
                                                   max_length=max_paragraph_len_global,
                                                   return_tensors='pt'
                                                   )
    dataset = TensorDataset(
        input_question['input_ids'].repeat(num_dat, 1),
        input_question['attention_mask'].repeat(num_dat, 1),
        input_question['token_type_ids'].repeat(num_dat, 1),
        inputs_paragraph['input_ids'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['attention_mask'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['token_type_ids'].unsqueeze(0).repeat(num_dat, 1, 1)
    )

    dataset_dev = TensorDataset(
        input_question['input_ids'].repeat(num_dat, 1),
        input_question['attention_mask'].repeat(num_dat, 1),
        input_question['token_type_ids'].repeat(num_dat, 1),
        inputs_paragraph['input_ids'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['attention_mask'].unsqueeze(0).repeat(num_dat, 1, 1),
        inputs_paragraph['token_type_ids'].unsqueeze(0).repeat(num_dat, 1, 1)
    )

    return DataLoader(dataset, batch_size=batch_size), DataLoader(dataset_dev, batch_size=batch_size)


train_dataloader, dev_dataloader = generate_natq_clean_dataloaders()

## nn.Module classes

class BertEncoder(nn.Module):

    def __init__(self, bert, max_seq_len, emb_dim = default_bert_emb_dim_global):
        super(BertEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.bert = bert
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _ = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        h_cls = h[:, 0]
        h_transformed = self.net(h_cls)
        return F.normalize(h_transformed)

class Retriver(nn.Module):
    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len=max_question_len_global, max_paragraph_len=max_paragraph_len_global,
                 emb_dim=default_bert_emb_dim_global):
        super(Retriver, self).__init__()
        self.bert_question_encoder = bert_question_encoder
        self.bert_paragraph_encoder = bert_paragraph_encoder
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_paragraph_len = max_paragraph_len
        self.emb_dim = emb_dim
        self.cache_hash2str = {}
        self.cache_hash2array = {}

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                         batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                         batch_token_type_ids_paragraphs):

        batch_size, num_document, max_len_size = batch_input_ids_paragraphs.size()

        h_question = self.bert_question_encoder(input_ids=input_ids_question, attention_mask=attention_mask_question,
                                        token_type_ids=token_type_ids_question)

        batch_input_ids_paragraphs_reshape = batch_input_ids_paragraphs.reshape(-1, max_len_size)
        batch_attention_mask_paragraphs_reshape = batch_attention_mask_paragraphs.reshape(-1, max_len_size)
        batch_token_type_ids_paragraphs_reshape = batch_token_type_ids_paragraphs.reshape(-1, max_len_size)

        h_paragraphs_batch_reshape = self.bert_paragraph_encoder(input_ids=batch_input_ids_paragraphs_reshape,
                                                 attention_mask=batch_attention_mask_paragraphs_reshape,
                                                 token_type_ids=batch_token_type_ids_paragraphs_reshape)
        h_paragraphs_batch = h_paragraphs_batch_reshape.reshape(batch_size, num_document, -1)
        return h_question, h_paragraphs_batch

    def str2hash(drlf, str):
        return hashlib.sha224(str.encode('utf-8')).hexdigest()

    def predict(self, question_str: str, batch_paragraph_strs: List[str]):
        self.eval()
        with torch.no_grad():
            ## Todo: embed all unique docs, then create ranking for all questions, then find overlap with constrained ranking
            batch_paragraph_array = np.random.random((len(batch_paragraph_strs), self.emb_dim))
            hashes = {}
            uncached_paragraphs = []
            uncached_hashes = []
            for ind, i in enumerate(batch_paragraph_strs):
                hash = self.str2hash(i)
                hashes[hash] = ind
                if hash in self.cache_hash2array:
                    batch_paragraph_array[ind,:] = deepcopy(self.cache_hash2array[hash])
                else:
                    uncached_paragraphs.append(i)
                    uncached_hashes.append(hash)
                    self.cache_hash2str[hash] = i
            uncached_paragraph_array = self.bert_paragraph_encoder(
                **self.tokenizer.batch_encode_plus(uncached_paragraphs,
                     add_special_tokens=True,
                     pad_to_max_length=True,
                     max_length=self.max_paragraph_len,
                     return_tensors='pt'
                 )
            ).detach().numpy()
            for ind, i in enumerate(uncached_paragraph_array):
                self.cache_hash2array[uncached_hashes[ind]] = deepcopy(i)
                batch_paragraph_array[ind,:] = deepcopy(i)

            question_array = self.bert_question_encoder(
                **self.tokenizer.encode_plus(question_str, add_special_tokens=True,
                                               max_length=max_question_len_global, pad_to_max_length=True,
                                               return_tensors='pt'))
            relevance_scores = torch.sigmoid(
                torch.mm(torch.tensor(batch_paragraph_array, dtype=question_array.dtype), question_array.T)).reshape(-1)
            rerank_index = torch.argsort(-relevance_scores)
            relevance_scores_numpy = relevance_scores.detach().numpy()
            rerank_index_numpy = rerank_index.detach().numpy()
            reranked_paragraphs = [batch_paragraph_strs[i] for i in rerank_index_numpy]
            reranked_relevance_scores = relevance_scores_numpy[rerank_index]
            return reranked_paragraphs, reranked_relevance_scores, rerank_index_numpy

class RetriverTrainer(pl.LightningModule):

    def __init__(self, retriever, emb_dim=default_bert_emb_dim_global):
        super(RetriverTrainer, self).__init__()
        self.retriever = retriever
        self.emb_dim = emb_dim

    def forward(self, **kwargs):
        return self.retriever(**kwargs)

    def step_helper(self, batch):
        input_ids_question, attention_mask_question, token_type_ids_question, \
        batch_input_ids_paragraphs, batch_attention_mask_paragraphs, \
        batch_token_type_ids_paragraphs = batch

        inputs = {
            'input_ids_question': input_ids_question,
            'attention_mask_question': attention_mask_question,
            'token_type_ids_question': token_type_ids_question,
            'batch_input_ids_paragraphs': batch_input_ids_paragraphs,
            'batch_attention_mask_paragraphs': batch_attention_mask_paragraphs,
            'batch_token_type_ids_paragraphs': batch_token_type_ids_paragraphs
        }

        h_question, h_paragraphs_batch = self(**inputs)
        batch_size, num_document, emb_dim = batch_input_ids_paragraphs.size()

        all_dots = torch.bmm(h_question.repeat(3, 1).unsqueeze(1),
            h_paragraphs_batch.reshape(-1, 768).unsqueeze(2)).reshape(batch_size, num_document)
        all_prob = torch.sigmoid(all_dots)

        pos_loss = - torch.log(all_prob[:, 0]).sum()
        neg_loss = - torch.log(1 - all_prob[:, 1:]).sum()
        loss = pos_loss + neg_loss
        return loss, all_prob

    def training_step(self, batch, batch_idx):
        """
        batch comes in the order of question, 1 positive paragraph,
        K negative paragraphs
        """

        train_loss, _ = self.step_helper(batch)
        # logs
        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        loss, all_prob = self.step_helper(batch)
        batch_size = all_prob.size()[0]
        _, y_hat = torch.max(all_prob, 1)
        y_true = torch.zeros(batch_size, dtype=y_hat.dtype).type_as(y_hat)
        val_acc = torch.tensor(accuracy_score(y_true.cpu(), y_hat.cpu())).type_as(y_hat)
        return {'val_loss': loss, 'val_acc': val_acc}


    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).sum() / len(outputs)
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).sum() / len(outputs)


        tqdm_dict = {'val_acc': avg_val_acc, 'val_loss': avg_val_loss}

        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'val_acc': avg_val_acc, 'val_loss': avg_val_loss}
        }
        return results


    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad])

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return dev_dataloader

if __name__ == '__main__':
    encoder_question = BertEncoder(bert_question, max_question_len_global)
    encoder_paragarph = BertEncoder(bert_paragraph, max_paragraph_len_global)
    ret = Retriver(encoder_question, encoder_paragarph, tokenizer)

    checkpoint_callback = ModelCheckpoint(
        filepath='out/{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    early_stopping = EarlyStopping('val_acc', mode='max')

    trainer = pl.Trainer(
        gpus=1,
        val_check_interval=0.1,
        min_epochs=1, max_epochs=10,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping)


    ret_trainee = RetriverTrainer(ret)

    trainer.fit(ret_trainee)

    # reranked_paragraphs, reranked_relevance_scores, rerank_index = ret.predict('I am beautiful lady?', ['You are a pretty girl',
    #                                                                                                'apple is tasty',
    #                                                                                                'He is a handsome boy'])
    # print(reranked_paragraphs, reranked_relevance_scores, rerank_index)