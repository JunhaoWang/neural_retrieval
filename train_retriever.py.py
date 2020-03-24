import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import Namespace
from transformers.data.processors.glue import MnliProcessor
import torch
from transformers import (
    BertModel,
    BertTokenizer,
    BertPreTrainedModel,
    BertConfig
)
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

## Berts
model_str = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_str)
bert_question = BertModel.from_pretrained(model_str)
bert_paragraph = BertModel.from_pretrained(model_str)

# ## Dataloaders
#
# def process_natq_clean(file_path):
#     ## if not exist save processed file
#     pass
#
#
# def generate_natq_clean_dataloaders():
#     ## convert things to data loaders
#     pass
#
#
# train_dataloader, dev_dataloader = generate_natq_clean_dataloaders()


## nn.Module classes

class BertEncoder(nn.Module):

    def __init__(self, bert, max_seq_len= 30, emb_dim = 768):
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
    def __init__(self, bert_question_encoder, bert_paragraph_encoder, emb_dim=768):
        super(Retriver, self).__init__()
        self.emb_dim = emb_dim
        self.bert_question_encoder = bert_question_encoder
        self.bert_paragraph_encoder = bert_paragraph_encoder

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                         batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                         batch_token_type_ids_paragraphs):
        h_question = self.bert_question_encoder(input_ids=input_ids_question, attention_mask=attention_mask_question,
                                        token_type_ids=token_type_ids_question)
        h_paragraphs_batch = self.bert_paragraph_encoder(input_ids=batch_input_ids_paragraphs,
                                                 attention_mask=batch_attention_mask_paragraphs,
                                                 token_type_ids=batch_token_type_ids_paragraphs)
        return h_question, h_paragraphs_batch


## Todo: torch lighting to train
# class RetriverTrainer(pl.LightningModule):
#
#     def __init__(self, bert_question_encoder, bert_paragraph_encoder, emb_dim=768):
#
#
#     def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
#
#
#     def training_step(self, batch, batch_nb):
#
#
#     def validation_step(self, batch, batch_nb):
#
#
#     def validation_epoch_end(self, outputs):
#
#
#     def test_step(self, batch, batch_nb):
#
#
#     def test_epoch_end(self, outputs):
#
#
#     def configure_optimizers(self):
#
#
#     def train_dataloader(self):
#
#
#     def val_dataloader(self):
#


if __name__ == '__main__':
    ## example
    txt = 'I am a brat'
    encoder_question = BertEncoder(bert_question, 30)
    encoder_paragarph = BertEncoder(bert_paragraph, 512)
    h = encoder_question(**tokenizer.encode_plus(txt, add_special_tokens=True,
        max_length=encoder_question.max_seq_len, pad_to_max_length=True,
    return_tensors='pt'))

    input_question = tokenizer.encode_plus(txt, add_special_tokens=True,
        max_length=encoder_question.max_seq_len, pad_to_max_length=True,
    return_tensors='pt')
    inputs_paragraph = tokenizer.batch_encode_plus(['I am horny', 'I am sleepy'],
                                         add_special_tokens=True,
                                         pad_to_max_length=True,
                                         max_length=30,
                                         return_tensors='pt'
                                         )

    inputs = {
        'input_ids_question': input_question['input_ids'],
        'attention_mask_question': input_question['attention_mask'],
        'token_type_ids_question': input_question['token_type_ids'],
        'batch_input_ids_paragraphs': inputs_paragraph['input_ids'],
        'batch_attention_mask_paragraphs': inputs_paragraph['attention_mask'],
        'batch_token_type_ids_paragraphs': inputs_paragraph['token_type_ids']
    }
    ret = Retriver(encoder_question, encoder_paragarph)
    h_q, h_ps = ret(**inputs)
    print(h.shape, h_q.shape, h_ps.shape)


