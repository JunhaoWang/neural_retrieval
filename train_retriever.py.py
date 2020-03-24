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
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup
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

## Dataloaders


def process_natq_clean(file_path):
    ## if not exist save processed file
    pass


def generate_natq_clean_dataloaders():
    ## convert things to data loaders
    txt = 'I am a question'
    input_question = tokenizer.encode_plus(txt, add_special_tokens=True,
                                           max_length=30, pad_to_max_length=True,
                                           return_tensors='pt')
    inputs_paragraph = tokenizer.batch_encode_plus(['I am positve' * 3, 'I am negative' * 4, 'I am negative'],
                                                   add_special_tokens=True,
                                                   pad_to_max_length=True,
                                                   max_length=512,
                                                   return_tensors='pt'
                                                   )
    dataset = TensorDataset(
        input_question['input_ids'].repeat(1, 1),
        input_question['attention_mask'].repeat(1, 1),
        input_question['token_type_ids'].repeat(1, 1),
        inputs_paragraph['input_ids'].unsqueeze(0).repeat(1, 1, 1),
        inputs_paragraph['attention_mask'].unsqueeze(0).repeat(1, 1, 1),
        inputs_paragraph['token_type_ids'].unsqueeze(0).repeat(1, 1, 1)
    )

    return DataLoader(dataset), DataLoader(dataset)


train_dataloader, dev_dataloader = generate_natq_clean_dataloaders()

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

    # def training_step(self, batch):
    #     """
    #     batch comes in the order of question, 1 positive paragraph,
    #     2 negative paragraphs
    #     """
    #     # batch
    #     input_ids_question, attention_mask_question, token_type_ids_question, \
    #     batch_input_ids_paragraphs, batch_attention_mask_paragraphs, \
    #     batch_token_type_ids_paragraphs = batch
    #
    #     inputs = {
    #         'input_ids_question': input_ids_question,
    #         'attention_mask_question': attention_mask_question,
    #         'token_type_ids_question': token_type_ids_question,
    #         'batch_input_ids_paragraphs': batch_input_ids_paragraphs,
    #         'batch_attention_mask_paragraphs': batch_attention_mask_paragraphs,
    #         'batch_token_type_ids_paragraphs': batch_token_type_ids_paragraphs
    #     }
    #
    #     h_question, h_paragraphs_batch = self(**inputs)
    #     h_paragraphs_pos = h_paragraphs_batch[:, 0, :]
    #     batch_h_paragraphs_negs = h_paragraphs_batch[:, 1:, :]
    #
    #     pos_loss = F.logsigmoid(torch.bmm(h_question.unsqueeze(1),
    #                                       h_paragraphs_pos.unsqueeze(-1))).reshape(-1)
    #     neg_loss = F.logsigmoid(torch.bmm(h_question.unsqueeze(1),
    #                                       batch_h_paragraphs_negs.permute(0, 2, 1))).sum(-1).reshape(-1)
    #     loss = (pos_loss + neg_loss).sum()
    #
    #     return loss



## Todo: torch lighting to train
class RetriverTrainer(pl.LightningModule):

    def __init__(self, retriever, emb_dim=768):
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
        h_paragraphs_pos = h_paragraphs_batch[:, 0, :]
        batch_h_paragraphs_negs = h_paragraphs_batch[:, 1:, :]
        pos_dot = torch.bmm(h_question.unsqueeze(1),
                  h_paragraphs_pos.unsqueeze(-1))
        neg_dot = torch.bmm(h_question.unsqueeze(1),
                                          batch_h_paragraphs_negs.permute(0, 2, 1))
        pos_loss = F.logsigmoid(pos_dot).reshape(-1)
        neg_loss = F.logsigmoid(neg_dot).sum(-1).reshape(-1)
        loss = (pos_loss + neg_loss).sum()

        return loss

    def training_step(self, batch, batch_idx):
        """
        batch comes in the order of question, 1 positive paragraph,
        2 negative paragraphs
        """

        train_loss = self.step_helper(batch)
        # logs
        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        loss = self.step_helper(batch)

        ## Todo: create val_acc
        val_acc = loss / 2

        return {'val_loss': loss, 'val_acc': val_acc}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad])
        # optimizer = AdamW([p for p in self.parameters() if p.requires_grad])
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps
        # )

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return dev_dataloader


if __name__ == '__main__':
    ## example

    encoder_question = BertEncoder(bert_question, 30)
    encoder_paragarph = BertEncoder(bert_paragraph, 512)
    # h = encoder_question(**tokenizer.encode_plus(txt, add_special_tokens=True,
    #     max_length=encoder_question.max_seq_len, pad_to_max_length=True,
    # return_tensors='pt'))

    # txt = 'I am a question'
    # input_question = tokenizer.encode_plus(txt, add_special_tokens=True,
    #     max_length=30, pad_to_max_length=True,
    # return_tensors='pt')
    # inputs_paragraph = tokenizer.batch_encode_plus(['I am positve' * 3, 'I am negative' * 4, 'I am negative'],
    #                                      add_special_tokens=True,
    #                                      pad_to_max_length=True,
    #                                      max_length=512,
    #                                      return_tensors='pt'
    #                                      )
    # train_dataset = TensorDataset(
    #     input_question['input_ids'].repeat(1, 1),
    #     input_question['attention_mask'].repeat(1, 1),
    #     input_question['token_type_ids'].repeat(1, 1),
    #     inputs_paragraph['input_ids'].unsqueeze(0).repeat(1, 1, 1),
    #     inputs_paragraph['attention_mask'].unsqueeze(0).repeat(1, 1, 1),
    #     inputs_paragraph['token_type_ids'].unsqueeze(0).repeat(1, 1, 1)
    # )
    #
    # inputs = {
    #     'input_ids_question': input_question['input_ids'],
    #     'attention_mask_question': input_question['attention_mask'],
    #     'token_type_ids_question': input_question['token_type_ids'],
    #     'batch_input_ids_paragraphs': inputs_paragraph['input_ids'],
    #     'batch_attention_mask_paragraphs': inputs_paragraph['attention_mask'],
    #     'batch_token_type_ids_paragraphs': inputs_paragraph['token_type_ids']
    # }

    ret = Retriver(encoder_question, encoder_paragarph)
    ret_trainee = RetriverTrainer(ret)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(ret_trainee)

    # h_q, h_ps = ret(**inputs)
    # print(h.shape, h_q.shape, h_ps.shape)



    # train_dataloader = DataLoader(train_dataset, batch_size=10)
    #
    # for batch in train_dataloader:
    #     pass
    #
    # loss = ret.training_step(batch)
    # print(loss)

