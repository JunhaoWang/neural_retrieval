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
import json
import os
from tqdm.auto import tqdm

## Berts
model_str = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_str)
bert_question = BertModel.from_pretrained(model_str)
bert_paragraph = BertModel.from_pretrained(model_str)

## Hyperparams that wont likely change in the future
num_dat_global = 40
batch_size_global = 3
max_question_len_global = 30
max_paragraph_len_global = 512
default_bert_emb_dim_global = 768
train_set_file_name = 'natq_train.pt'
dev_set_file_name = 'natq_dev.pt'
natq_json_file = 'natq_clean.json'
data_folder_name = 'natq/'
assert  data_folder_name[-1] == '/'

def remove_html_toks(s):
    html_toks = [
        '<P>',
        '</P>',
        '<H1>',
        '</H1>',
        '</H2>',
        '</H2>',
    ]
    for i in html_toks:
        s = s.replace(i, '')
    return s

def process_natq_clean(folder_name = data_folder_name, input_file = natq_json_file, output_train_file = train_set_file_name,
                       output_dev_file = dev_set_file_name, max_question_len = max_question_len_global,
                       max_paragraph_len = max_paragraph_len_global):
    assert  folder_name[-1] == '/'

    if not os.path.exists(folder_name + input_file):
        raise Exception('{} not found in {}'.format(input_file, folder_name))

    train_exists = False
    dev_exists = False

    if os.path.exists(folder_name + output_train_file):
        train_exists = True
    if os.path.exists(folder_name + output_dev_file):
        dev_exists = True

    if train_exists and dev_exists:
        return

    if not train_exists:
        train_input_ids_question, train_attention_mask_question, train_token_type_ids_question, \
        train_batch_input_ids_paragraphs, train_batch_attention_mask_paragraphs, \
        train_batch_token_type_ids_paragraphs = [], [], [], [], [], []
    if not dev_exists:
        dev_input_ids_question, dev_attention_mask_question, dev_token_type_ids_question, \
        dev_batch_input_ids_paragraphs, dev_batch_attention_mask_paragraphs, \
        dev_batch_token_type_ids_paragraphs = [], [], [], [], [], []

    with open(folder_name + input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for l in tqdm(f):
            d = json.loads(l)

            if d['num_positives'] >= 1 and d['num_negatives'] >= 2:

                if d['dataset'] == 'train' and train_exists:
                    continue
                if d['dataset'] == 'dev' and dev_exists:
                    continue

                q = d['question']
                paras = d['right_paragraphs'][:1] + d['wrong_paragraphs'][:2]
                paras = [remove_html_toks(i) for i in paras]

                input_question = tokenizer.encode_plus(q, add_special_tokens=True,
                                                       max_length=max_question_len, pad_to_max_length=True,
                                                       return_tensors='pt')
                inputs_paragraph = tokenizer.batch_encode_plus(paras,
                                                               add_special_tokens=True,
                                                               pad_to_max_length=True,
                                                               max_length=max_paragraph_len,
                                                               return_tensors='pt'
                                                               )

                if d['dataset'] == 'train':
                    train_input_ids_question.append(input_question['input_ids'])
                    train_attention_mask_question.append(input_question['attention_mask'])
                    train_token_type_ids_question.append(input_question['token_type_ids'])
                    train_batch_input_ids_paragraphs.append(inputs_paragraph['input_ids'].unsqueeze(0))
                    train_batch_attention_mask_paragraphs.append(inputs_paragraph['attention_mask'].unsqueeze(0))
                    train_batch_token_type_ids_paragraphs.append(inputs_paragraph['token_type_ids'].unsqueeze(0))

                elif d['dataset'] == 'dev':
                    dev_input_ids_question.append(input_question['input_ids'])
                    dev_attention_mask_question.append(input_question['attention_mask'])
                    dev_token_type_ids_question.append(input_question['token_type_ids'])
                    dev_batch_input_ids_paragraphs.append(inputs_paragraph['input_ids'].unsqueeze(0))
                    dev_batch_attention_mask_paragraphs.append(inputs_paragraph['attention_mask'].unsqueeze(0))
                    dev_batch_token_type_ids_paragraphs.append(inputs_paragraph['token_type_ids'].unsqueeze(0))
    if not dev_exists:
        dev_set = TensorDataset(
            torch.cat(dev_input_ids_question),
            torch.cat(dev_attention_mask_question),
            torch.cat(dev_token_type_ids_question),
            torch.cat(dev_batch_input_ids_paragraphs),
            torch.cat(dev_batch_attention_mask_paragraphs),
            torch.cat(dev_batch_token_type_ids_paragraphs),
        )
        torch.save(dev_set, folder_name + output_dev_file)

    if not train_exists:
        train_set = TensorDataset(
            torch.cat(train_input_ids_question),
            torch.cat(train_attention_mask_question),
            torch.cat(train_token_type_ids_question),
            torch.cat(train_batch_input_ids_paragraphs),
            torch.cat(train_batch_attention_mask_paragraphs),
            torch.cat(train_batch_token_type_ids_paragraphs),
        )

        torch.save(train_set, folder_name + output_train_file)


def generate_natq_dataloaders(folder_path=data_folder_name, input_train_file=train_set_file_name,
        input_dev_file=dev_set_file_name, input_json_file = natq_json_file, batch_size = batch_size_global):
    assert  folder_path[-1] == '/'

    if (not os.path.exists(folder_path + input_train_file)) or (not os.path.exists(folder_path + input_dev_file)):
        process_natq_clean(folder_path, input_json_file, input_train_file, input_dev_file)

    train_set = torch.load(folder_path+ input_train_file)
    dev_set = torch.load(folder_path + input_dev_file)
    return DataLoader(train_set, batch_size=batch_size), DataLoader(dev_set, batch_size=batch_size)


def generate_fake_dataloaders(num_dat = num_dat_global, batch_size = batch_size_global,
                              max_question_len=max_question_len_global, max_paragraph_len=max_paragraph_len_global):
    ## convert things to data loaders
    txt = 'I am a question'
    input_question = tokenizer.encode_plus(txt, add_special_tokens=True,
                                           max_length=max_question_len, pad_to_max_length=True,
                                           return_tensors='pt')
    inputs_paragraph = tokenizer.batch_encode_plus(['I am positve' * 3, 'I am negative' * 4, 'I am negative'],
                                                   add_special_tokens=True,
                                                   pad_to_max_length=True,
                                                   max_length=max_paragraph_len,
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



train_dataloader, dev_dataloader = generate_natq_dataloaders()

## nn.Module classes

class BertEncoder(nn.Module):

    def __init__(self, bert, max_seq_len, emb_dim = default_bert_emb_dim_global):
        super(BertEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.bert = bert
        self.num_forward = 0
        for param in self.bert.parameters():
            param.requires_grad = False
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.num_forward == 30000:
            for param in self.bert.parameters():
                param.requires_grad = True
        h, _ = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        h_cls = h[:, 0]
        h_transformed = self.net(h_cls)
        self.num_forward += 1
        if self.num_forward % 999 == 1:
            print('Bert Encoder forwarded {} times'.format(self.num_forward))
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

    def refresh_cache(self):
        self.cache_hash2array = {}
        self.cache_hash2str = {}

    def predict(self, question_str: str, batch_paragraph_strs: List[str], refresh_cache = False):
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
            inputs = self.tokenizer.batch_encode_plus(
                uncached_paragraphs,
                 add_special_tokens=True,
                 pad_to_max_length=True,
                 max_length=self.max_paragraph_len,
                 return_tensors='pt'
             )
            if len(inputs):
                tmp_device = next(self.bert_paragraph_encoder.parameters()).device
                inputs = {i:inputs[i].to(tmp_device) for i in inputs}
                uncached_paragraph_array = self.bert_paragraph_encoder(
                    **inputs
                ).detach().cpu().numpy()
                for ind, i in enumerate(uncached_paragraph_array):
                    self.cache_hash2array[uncached_hashes[ind]] = deepcopy(i)
                    batch_paragraph_array[ind,:] = deepcopy(i)
            inputs = self.tokenizer.encode_plus(question_str, add_special_tokens=True,
                   max_length=self.max_question_len, pad_to_max_length=True,
                   return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            inputs = {i:inputs[i].to(tmp_device) for i in inputs}
            question_array = self.bert_question_encoder(
                **inputs
            )
            relevance_scores = torch.sigmoid(
                torch.mm(torch.tensor(batch_paragraph_array, dtype=question_array.dtype).to(question_array.device),
                         question_array.T)).reshape(-1)
            rerank_index = torch.argsort(-relevance_scores)
            relevance_scores_numpy = relevance_scores.detach().cpu().numpy()
            rerank_index_numpy = rerank_index.detach().cpu().numpy()
            reranked_paragraphs = [batch_paragraph_strs[i] for i in rerank_index_numpy]
            reranked_relevance_scores = relevance_scores_numpy[rerank_index_numpy]
            return reranked_paragraphs, reranked_relevance_scores, rerank_index_numpy

class RetriverTrainer(pl.LightningModule):

    def __init__(self, retriever, emb_dim=default_bert_emb_dim_global):
        super(RetriverTrainer, self).__init__()
        self.retriever = retriever
        self.emb_dim = emb_dim

    def forward(self, **kwargs):
        return self.retriever(**kwargs)

    # def step_helper(self, batch):
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
    #     batch_size, num_document, emb_dim = batch_input_ids_paragraphs.size()
    #
    #     all_dots = torch.bmm(h_question.repeat(num_document, 1).unsqueeze(1),
    #         h_paragraphs_batch.reshape(-1, self.emb_dim).unsqueeze(2)).reshape(batch_size, num_document)
    #     all_prob = torch.sigmoid(all_dots)
    #
    #     pos_loss = - torch.log(all_prob[:, 0]).sum()
    #     neg_loss = - torch.log(1 - all_prob[:, 1:]).sum()
    #     loss = pos_loss + neg_loss
    #     return loss, all_prob

    def step_helper_classification(self, batch):
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

        all_dots = torch.bmm(h_question.repeat(num_document, 1).unsqueeze(1),
            h_paragraphs_batch.reshape(-1, self.emb_dim).unsqueeze(2)).reshape(batch_size, num_document)
        logits = all_dots
        all_prob = torch.sigmoid(all_dots)

        loss = nn.CrossEntropyLoss()(
            logits, torch.zeros(logits.size()[0],dtype=torch.long).to(logits.device)
        )
        return loss, all_prob

    def training_step(self, batch, batch_idx):
        """
        batch comes in the order of question, 1 positive paragraph,
        K negative paragraphs
        """

        train_loss, _ = self.step_helper_classification(batch)
        # logs
        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        loss, all_prob = self.step_helper_classification(batch)
        
        
        return {'val_loss': loss}


    def validation_epoch_end(self, outputs):
        try:
            avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        except:
            avg_val_loss = torch.cat([x['val_loss'] for x in outputs], 0).mean()
        
        hits = []
        c = 0
        self.retriever.refresh_cache()
        with open('natq/natq_clean.json', 'r') as f:
            for l in f:
                d = json.loads(l)
                if d['num_positives'] >= 1 and d['num_negatives'] >= 2:
                    if d['dataset'] == 'dev':
                        q = d['question']
                        txts = [remove_html_toks(i) for i in d['right_paragraphs'][:1] + d['wrong_paragraphs'][:2]]
                        r = ret_trainee.retriever.predict(q, txts)
                        if r[0][0] == txts[0]:
                            hits.append(1)
                        else:
                            hits.append(0)
                        c += 1
                        if c % 99 == 1:
                            print(c, sum(hits) / len(hits))
        print('Validation Top 1 Hit Accuracy: ', sum(hits) / len(hits))
        self.retriever.refresh_cache()
        avg_val_acc = torch.tensor([sum(hits) / len(hits)]).to(avg_val_loss.device)
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

#     def on_post_performance_check(self):
#         hits = []
#         c = 0
#         self.retriever.refresh_cache()
#         with open('natq/natq_clean.json', 'r') as f:
#             for l in f:
#                 d = json.loads(l)
#                 if d['num_positives'] >= 1 and d['num_negatives'] >= 2:
#                     if d['dataset'] == 'dev':
#                         q = d['question']
#                         txts = [remove_html_toks(i) for i in d['right_paragraphs'][:1] + d['wrong_paragraphs'][:2]]
#                         r = ret_trainee.retriever.predict(q, txts)
#                         logits.append(r[1])
#                         if r[0][0] == txts[0]:
#                             hits.append(1)
#                         else:
#                             hits.append(0)
#                         c += 1
#                         if c % 99 == 1:
#                     print(c, sum(hits) / len(hits))
        

if __name__ == '__main__':
    encoder_question = BertEncoder(bert_question, max_question_len_global)
    encoder_paragarph = BertEncoder(bert_paragraph, max_paragraph_len_global)
    ret = Retriver(encoder_question, encoder_paragarph, tokenizer)
    os.makedirs('out', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath='out/crossentropy-{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    early_stopping = EarlyStopping('val_acc', mode='max')

    trainer = pl.Trainer(
        gpus=1,
#         gradient_clip_val=0.5,
#         distributed_backend='dp',
        val_check_interval=0.05,
        min_epochs=1,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping)

    ret_trainee = RetriverTrainer(ret)

    trainer.fit(ret_trainee)
