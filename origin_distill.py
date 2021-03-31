from functools import partial
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from distill import load_and_cache_examples
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
# device
device = torch.device('cuda')

# Define models
bert_config = BertConfig.from_json_file('bert_config/bert_config.json')
bert_config_T3 = BertConfig.from_json_file('bert_config/bert_config_T3.json')

bert_config.output_hidden_states = True
bert_config_T3.output_hidden_states = True

teacher_model = BertForSequenceClassification(bert_config)  # , num_labels = 2
# Teacher should be initialized with pre-trained weights and fine-tuned on the downstream task.
# For the demonstration purpose, we omit these steps here

student_model = BertForSequenceClassification(
    bert_config_T3)  # , num_labels = 2

teacher_model.to(device=device)
student_model.to(device=device)

# Define Dict Dataset


class DictDataset(Dataset):
    def __init__(self, all_input_ids, all_attention_mask, all_labels):
        assert len(all_input_ids) == len(all_attention_mask) == len(all_labels)
        self.all_input_ids = all_input_ids
        self.all_attention_mask = all_attention_mask
        self.all_labels = all_labels

    def __getitem__(self, index):
        return {'input_ids': self.all_input_ids[index],
                'attention_mask': self.all_attention_mask[index],
                'labels': self.all_labels[index]}

    def __len__(self):
        return self.all_input_ids.size(0)

def simple_adaptor(batch, model_outputs):
    # The second element of model_outputs is the logits before softmax
    # The third element of model_outputs is hidden states
    # print("\noutput_length:\n")
    # print(len(model_outputs))
    # print(model_outputs)
    sys.exit()
    return {'logits': model_outputs[1],
            'hidden': model_outputs[2],
            'inputs_mask': batch['attention_mask']}


# Define callback function
def predict(model, eval_dataset, step, device):
    '''
    eval_dataset: 验证数据集
    '''
    model.eval()
    pred_logits = []
    label_ids = []
    dataloader = DataLoader(eval_dataset, batch_size=32)
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        with torch.no_grad():
            outputs= model(input_ids=input_ids, attention_mask=attention_mask)
            logits,_ = outputs
            cpu_logits = logits.detach().cpu()
        for i in range(len(cpu_logits)):
            pred_logits.append(cpu_logits[i].numpy())
            label_ids.append(labels[i])
    model.train()
    pred_logits = np.array(pred_logits)
    label_ids = np.array(label_ids)
    y_p = pred_logits.argmax(axis=-1)
    accuracy = (y_p == label_ids).sum() / len(label_ids)
    print("Number of examples: ", len(y_p))
    print("Acc: ", accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--data_dir", default="data/sst2", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--student_config", default="bert_config/bert_config_T3.json", type=str,
                        help="the json file of student model config")
    parser.add_argument("--teacher_pretrained_path", default="/home/machunping/fine_tuned_bert_sst2", type=str,
                        help="the path of teacher pretrained model")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch_size")
    parser.add_argument('--max_seq_length', type=int, default=64,
                        help="")
    parser.add_argument("--task_name", default="sst-2", type=str, help="")
    parser.add_argument("--model_type", default='bert', type=str, help="")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    args = parser.parse_args()

    # Prepare random data
    all_input_ids = torch.randint(
        low=0, high=100, size=(
            10, 12))  # 100 examples of length 128
    all_attention_mask = torch.ones_like(all_input_ids)
    all_labels = torch.randint(low=0, high=2, size=(10,))
    dataset = DictDataset(all_input_ids, all_attention_mask, all_labels)
    eval_dataset = DictDataset(all_input_ids, all_attention_mask, all_labels)
    dataloader = DataLoader(dataset, batch_size=2)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_pretrained_path)
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, mode='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)


    num_epochs = 1
    num_training_steps = len(dataloader) * num_epochs
    # Optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=1e-4)

    scheduler_class = get_linear_schedule_with_warmup
    # arguments dict except 'optimizer'
    scheduler_args = {
        'num_warmup_steps': int(
            0.1 * num_training_steps),
        'num_training_steps': num_training_steps}


    # display model parameters statistics
    print("\nteacher_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
    print(result)

    print("student_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
    print(result)





    callback_fun = partial(
        predict,
        eval_dataset=eval_dataset,
        device=device)  # fill other arguments


    # Initialize configurations and distiller
    train_config = TrainingConfig(device=device)
    distill_config = DistillationConfig(
        temperature=8,
        hard_label_weight=0,
        kd_loss_type='ce',
        probability_shift=False,
        intermediate_matches=[
            {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
            {'layer_T': 8, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
            {'layer_T': [0, 0], 'layer_S':[0, 0], 'feature':'hidden', 'loss': 'nst', 'weight': 1},
            {'layer_T': [8, 8], 'layer_S':[2, 2], 'feature':'hidden', 'loss': 'nst', 'weight': 1}]
    )

    print("train_config:")
    print(train_config)

    print("distill_config:")
    print(distill_config)

    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

    # Start distilling
    with distiller:
        distiller.train(
            optimizer,
            dataloader,
            num_epochs=num_epochs,
            scheduler_class=scheduler_class,
            scheduler_args=scheduler_args,
            callback=callback_fun)


