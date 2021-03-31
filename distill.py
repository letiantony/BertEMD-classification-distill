import argparse
from functools import partial
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer,AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from data_processor import glue_processors as processors
from data_processor import glue_output_modes as output_modes
import os
import logging
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import sys


logger = logging.getLogger(__name__)

# Define Dict Dataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_and_cache_examples(args, task, tokenizer, mode="train"):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    train_stage = mode
    if mode == "eval":
        train_stage = "dev"
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        train_stage,
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if mode == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif mode == "eval":
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                # pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                # pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = DictDataset(all_input_ids, all_attention_mask, all_labels)
    #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


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
    return {'logits': model_outputs[0],
            'hidden': model_outputs[1],
            'inputs_mask': batch['attention_mask']}


# Define callback function
def predict(model, eval_dataset, device):
    '''
    eval_dataset: 验证数据集
    '''
    model.eval()
    pred_logits = []
    label_ids = []
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32)
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids, attention_mask=attention_mask)
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
 # fill other arguments


def main():
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
    parser.add_argument("--task_name", default="sst-2", type=str, 
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--model_type", default='bert', type=str, help="")
    parser.add_argument('--overwrite_cache', action='store_true',
                                        help="Overwrite the cached training and evaluation sets")
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    bert_config_T3 = BertConfig.from_json_file(args.student_config)
    bert_config_T3.output_hidden_states = True

    print("teacher model loading")

    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_pretrained_path)

    student_model = BertForSequenceClassification(bert_config_T3)  # , num_labels = 2

    teacher_model.to(device=device)
    student_model.to(device=device)
    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()
    args.output_mode = output_modes[args.task_name]
    processor = processors[args.task_name]()

    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.teacher_pretrained_path)
    print("loading data")
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, mode='train')
    dev_dataset = load_and_cache_examples(args, args.task_name, tokenizer, mode='eval')
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, mode='test')
    train_dataloader = DataLoader(train_dataset)
    num_epochs = args.num_epochs
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    num_training_steps = len(train_dataloader) * num_epochs


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
        eval_dataset=dev_dataset,
        device=device)  #

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
            {'layer_T': [0, 0], 'layer_S': [0, 0], 'feature': 'hidden', 'loss': 'nst', 'weight': 1},
            {'layer_T': [8, 8], 'layer_S': [2, 2], 'feature': 'hidden', 'loss': 'nst', 'weight': 1}]
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
            train_dataloader,
            num_epochs=num_epochs,
            scheduler_class=scheduler_class,
            scheduler_args=scheduler_args,
            callback=callback_fun)

    # testing
    predict(student_model,test_dataset,device)

if __name__ == "__main__":
    main()
