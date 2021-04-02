from functools import partial
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
import time

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
    return {'logits': model_outputs[1],
            'hidden': model_outputs[2],
            'inputs_mask': batch['attention_mask']}


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

# Define callback function
def predict(model, eval_dataset, device):
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
    parser.add_argument("--data_dir", default="/home/machunping/thunews", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--student_config", default="bert_config/bert_config_T3.json", type=str,
                        help="the json file of student model config")
    parser.add_argument("--teacher_pretrained_path", default="/home/machunping/fine_tuned_bert_thucnews", type=str,
                        help="the path of teacher pretrained model")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_epochs', type=int, default=30,
                        help="training epochs")
    parser.add_argument('--num_labels', type=int, default=10,
                        help="numbers of labels")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="batch_size")
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="")
    parser.add_argument("--task_name", default="thucnews", type=str, help="")
    parser.add_argument("--model_type", default='bert', type=str, help="")
    parser.add_argument("--output_dir", default='saved_model', type=str, help="")

    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    args = parser.parse_args()

    # device
    device = torch.device("cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Define models
    # bert_config = BertConfig.from_json_file(args.teacher_pretrained_path+"/config.json")
    bert_config_T3 = BertConfig.from_json_file('bert_config/thucnews_bert_config_T3.json')

    # bert_config.output_hidden_states = True
    bert_config_T3.output_hidden_states = True


    student_model = BertForSequenceClassification(bert_config_T3)  # , num_labels = 2
    student_model.num_labels = args.num_labels
    model_path = "/home/machunping/bertemd/thucnew_saved_model_big/model.pkl"
    student_model.load_state_dict(torch.load(model_path))


    # teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_pretrained_path)
    student_model.to(device=device)

    # Define Dict Dataset

    # Prepare random data
    tokenizer = BertTokenizer.from_pretrained(args.teacher_pretrained_path)

    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, mode='test')

    print("test model result:")
    start = time.time()
    predict(model=student_model, eval_dataset=test_dataset, device=device)
    print((time.time()-start)/len(test_dataset))
