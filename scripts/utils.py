import pandas as pd
from datasets import Dataset, DatasetDict
from prompt import PROMPT

def read_data(path):
    df = pd.read_json(path, lines=True)
    # df = df[df['have_ner'] != 0]
    df['labels'] = df.apply(lambda x: '\n'.join([f"{entity_word}::{entity_type}" for entity_word, entity_type in zip(x['entity_words'], x['entity_types'])]),
                            axis=1)
    df['labels'] = df.apply(lambda x: 'None' if x == '' else x)
    return df[['id', 'domain', 'sentence', 'have_ner', 'labels']]

def construct_prompt(df):
    output_text = df['labels'].tolist()
    input_text = []
    
    for _, row in df.iterrows():
        input_sentence = row['sentence']
        prompt = PROMPT.format(input_sentence)

        input_text.append(prompt)
    
    assert len(input_text) == len(output_text)
    return pd.DataFrame(list(zip(input_text, output_text)), columns=['text', 'label'])

def create_dataset(data_dir):
    train = read_data(f'{data_dir}/train_sentence.json')
    dev = read_data(f'{data_dir}/dev_sentence.json')
    test = read_data(f'{data_dir}/test_sentence.json')
    dataset = DatasetDict({
        'train': Dataset.from_pandas(construct_prompt(train)),
        'dev': Dataset.from_pandas(construct_prompt(dev)),
        'test': Dataset.from_pandas(construct_prompt(test))
    })
    
    return dataset

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def evaluate(preds, golds, output_file="results.txt"):
    tp = 0.
    fp = 0.
    fn = 0.
    for pred, gold in zip(preds, golds):
        pred = pred[0].split('\n')
        gold = gold[0].split('\n')
        for entity in gold:
            if entity in pred:
                tp += 1
            else:
                fn += 1
        for aspect in pred:
            if aspect not in gold:
                fp += 1
    precision = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    recall = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    
    with open(output_file, 'a') as f:
        f.write(f"\n\nExperiment Results:\n")
        f.write(f"Model: {model_name}")
        f.write(f"tp: {tp}, fp: {fp}, fn: {fn}\n")
        f.write(f"p: {precision}, r: {recall}, f1: {f1}\n")
        
    print(f"tp: {tp}, fp: {fp}, fn: {fn}")
    print(f"p: {precision}, r: {recall}, f1: {f1}")
    return {'precision': precision, 'recall': recall, 'f1': f1}