import pandas as pd
from datasets import Dataset, DatasetDict
from prompt import PROMPT
import evaluate 

metric = evaluate.load('seqeval')
def read_data(path):
    df = pd.read_json(path, lines=True)

    df['raw_labels'] = df.apply(lambda x: extract_aspect(x['words'], x['tags']), axis=1)
    df['labels'] = df.apply(lambda x: '\n'.join(x['raw_labels']), axis=1)
    df['labels'] = df['labels'].apply(lambda x: 'Nan' if x == '' else x)
    df['sentence'] = df['words'].apply(lambda x: ' '.join(x))
    return df[['words', 'tags', 'sentence', 'labels', 'raw_labels']]

def construct_prompt(df):
    output_text = df['labels'].tolist()
    raw_tokens = df['words'].tolist()
    tags = df['tags']
    input_text = []
    
    for _, row in df.iterrows():
        input_sentence = row['sentence']
        prompt = PROMPT.format(input_sentence)

        input_text.append(prompt)
    
    assert len(input_text) == len(output_text)
    return pd.DataFrame(list(zip(raw_tokens, tags, input_text, output_text)), columns=['words', 'tags', 'text', 'label'])

def extract_aspect(tokens, ner_tags):
    assert len(tokens) == len(ner_tags)
    entities = []
    entity_tokens = []
    for idx, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith('B') and len(entity_tokens) == 0:
          print(token, tag[2:])
          entity_tokens.append(token)
          entity_type = tag[2:]
        elif tag.startswith('I'):
          entity_tokens.append(token)
        else:
          if len(entity_tokens) > 0:
            entities.append(' '.join(entity_tokens)+f'::{entity_type}')
            entity_tokens = []

    return entities

def create_dataset(data_dir, level):
    train = read_data(f'{data_dir}/{level}/train_{level}.json')
    test = read_data(f'{data_dir}/{level}/test_{level}.json')
    dataset = DatasetDict({
        'train': Dataset.from_pandas(construct_prompt(train)),
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


def get_labels():
    return ['AGE', 'DATE', 'GENDER', 'JOB', 'LOCATION', 'NAME', 'ORGANIZATION', 'PATIENT_ID', 'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']


def convert_to_bio(words, labels):
    """
    Convert generation-based NER output to BIO tagging format for each sample.

    Parameters:
    words (str): The input words.
    labels (str): The NER output in the format 'ENTITY::ENTITY_TYPE'.

    Returns:
    list: A list of tuples where each tuple contains a token and its corresponding BIO tag.
    """

    # Tokenize the input text
    # tokens = len(words)
    # Initialize BIO tags
    bio_tags = ["O"] * len(words)

    # Process each entity in the NER output
    for label in labels:
        try:
            entity, entity_type = label.split('::')
            entity_tokens = entity.split()
            if entity_type not in get_labels():
                continue
            for idx in range(len(words) - len(entity_tokens) + 1):
                if words[idx:idx+len(entity_tokens)] == entity_tokens:
                    bio_tags[idx:idx+len(entity_tokens)] = ["B-" + entity_type] + ["I-" + entity_type] * (len(entity_tokens) - 1)
        except:
            continue
    # Return the tokens and their corresponding BIO tags
    return bio_tags

def convert_to_list(example):
    entities = []
    if example == 'Nah':
      return entities
    try:
      for label in example.split('\n'):
          entities.append(label)
    except:
      print(example)
    return entities

def evaluate(df, model_name, output_file="results.txt"):
    df['golds'] = df['gold'].apply(convert_to_list)
    df['preds'] = df['pred'].apply(convert_to_list)
    
    df['pred_tags'] = df.apply(lambda x: convert_to_bio(x['words'], x['preds']), axis=1)

    results = metric.compute(predictions=df.pred_tags.tolist(), references=df.tags.tolist())
    
    # Calculate Macro F1
    f1_scores = [scores['f1'] for entity, scores in results.items() if entity not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']]
    macro_f1 = sum(f1_scores) / len(f1_scores)

    with open(output_file, 'a') as f:
        f.write(f"\n\nExperiment Results:\n")
        f.write(f"Model: {model_name}\n\n")
        
        for entity, scores in results.items():
            if entity not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
                f.write(f"{entity}: F1 Score: {scores['f1']:.4f}\n")

        f.write("\nOverall Scores:\n")
        f.write(f"Micro F1 Score: {results['overall_f1']:.4f}\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write('=' * 50)
evaluate(pred, 'bloom560m')