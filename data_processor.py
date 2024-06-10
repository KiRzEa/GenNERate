import pandas as pd
from datasets import Dataset, DatasetDict
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def __init__(self, data_dir, level):
        self.data_dir = data_dir
        self.level = level
    
    @abstractmethod
    def read_data(self, path):
        pass
    
    @abstractmethod
    def construct_prompt(self, df):
        pass

    def extract_aspect(self, tokens, ner_tags):
        """
        Extract entities from tokens based on NER tags.

        Parameters:
        tokens (list): List of tokens.
        ner_tags (list): List of NER tags corresponding to the tokens.

        Returns:
        list: List of extracted entities with their types.
        """
        assert len(tokens) == len(ner_tags)
        entities = []
        entity_tokens = []
        for idx, (token, tag) in enumerate(zip(tokens, ner_tags)):
            if tag.startswith('B') and len(entity_tokens) == 0:
                entity_tokens.append(token)
                entity_type = tag[2:]
            elif tag.startswith('I'):
                entity_tokens.append(token)
            else:
                if len(entity_tokens) > 0:
                    entities.append(' '.join(entity_tokens) + f'::{entity_type}')
                    entity_tokens = []

        return entities

    def create_dataset(self):
        """
        Create a dataset dictionary for training and testing data.

        Returns:
        DatasetDict: A dictionary containing the training and testing datasets.
        """
        train = self.read_data(f'{self.data_dir}/{self.level}/train_{self.level}.json')
        test = self.read_data(f'{self.data_dir}/{self.level}/test_{self.level}.json')
        dataset = DatasetDict({
            'train': Dataset.from_pandas(self.construct_prompt(train)),
            'test': Dataset.from_pandas(self.construct_prompt(test))
        })
        return dataset

    def get_labels(self):
        """
        Get the list of entity labels.

        Returns:
        list: List of entity labels.
        """
        return ['AGE', 'DATE', 'GENDER', 'JOB', 'LOCATION', 'NAME', 'ORGANIZATION', 'PATIENT_ID', 'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']

class MyDataProcessor(DataProcessor):
    def read_data(self, path):
        """
        Read data from a JSON file and preprocess it.

        Parameters:
        path (str): Path to the JSON file.

        Returns:
        DataFrame: Preprocessed data.
        """
        df = pd.read_json(path, lines=True)
        df['raw_labels'] = df.apply(lambda x: self.extract_aspect(x['words'], x['tags']), axis=1)
        df['labels'] = df.apply(lambda x: '\n'.join(x['raw_labels']), axis=1)
        df['labels'] = df['labels'].apply(lambda x: 'Nan' if x == '' else x)
        df['sentence'] = df['words'].apply(lambda x: ' '.join(x))
        return df[['words', 'tags', 'sentence', 'labels', 'raw_labels']]

    def construct_prompt(self, df):
        """
        Construct prompts from the data.

        Parameters:
        df (DataFrame): DataFrame containing the data.

        Returns:
        DataFrame: DataFrame with constructed prompts.
        """
        output_text = df['labels'].tolist()
        raw_tokens = df['words'].tolist()
        input_text = []
        
        for _, row in df.iterrows():
            input_sentence = row['sentence']
            prompt = PROMPT.format(input_sentence)
            input_text.append(prompt)
        
        assert len(input_text) == len(output_text)
        return pd.DataFrame(list(zip(raw_tokens, df['tags'], input_text, output_text)), columns=['words', 'tags', 'text', 'label'])