import evaluate

class NEREvaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metric = evaluate.load('seqeval')

    def get_labels(self):
        """
        Returns a list of all supported label types.

        This function retrieves all the different categories (e.g., AGE, NAME)
        that can be used to label data points within the current context.

        Returns:
            list: A list containing all supported label types as strings.
        """
        return ['AGE', 'DATE', 'GENDER', 'JOB', 'LOCATION', 'NAME', 'ORGANIZATION', 'PATIENT_ID', 'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']

    def convert_to_bio(self, words, labels):
        """
        Convert generation-based NER output to BIO tagging format.

        Parameters:
        words (list): List of tokens.
        labels (list): List of NER output in the format 'ENTITY::ENTITY_TYPE'.

        Returns:
        list: List of BIO tags corresponding to the tokens.
        """
        bio_tags = ["O"] * len(words)
        start = 0
        for label in labels:
            try:
                entity, entity_type = label.strip().split('::')
                entity_tokens = entity.split()
                if entity_type not in self.get_labels():
                    continue
                for idx in range(start, len(words) - len(entity_tokens) + 1):
                    if (words[idx:idx + len(entity_tokens)] == entity_tokens): #and (bio_tags[idx:idx + len(entity_tokens)] == ['O'] * len(entity_tokens)):
                        assert len(entity_tokens) == len(bio_tags[idx:idx + len(entity_tokens)])
                        bio_tags[idx:idx + len(entity_tokens)] = ["B-" + entity_type] + ["I-" + entity_type] * (len(entity_tokens) - 1)
                        start = idx + len(entity_tokens)
                        break
            except Exception as e:
                print(e)
                print(label)
                continue
        return bio_tags

    def convert_to_list(self, example):
        """
        Convert newline-separated entity labels to a list.

        Parameters:
        example (str): Newline-separated entity labels.

        Returns:
        list: List of entity labels.
        """
        entities = []
        example = example.strip()
        if example == 'Nah':
            return entities
        try:
            for label in example.split('\n'):
                if len(label.split('::')) == 2:
                    entities.append(label.strip())
                else:
                    continue
        except Exception as e:
            print(e)
        return entities

    def evaluate(self, df, level, output_file="results.txt"):
        """
        Evaluate the model predictions and write the results to a file.

        Parameters:
        df (DataFrame): DataFrame containing the gold and predicted labels.
        output_file (str): File to write the results.
        """
        df['preds'] = df['pred'].apply(self.convert_to_list)
        df['pred_tags'] = df.apply(lambda x: self.convert_to_bio(x['words'], x['preds']), axis=1)

        results = self.metric.compute(predictions=df.pred_tags.tolist(), references=df.tags.tolist())

        # Calculate Macro F1
        f1_scores = [scores['f1'] for entity, scores in results.items() if entity not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']]
        macro_f1 = sum(f1_scores) / len(f1_scores)

        with open(output_file, 'a') as f:
            f.write(f"Experiment Results:\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Level: {level}\n\n")

            for entity, scores in results.items():
                if entity not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
                    f.write(f"{entity}: F1 Score: {scores['f1']:.4f}\n")

            f.write("\nOverall Scores:\n")
            f.write(f"Micro F1 Score: {results['overall_f1']:.4f}\n")
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
            f.write('=' * 50 + '\n')

        df[['words', 'tags','pred_tags', 'preds']].to_csv(f"{self.model_name.replace('/', '-')}_{level}.csv", index=False)