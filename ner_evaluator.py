import evaluate

class NEREvaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metric = evaluate.load('seqeval')

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
        for label in labels:
            try:
                entity, entity_type = label.split('::')
                entity_tokens = entity.split()
                if entity_type not in self.get_labels():
                    continue
                for idx in range(len(words) - len(entity_tokens) + 1):
                    if words[idx:idx + len(entity_tokens)] == entity_tokens:
                        bio_tags[idx:idx + len(entity_tokens)] = ["B-" + entity_type] + ["I-" + entity_type] * (len(entity_tokens) - 1)
            except:
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
        if example == 'Nah':
            return entities
        try:
            for label in example.split('\n'):
                entities.append(label)
        except:
            print(example)
        return entities

    def evaluate(self, df, output_file="results.txt"):
        """
        Evaluate the model predictions and write the results to a file.

        Parameters:
        df (DataFrame): DataFrame containing the gold and predicted labels.
        output_file (str): File to write the results.
        """
        df['golds'] = df['gold'].apply(self.convert_to_list)
        df['preds'] = df['pred'].apply(self.convert_to_list)
        df['pred_tags'] = df.apply(lambda x: self.convert_to_bio(x['words'], x['preds']), axis=1)

        results = self.metric.compute(predictions=df.pred_tags.tolist(), references=df.tags.tolist())

        # Calculate Macro F1
        f1_scores = [scores['f1'] for entity, scores in results.items() if entity not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']]
        macro_f1 = sum(f1_scores) / len(f1_scores)

        with open(output_file, 'a') as f:
            f.write(f"\n\nExperiment Results:\n")
            f.write(f"Model: {self.model_name}\n\n")
            
            for entity, scores in results.items():
                if entity not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
                    f.write(f"{entity}: F1 Score: {scores['f1']:.4f}\n")

            f.write("\nOverall Scores:\n")
            f.write(f"Micro F1 Score: {results['overall_f1']:.4f}\n")
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
            f.write('=' * 50)