import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer, GenerationConfig, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import pandas as pd

from prompt import *
from data_processor import MyDataProcessor  # Assume the data processor code is in data_processor.py
from ner_evaluator import NEREvaluator  # Assume the evaluator code is in ner_evaluator.py

class NERTrainingPipeline:
    """
    A class representing the NER training pipeline.

    Args:
        args: An object containing the training pipeline arguments.

    Attributes:
        model_name (str): The name of the model.
        lr (float): The learning rate.
        num_epochs (int): The number of epochs for training.
        batch_size (int): The batch size for training.
        data_dir (str): The directory containing the training data.
        syllable (bool): Whether the data is syllable-based.
        device (torch.device): The device for training (CPU or GPU).
        peft_config (LoraConfig): The configuration for the PEFT model.
        dataset (DatasetDict): The dataset for training and testing.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        base_model (AutoModelForCausalLM): The base model for training.
        model (torch.nn.Module): The PEFT model for training.
        max_input_length (int): The maximum length of input sequences.
        max_output_length (int): The maximum length of output sequences.
        max_length (int): The maximum combined length of input and output sequences.
        processed_datasets (DatasetDict): The preprocessed datasets.
        train_dataloader (DataLoader): The DataLoader for training.
        test_dataloader (DataLoader): The DataLoader for testing.
    """
    def __init__(self, args):
        self.model_name = args.model_name
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.syllable = args.syllable
        self.bf16 = args.bf16
        self.fp16 = args.fp16
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.architectures = self.config.architectures[0]
        
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=512,
            lora_alpha=1024,
            lora_dropout=0.05,
            target_modules=self.get_target_modules()
        )

        
        self.training_arguments = TrainingArguments(
            output_dir="checkpoint",
            per_device_train_batch_size=self.batch_size,
            # gradient_accumulation_steps=16,
            # optim='adamw_8bit',
            num_train_epochs=self.num_epochs,
            logging_steps=512,
            save_strategy="no",
            load_best_model_at_end=False,
            warmup_ratio = 0.1,
            learning_rate=self.lr,
            report_to="none",
            bf16=self.bf16,
            fp16=self.fp16,
            half_precision_backend="auto"
        )


        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        
        self.dataset = self.create_dataset()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token='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')
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                                device_map="auto", 
                                                                quantization_config=self.quant_config, 
                                                                torch_dtype=torch.float16)
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # response_template_ids = self.tokenizer.encode(response_template, add_special_tokens=False)[1:]
        # self.collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=self.tokenizer)
        # self.processed_datasets = self.dataset.remove_columns([col for col in self.dataset['train'].column_names if col not in ['input', 'output']])        
        self.max_input_length, self.max_output_length, self.max_length = self.get_max_lengths()
        
        self.processed_datasets = self.preprocess_datasets()
        print(self.processed_datasets)

        # self.trainer = SFTTrainer(
        #     model=self.base_model,
        #     train_dataset=self.processed_datasets["train"],
        #     peft_config=self.peft_config,
        #     max_seq_length=self.max_length,
        #     args=self.training_arguments,
        #     data_collator=self.collator,
        #     formatting_func=formatting_prompts_func
        # )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=self.processed_datasets["train"],
            data_collator=self.collator
        )

        self.print_trainable_parameters()

    def get_target_modules(self):
        if self.architectures == 'BloomForCausalLM':
            target_modules = [
                            "query_key_value",
                            "dense",
                            "dense_h_to_4h",
                            "dense_4h_to_h",
                        ]
        elif self.architectures == 'LlamaForCausalLM':
            target_modules=[
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ]
            return target_modules

    def preprocess_function(self, examples):
        """
        Preprocess the dataset examples.

        Args:
            examples (dict): A dictionary containing input and output examples.

        Returns:
            dict: Preprocessed model inputs.
        """
        # batch_size = len(examples["input"])
        # inputs = [instruction_template + item + response_template + " " for item in examples["input"]]
        # targets = examples["output"]
        # model_inputs = self.tokenizer(inputs)
        # labels = self.tokenizer(targets, add_special_tokens=False)

        # for i in range(batch_size):
        #     sample_input_ids = model_inputs["input_ids"][i]
        #     label_input_ids = labels["input_ids"][i] + [self.tokenizer.eos_token_id]
        #     model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        #     labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        #     model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        # for i in range(batch_size):
        #     sample_input_ids = model_inputs["input_ids"][i]
        #     label_input_ids = labels["input_ids"][i]
        #     model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
        #     model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs["attention_mask"][i]
        #     labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
        #     model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
        #     model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length]) 
        #     labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])

        # model_inputs["labels"] = labels["input_ids"]
        # return model_inputs
        return {
            'prompt': [instruction_template + inp + response_template for inp in examples['input']],
            'completion': [out for out in examples['output']]
        }

    def preprocess_datasets(self):
        """
        Preprocess the training and testing datasets.

        Returns:
            DatasetDict: The preprocessed datasets.
        """
        return self.dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    def get_max_lengths(self):
        """
        Calculate the maximum lengths of input and output sequences.

        Returns:
            tuple: A tuple containing the maximum input length, maximum output length, and maximum combined length.
        """
        output_length = [len(self.tokenizer(review)["input_ids"]) for review in self.dataset['train']["output"]]
        input_length = [len(self.tokenizer(instruction_template + review + response_template)["input_ids"]) for review in self.dataset['train']["input"]]
        max_output_length = max(output_length)
        max_input_length = max(input_length)
        max_length = max([inp + out for inp, out in zip(output_length, input_length)])
        return max_input_length, max_output_length, max_length

    def create_dataset(self):
        """
        Create the dataset for training and testing.

        Returns:
            DatasetDict: The dataset containing training and testing data.
        """
        processor = MyDataProcessor(self.data_dir, 'syllable' if self.syllable else 'word')
        return processor.create_dataset()

    def print_trainable_parameters(self):
        """
        Print the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.trainer.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def get_prediction(self, examples):
        """
        Generate predictions for input examples.

        Args:
            examples (str): The input examples.

        Returns:
            list: The list of predicted labels.
        """
        input_ids = self.tokenizer([instruction_template + example + response_template for example in examples],
         max_length=self.max_input_length, return_tensors="pt", padding="max_length", truncation=True
         ).input_ids.to(self.device)
        outputs = self.trainer.model.generate(input_ids=input_ids, max_new_tokens=self.max_output_length, eos_token_id=self.tokenizer.eos_token_id)
        preds = outputs[:, self.max_input_length:].detach().cpu().numpy()
        return self.tokenizer.batch_decode(preds, skip_special_tokens=True)

    def train(self):
        """
        Train the model.
        """

        start_time = time.time()
        self.trainer.train()

        stop_time = time.time()
        print("Training time (seconds): ", stop_time - start_time)

    def get_prediction(self, examples):
        """
        Generate predictions for input examples.

        Args:
            example (str): The input example.

        Returns:
            list: The list of predicted labels.
        """
        inputs = self.tokenizer([instruction_template + example + response_template for example in examples], return_tensors="pt", max_length=self.max_input_length, truncation=True, padding="max_length")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        outputs = self.trainer.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.max_output_length, eos_token_id=self.tokenizer.eos_token_id)
        
        preds = outputs.detach().cpu().numpy()
        return self.tokenizer.batch_decode(preds, skip_special_tokens=True)

    def evaluate(self):
        """
        Evaluate the trained model using the test dataset.

        This function evaluates the trained model by generating predictions
        for the test dataset, saving the predictions to a CSV file, and
        performing evaluation using an NER evaluator.

        Returns:
            None
        """
        evaluator = NEREvaluator(self.model_name)
        self.trainer.model = torch.compile(self.trainer.model)
        self.trainer.model.eval()
        start_time = time.time()
        test_pred = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(self.dataset['test']['input']), self.batch_size * 2)):
                batch_text = self.dataset['test']['input'][i: i + self.batch_size * 2]
                batch_pred = [pred.split(response_template)[1].strip() for pred in self.get_prediction(batch_text)]
                test_pred.extend(batch_pred)
                print(test_pred[-1])
        stop_time = time.time()
        print("Inference time (seconds): ", stop_time - start_time)

        df = pd.DataFrame(list(zip(
            self.dataset['test']['words'],
            self.dataset['test']['tags'],
            self.dataset['test']['input'],
            test_pred
        )), columns=['words', 'tags', 'input', 'pred'])

        df.to_csv(self.model_name.replace("/", "-") + "_test.csv", index=False)
        evaluator.evaluate(df, 'syllable' if self.syllable else 'word')
