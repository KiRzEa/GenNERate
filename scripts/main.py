import argparse
from pipeline import NERTrainingPipeline
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bigscience/bloom-560m")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="../data/sentence")
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--syllable", action='store_true')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pipeline = NERTrainingPipeline(args)
    pipeline.train()
    pipeline.evaluate()
