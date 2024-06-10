import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bigscience/bloom-560m")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="../data/sentence")
    parser.add_argument("--syllable", action='store_true')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pipeline = NERTrainingPipeline(args)
    pipeline.train()
    pipeline.evaluate()
