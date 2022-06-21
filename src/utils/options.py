from src.datasets import DATASETS
from src.dataloaders import DATALOADERS
from src.models import MODELS
from src.trainers import TRAINERS

import argparse

parser = argparse.ArgumentParser(description='SASRec')

################
# Test
################
parser.add_argument('--load_pretrained_weights', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='item', choices=DATASETS.keys())
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--data_path', type=str, default='data/ml-1m')
parser.add_argument('--use_pretrained_vectors', action='store_true', help='If setting, use product2vector, otherwise, train from scratch.')

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='sasrec', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

################
# NegativeSampler
################
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=98765)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='sasrec_sample', choices=TRAINERS.keys())
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')

# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD','Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Adam Epsilon')

# training #
parser.add_argument('--verbose', type=int, default=10)
# training on large gpu #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')

# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@5', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='sasrec', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=0)
# Transformer Blocks #
parser.add_argument('--trm_max_len', type=int, default=50, help='Length of sequence for bert')
parser.add_argument('--trm_hidden_dim', type=int, default=128, help='Size of hidden vectors (d_model)')
parser.add_argument('--trm_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--trm_num_heads', type=int, default=1, help='Number of heads for multi-attention')
parser.add_argument('--trm_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
parser.add_argument('--trm_att_dropout', type=float, default=0.2, help='Dropout probability to use throughout the attention scores')
parser.add_argument('--local_num_heads', type=int, default=3, help='Local att heads num')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')
parser.add_argument('--eval_all', action='store_true')
