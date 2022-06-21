from src.utils.loggers import *
from src.utils.utils import AverageMeterSet, STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from src.models import model_factory

import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only=False):
        self.user2seq = user2seq
        self.args = args
        self.device = args.device
        self.model = model.to_device(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer(embed_only)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.ckpt_root = ckpt_root
        self.writer, self.train_loggers, self.val_loggers, self.test_loggers = self._create_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers, self.test_loggers)

        self.cases = {}

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        accum_iter = 0
        best_target = 0
        best_num = 0
        self.validate_test(0, accum_iter, mode='test')

        for epoch in range(self.num_epochs):
            if best_num >= 5:
                break
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            if (epoch % self.args.verbose) == (self.args.verbose - 1):
                target = self.validate_test(epoch, accum_iter, mode='val', target=self.args.best_metric)
                self.validate_test(epoch, accum_iter, mode='test')
                best_num = best_num + 1 if target < best_target else 1
                best_target = max(target, best_target)
        
    def train_one_epoch(self, epoch, accum_iter):

        self.model.train()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader, ncols=100)

        for batch_idx, batch in enumerate(tqdm_dataloader):

            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.5f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            log_data = {
                'state_dict': (self._create_state_dict(skip=True)),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }

        log_data.update(average_meter_set.averages())
        self.logger_service.log_train(log_data)

        torch.cuda.empty_cache()

        return accum_iter

    def validate_test(self, epoch, accum_iter, mode="val", target=None):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader) if mode == "val" else tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, mode)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] +\
                                      ['MRR'] + ['AUC'] + ['loss']
                                    #   ['NDCG_meta@%d' % k for k in self.metric_ks[:3]] +\
                                    #   ['Recall_meta@%d' % k for k in self.metric_ks[:3]]+\
                                     

                description = mode.upper() + " " + ': , '.join(s + ' {:.5f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R').replace('MRR', 'M')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())

        torch.cuda.empty_cache()

        if mode == "val":
            self.logger_service.log_val(log_data)
        else:
            self.logger_service.log_test(log_data)

        if target is not None:
            return average_meter_set[target].avg

    def test(self):
        print('Test best model with test set!')

        del self.model
        gc.collect()

        best_model = torch.load(os.path.join(self.ckpt_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model = model_factory(self.args)
        self.model.load_state_dict(best_model)
        self.mode = self.model.to_device(self.device)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        # decode_dict = {'seqs': [], 'candidates': [], 'meta_seqs': [], 'meta_candidates': [], 'ranks': [], 'meta_ranks': []}

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)

            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch, mode='final_test')#, seqs, candidates, meta_seqs, meta_candidates, ranks, meta_ranks = self.calculate_metrics(batch, mode='final_test')
                
                # decode_dict['seqs'] += seqs
                # decode_dict['candidates'] += candidates
                # decode_dict['meta_seqs'] += meta_seqs
                # decode_dict['meta_candidates'] += meta_candidates
                # decode_dict['ranks'] += ranks
                # decode_dict['meta_ranks'] += meta_ranks

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]] +\
                                      ['MRR'] + ['AUC'] + ['loss']
                                    #   ['NDCG_meta@%d' % k for k in self.metric_ks[:3]] +\
                                    #   ['Recall_meta@%d' % k for k in self.metric_ks[:3]]+\
                                      

                description = 'FINAL TEST: ' + ', '.join(s + ' {:.5f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R').replace('MRR', 'M')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.ckpt_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)

            # with open('Decode_results.json', 'w', encoding='utf8') as f:
            #     json.dump(decode_dict, f)
            
    def _create_optimizer(self, embed_only=False):
        args = self.args
        if embed_only:
            named_parameters = [(n, p) for n, p in self.model.named_parameters() if 'embed' in n]
        else:
            named_parameters = self.model.named_parameters()

        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if args.optimizer.lower() == 'adam':
            
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.98))
            # return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.ckpt_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
            # val_loggers.append(
            #     MetricGraphPrinter(writer, key='NDCG_meta@%d' % k, graph_name='NDCG_meta@%d' % k, group_name='Validation'))
            # val_loggers.append(
            #     MetricGraphPrinter(writer, key='Recall_meta@%d' % k, graph_name='Recall_meta@%d' % k, group_name='Validation'))

        val_loggers.append(MetricGraphPrinter(writer, key='loss', graph_name='loss', group_name='Validation'))
        val_loggers.append(MetricGraphPrinter(writer, key='MRR', graph_name='MRR', group_name='Validation'))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))

        test_loggers = []
        for k in self.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Test'))
            test_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Test'))
            # test_loggers.append(
            #     MetricGraphPrinter(writer, key='NDCG_meta@%d' % k, graph_name='NDCG_meta@%d' % k, group_name='Test'))
            # test_loggers.append(
            #     MetricGraphPrinter(writer, key='Recall_meta@%d' % k, graph_name='Recall_meta@%d' % k, group_name='Test'))
        test_loggers.append(MetricGraphPrinter(writer, key='loss', graph_name='loss', group_name='Test'))
        test_loggers.append(MetricGraphPrinter(writer, key='MRR', graph_name='MRR', group_name='Test'))
        
        return writer, train_loggers, val_loggers, test_loggers

    def _create_state_dict(self, skip=False):
        if skip:
            return None
        return {
            STATE_DICT_KEY: self.model.device_state_dict(),
        }
