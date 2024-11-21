import logging
import sys

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from defenses.deepsight import Deepsight
from defenses.fldetector import FLDetector
from .Attacker_dba import Attacker_dba
from .attacker_d2ba import Attacker_d2ba
from .attacker_iba import Attacker_iba
from .attacker_match_distribution import Attacker_match_distribution
from .attacker_max_feature_map import Attacker_max_feature_map
from .attacker_max_feature_map_generator import Attacker_max_feature_map_generator
from defenses.defense import RFA, WeightDiffClippingDefense, Krum, CRFL, RLR, FoolsGold, FLAME, FLTrust

sys.path.append("../")
import time
import wandb

import torch

import random
import numpy as np
import copy
import os
import seaborn as sns
from .attacker import Attacker
from .aggregator import Aggregator
from math import ceil
import pickle


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FLer:
    def __init__(self, helper):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        self.helper = helper

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attack_sum = 0
        self.aggregator = Aggregator(self.helper)
        self.start_time = time.time()
        self.attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        if self.helper.config.is_poison:
            if self.helper.config.attacker == 'A3FL':
                self.attacker = Attacker(self.helper)
            elif self.helper.config.attacker == 'max-feature-map':
                self.attacker = Attacker_max_feature_map(self.helper)
            elif self.helper.config.attacker == 'max-feature-map_generator':
                self.attacker = Attacker_max_feature_map_generator(self.helper)
            elif self.helper.config.attacker == 'dba':
                self.attacker = Attacker_dba(self.helper)
            elif self.helper.config.attacker == 'iba':
                self.attacker = Attacker_iba(self.helper, 'cuda')
            elif self.helper.config.attacker == 'match_distribution':
                self.attacker = Attacker_match_distribution(self.helper)
            elif self.helper.config.attacker == 'd2ba':
                self.attacker = Attacker_d2ba(self.helper)

            logger.info(f'Attacker: {self.helper.config.attacker}')

        else:
            self.attacker = None
        if self.helper.config.sample_method == 'random_updates':
            self.init_advs()
        if self.helper.config.load_benign_model:  
            if self.helper.config.dirichlet_alpha == 0.9:
                model_path = f'../saved/benign_new/{self.helper.config.model}_{self.helper.config.dataset}_{self.helper.config.agg_method}_{self.helper.config.defense_technique}_{self.helper.config.poison_start_epoch}.pt'
            else:
                model_path = f'../saved/benign_new/{self.helper.config.model}_{self.helper.config.dataset}_{self.helper.config.agg_method}_{self.helper.config.defense_technique}_{self.helper.config.poison_start_epoch}_{self.helper.config.dirichlet_alpha}.pt'
            state_dict = torch.load(
                model_path, map_location='cuda')[
                'model']
            
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('linear.'):
                    new_key = key.replace('linear.', 'fc.')
                else:
                    new_key = key
                new_state_dict[new_key] = value

            
            self.helper.global_model.load_state_dict(new_state_dict)
            loss, acc = self.test_once()
            logger.info(f'Load benign model {model_path}, acc {acc:.3f}')

        self.load_defender() if self.helper.config.defense_technique else None
        
        self.defense = Deepsight(self.helper)
        

        return

    def load_defender(self):
        arguments = self.helper.config  
        if arguments["defense_technique"] == "none":
            self._defender = None
        elif arguments["defense_technique"] in ["norm-clipping", "norm-clipping-adaptive"]:
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "weak-dp":
            self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        elif arguments["defense_technique"] == "krum":
            self._defender = Krum(mode='krum', num_workers=self.helper.config.num_sampled_participants,
                                  num_adv=self.helper.config.num_adversaries)
        elif arguments["defense_technique"] == "multi-krum":
            self._defender = Krum(mode='multi-krum', num_workers=self.helper.config.num_sampled_participants,
                                  num_adv=self.helper.config.num_adversaries)
        elif arguments["defense_technique"] == "rfa":
            self._defender = RFA()
        elif arguments["defense_technique"] == "crfl":
            self._defender = CRFL()
        elif arguments["defense_technique"] == "rlr":
            pytorch_total_params = sum(p.numel() for p in self.helper.global_model.parameters())
            args_rlr = {
                'aggr': 'avg',
                'noise': 0,
                'clip': 0,
                'server_lr': 1,
            }
            theta = 4  
            self._defender = RLR(pytorch_total_params, device='cuda', args=args_rlr, robustLR_threshold=theta)
        elif arguments["defense_technique"] == "foolsgold":
            pytorch_total_params = sum(p.numel() for p in self.helper.global_model.parameters())
            
            
            
            self._defender = FoolsGold(num_clients=self.helper.config.num_sampled_participants,
                                       num_classes=self.helper.num_classes, num_features=pytorch_total_params)
        elif arguments["defense_technique"] == "flame":
            self._defender = FLAME(num_clients=self.helper.config.num_sampled_participants)
        elif arguments["defense_technique"] == "deepsight":
            self._defender = Deepsight(self.helper)
        elif arguments["defense_technique"] == "fltrust":
            
            clean_dataset_size = 300  
            
            indices = list(range(len(self.helper.train_dataset)))
            clean_indices = random.sample(indices, clean_dataset_size)
            root_dataloader = self.helper.get_train(clean_indices)
            self._defender = FLTrust(root_dataloader, self.helper)  
        else:
            raise NotImplementedError("Unsupported defense method!")
        logger.info(f"Defense technique: {arguments['defense_technique']}")
        return self._defender

    def init_advs(self):
        num_updates = self.helper.config.num_sampled_participants * self.helper.config.end_poison_epochs
        num_poison_updates = ceil(self.helper.config.sample_poison_ratio * num_updates)
        updates = list(range(num_updates))
        advs = np.random.choice(updates, num_poison_updates, replace=False)
        logger.info(f'Using random updates, sampled {",".join([str(x) for x in advs])}')
        adv_dict = {}
        for adv in advs:
            epoch = adv // self.helper.config.num_sampled_participants
            idx = adv % self.helper.config.num_sampled_participants
            if epoch in adv_dict:
                adv_dict[epoch].append(idx)
            else:
                adv_dict[epoch] = [idx]
        self.advs = adv_dict

    def calculate_mislead_accuracy(self, not_corrected_labels, not_corrected_labels_orglabels):
        
        not_corrected_labels = np.array(not_corrected_labels)
        not_corrected_labels_orglabels = np.array(not_corrected_labels_orglabels)

        
        not_target_class_indices = not_corrected_labels_orglabels != self.helper.config.target_class

        
        target_class_indices = not_corrected_labels == self.helper.config.target_class

        
        mislead_indices = not_target_class_indices & target_class_indices

        
        mislead_accuracy = np.sum(mislead_indices) / len(not_corrected_labels)

        return mislead_accuracy

    def test_once(self, poison=False, output_distribution=False):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        start_time = time.time()
        model = self.helper.global_model
        model.eval()
        not_corrected_labels = []
        not_corrected_labels_orglabels = []
        misled_count = 0  
        not_target_count = 0  

        with torch.no_grad():
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                org_label = targets.cuda()
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                if poison:
                    not_corrected = pred != self.helper.config.target_class
                    not_corrected_labels.extend(pred[not_corrected].cpu().numpy().flatten())
                    not_corrected_labels_orglabels.extend(org_label[not_corrected].cpu().numpy().flatten())

                    
                    misled = (pred == self.helper.config.target_class) & (org_label != self.helper.config.target_class)
                    misled_count += misled.sum().item()
                    not_target_count += (org_label != self.helper.config.target_class).sum().item()

                num_data += output.size(0)

            acc = 100.0 * (float(correct) / float(num_data))
            loss = total_loss / float(num_data)
            model.train()

            
            
            
            
            
            
            
            
            
            
            
            
            
            

            end_time = time.time()
            logger.info(f"Test Once time: {end_time - start_time:.3f}s")

            
            if poison:
                
                misled_acc = 100.0 * misled_count / not_target_count if not_target_count > 0 else 0
                return loss, acc, misled_acc

            return loss, acc

    def test_local_once(self, model, poison=False):
        model.eval()
        with torch.no_grad():
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        acc = 100.0 * (float(correct) / float(num_data))
        loss = total_loss / float(num_data)
        model.train()
        return loss, acc

    def log_once(self, epoch, loss, acc, bkd_loss, bkd_acc, bkd_misled_acc):

        log_dict = {
            'epoch': epoch,
            'test_acc': acc,
            'test_loss': loss,
            'bkd_acc': bkd_acc,
            'bkd_loss': bkd_loss,
            'bkd_misled_acc': bkd_misled_acc
        }
        if epoch >= 0:
            wandb.log(log_dict, step=epoch)
            

        logger.info('|'.join([f'{k}:{float(log_dict[k]):.3f}' for k in log_dict]))
        self.save_model(epoch, log_dict)

    def save_model(self, epoch, log_dict):
        if epoch % self.helper.config.save_every == 0:
            log_dict['model'] = self.helper.global_model.state_dict()
            if self.helper.config.is_poison:
                pass
            else:
                assert self.helper.config.lr_method == 'linear'
                if self.helper.config.load_benign_model:
                    save_epoch = epoch + self.helper.config.poison_start_epoch
                else:
                    save_epoch = epoch
                if self.helper.config.dirichlet_alpha == 0.9:
                    save_path = f'../saved/benign_new/{self.helper.config.model}_{self.helper.config.dataset}_{self.helper.config.agg_method}_{self.helper.config.defense_technique}_{save_epoch}.pt'
                else:
                    save_path = f'../saved/benign_new/{self.helper.config.model}_{self.helper.config.dataset}_{self.helper.config.agg_method}_{self.helper.config.defense_technique}_{save_epoch}_{self.helper.config.dirichlet_alpha}.pt'
                torch.save(log_dict, save_path)
                logger.info(f'Model saved at {save_path}')

    def save_res(self, accs, asrs):
        log_dict = {
            'accs': accs,
            'asrs': asrs
        }
        atk_method = self.helper.config.attacker_method
        if self.helper.config.sample_method == 'random':
            file_name = f'{self.helper.config.dataset}/{self.helper.config.agg_method}_{atk_method}_r_{self.helper.config.num_adversaries}_{self.helper.config.end_poison_epochs}_ts{self.helper.config.trigger_size}.pkl'
        elif self.helper.config.sample_method == 'attack_rounds_frequency':
            file_name = f'{self.helper.config.dataset}/{self.helper.config.agg_method}_{atk_method}_arf_{self.helper.config.num_adversaries}_{self.helper.config.end_poison_epochs}_ts{self.helper.config.trigger_size}.pkl'
        else:
            raise NotImplementedError
        save_path = os.path.join(f'../saved/res/{file_name}')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        f_save = open(save_path, 'wb')
        pickle.dump(log_dict, f_save)
        f_save.close()
        logger.info(f'results saved at {save_path}')

    def train(self):
        logger.info('Training')
        accs = []
        asrs = []
        self.local_asrs = {}
        
        with logging_redirect_tqdm():
            for epoch in tqdm(range(-2, self.helper.config.epochs), position=0, leave=True):
                self.norm_diff_collector = []
                self.helper.client_models = []
                sampled_participants = self.sample_participants(epoch)
                weight_accumulator, weight_accumulator_by_client = self.train_once(epoch, sampled_participants)
                
                
                
                
                self.defense.update_global_model(weight_accumulator, self.helper.global_model)
                loss, acc = self.test_once()
                if self.helper.config.is_poison:
                    bkd_loss, bkd_acc, bkd_misled_acc = self.test_once(poison=True, output_distribution=True)
                else:
                    bkd_loss = bkd_acc = bkd_misled_acc = 0
                
                self.log_once(epoch, loss, acc, bkd_loss, bkd_acc, bkd_misled_acc)
                accs.append(acc)
                asrs.append(bkd_acc)

                
                
                
                
                
                
                

                
                
                

            if self.helper.config.is_poison:
                self.save_res(accs, asrs)

    def visualize_tsne(self,model, epoch, participant_id, poison=False):
        
        model.eval()
        features = []
        labels = []
        source_class = self.helper.config.visual_source_class
        target_class = self.helper.config.target_class
        poisoned_indices = []
        with torch.no_grad():
            for data, target in self.helper.train_data[participant_id]:
                data, target = data.cuda(), target.cuda()
                if poison:
                    data, target,bkd_indices = self.attacker.poison_input_visual(data, target, eval=False)
                    poisoned_indices.extend(bkd_indices)
                feature = model.features(data)  
                features.append(feature.cpu().numpy())
                labels.append(target.cpu().numpy())


        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        
        source_indices = np.where(labels == source_class)[0]
        poisoned_indices = np.array(poisoned_indices)

        target_indices = np.where(labels == target_class)[0]
        target_indices = np.setdiff1d(target_indices, poisoned_indices)

        

        source_features = features[source_indices]
        target_features = features[target_indices]
        if len(poisoned_indices) == 0:
            poisoned_features = np.zeros((0, features.shape[1]))
        else:
            poisoned_features = features[poisoned_indices]

        source_labels = labels[source_indices]
        target_labels = labels[target_indices]
        if len(poisoned_indices) == 0:
            poisoned_labels = np.zeros(0)
        else:
            poisoned_labels = labels[poisoned_indices]

        
        combined_features = np.concatenate([source_features, target_features, poisoned_features], axis=0)
        combined_labels = np.concatenate([source_labels, target_labels, poisoned_labels], axis=0)

        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(combined_features)

        
        df = pd.DataFrame(features_tsne, columns=['Component 1', 'Component 2'])
        df['Label'] = combined_labels
        df['Type'] = [
            'Source Class' if label == source_class else 'Target Class' if label == target_class else 'Poisoned Samples'
            for label in combined_labels]

        
        csv_filename = f'../saved/tsne/tsne_epoch_{epoch}_participant_{participant_id}_poison{poison}.csv'
        df.to_csv(csv_filename, index=False)

        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='Component 1', y='Component 2', hue='Type', style='Type', data=df,
                        palette=['blue', 'green', 'red'], s=100)
        plt.title(f't-SNE Visualization at Epoch {epoch} Poisoned: {poison}')
        plt.savefig(f'tsne_epoch_{epoch}_participant_{participant_id}_{poison}.png')
        print(f'tsne_epoch_{epoch}_participant_{participant_id}{poison}.png saved.')
        plt.close()
        model.train()

    def train_once(self, epoch, sampled_participants):
        
        start_time = time.time()
        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator2 = self.create_weight_accumulator()
        weight_accumulator_by_client = []
        weight_accumulator_by_client2 = []
        
        client_count = 0
        attacker_idxs = []
        global_model_copy = self.create_global_model_copy()
        local_asr = []
        num_data_points = []  
        norm_diff_collector = []  

        
        
        
        
        
        
        
        
        
        

        for participant_id in sampled_participants:
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            model.name = f'participant_{participant_id}'
            model.train()
            if not self.if_adversary(epoch, participant_id, sampled_participants):
                self.train_benign(participant_id, model, epoch)
                if self.helper.config.defense_technique == "norm-clipping-adaptive":
                    norm_diff = self.calc_norm_diff(model, self.helper.global_model, epoch, participant_id,
                                                    mode="normal")
                    norm_diff_collector.append(norm_diff)
            else:
                attacker_idxs.append(client_count)
                time_start = time.time()
                self.train_malicious(participant_id, model, epoch)
                if self.helper.config.defense_technique == "norm-clipping-adaptive":
                    norm_diff = self.calc_norm_diff(model, self.helper.global_model, epoch, participant_id, mode="bad")
                    norm_diff_collector.append(norm_diff)
                time_end = time.time()
                logger.info(f"Attacker {participant_id} training time: {time_end - time_start:.3f}s")

            
            
            self.helper.client_models.append(copy.deepcopy(model))
            num_data_points.append(len(
                self.helper.train_data[participant_id]))  

            client_count += 1
        
        end_time = time.time()
        if len(attacker_idxs) > 0:
            logger.info(
                f'Epoch {epoch}, poisoning by {attacker_idxs}, attack sum {self.attack_sum}, time {end_time - start_time:.3f}s.')
        else:
            logger.info(f'Epoch {epoch}, no adversary, time {end_time - start_time:.3f}s.')

        total_num_dps_per_round = sum(num_data_points)
        net_freq = [num_data_points[i] / total_num_dps_per_round for i in
                    range(self.helper.config.num_sampled_participants)]
        net_freq = [0.1 for i in range(self.helper.config.num_sampled_participants)]

        pytorch_total_params = sum(p.numel() for p in self.helper.global_model.parameters())
        
        memory_size = 0
        delta_memory = np.zeros((self.helper.config.num_sampled_participants, pytorch_total_params, memory_size))
        summed_deltas = np.zeros((self.helper.config.num_sampled_participants, pytorch_total_params))
        
        logger.info(f"Conduct defense technique: {self.helper.config.defense_technique}")
        if self.helper.config.defense_technique == "foolsgold":
            flatten_net_avg = self.flatten_model(copy.deepcopy(self.helper.global_model))
            delta = np.zeros((self.helper.config.num_sampled_participants, pytorch_total_params))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            for global_client_indx, local_net in enumerate(self.helper.client_models):
                flatten_local_model = self.flatten_model(local_net)
                local_update_diff = flatten_local_model - flatten_net_avg
                local_update_diff = local_update_diff.detach().cpu().numpy()
                delta[global_client_indx, :] = local_update_diff
                if np.linalg.norm(delta[global_client_indx, :]) > 1:
                    delta[global_client_indx, :] = delta[global_client_indx, :] / np.linalg.norm(
                        delta[global_client_indx, :])
                if memory_size > 0:
                    delta_memory[global_client_indx, :, epoch % memory_size] = delta[
                                                                               global_client_indx, :]

            if memory_size > 0:
                summed_deltas = np.sum(delta_memory, axis=2)
            else:
                summed_deltas += delta

            wv = self._defender.exec(delta=delta, summed_deltas=summed_deltas, net_avg=self.helper.global_model,
                                     r=epoch, device='cuda')

            
            net_freq = wv
            logger.info(f"net_freq: {net_freq}")
        elif self.helper.config.defense_technique == "none":
            pass
        elif self.helper.config.defense_technique == "norm-clipping":
            for net_idx, net in enumerate(self.helper.client_models):
                self._defender.exec(client_model=net, global_model=self.helper.global_model)
        elif self.helper.config.defense_technique == "norm-clipping-adaptive":
            self._defender.norm_bound = np.mean(norm_diff_collector)
            for net_idx, net in enumerate(self.helper.client_models):
                self._defender.exec(client_model=net, global_model=self.helper.global_model)
        elif self.helper.config.defense_technique == "weak-dp":
            for net_idx, net in enumerate(self.helper.client_models):
                self._defender.exec(client_model=net, global_model=self.helper.global_model)
        elif self.helper.config.defense_technique == "krum":
            logger.info("start performing krum...")
            self.helper.client_models, net_freq = self._defender.exec(client_models=self.helper.client_models,
                                                                      num_dps=num_data_points,
                                                                      g_user_indices=sampled_participants,
                                                                      device='cuda')
        elif self.helper.config.defense_technique == "multi-krum":
            self.helper.client_models, net_freq = self._defender.exec(client_models=self.helper.client_models,
                                                                      num_dps=num_data_points,
                                                                      g_user_indices=sampled_participants,
                                                                      device='cuda')
        elif self.helper.config.defense_technique == 'rlr':
            logger.info(f"num_data_points: {num_data_points}")
            for net_idx, net in enumerate(self.helper.client_models):
                weight_accumulator2, single_wa = self.update_weight_accumulator(net, weight_accumulator2)
                weight_accumulator_by_client2.append(single_wa)

            self.helper.client_models, net_freq = self._defender.exec(
                weight_accumulator_by_client=weight_accumulator_by_client2,
                num_dps=num_data_points,
                global_model=copy.deepcopy(
                    self.helper.global_model))
        elif self.helper.config.defense_technique == "rfa":
            self.helper.client_models, net_freq = self._defender.exec(client_models=self.helper.client_models,
                                                                      net_freq=net_freq, maxiter=500, eps=1e-5,
                                                                      ftol=1e-7, device='cuda')
        elif self.helper.config.defense_technique == "crfl":
            cp_net_avg = copy.deepcopy(self.helper.global_model)
            self.helper.client_models, net_freq = self._defender.exec(target_model=cp_net_avg, epoch=epoch,
                                                                      sigma_param=0.01,
                                                                      dataset_name=self.helper.config.dataset,
                                                                      device='cuda')
        elif self.helper.config.defense_technique == "flame":
            
            local_models = [model.state_dict() for model in self.helper.client_models]
            global_model = copy.deepcopy(self.helper.global_model)
            for net_idx, net in enumerate(self.helper.client_models):
                weight_accumulator2, single_wa = self.update_weight_accumulator(net, weight_accumulator2)
                weight_accumulator_by_client2.append(single_wa)

            
            self.helper.client_models, net_freq = self._defender.exec(
                local_model=local_models,
                update_params=weight_accumulator_by_client2,
                global_model=global_model
            )

            
            
            
            
            
        elif self.helper.config.defense_technique == "deepsight":

            for net_idx, net in enumerate(self.helper.client_models):
                _, single_wa = self.update_weight_accumulator(net, weight_accumulator2)
                weight_accumulator_by_client2.append(single_wa)

            weight_accumulator = self._defender.exec(weight_accumulator, self.helper.global_model,
                                                     weight_accumulator_by_client2)

            return weight_accumulator, weight_accumulator_by_client2

        elif self.helper.config.defense_technique == "fltrust":
            logger.info("start performing FLTrust...")

            
            
            
            self.helper.client_models, net_freq = self._defender.exec(client_models=self.helper.client_models,
                                                                      central_model=copy.deepcopy(
                                                                          self.helper.global_model),
                                                                      lr=self.get_lr(epoch))
        else:
            raise NotImplementedError("Unsupported defense method!")

        
        for net_idx, net in enumerate(self.helper.client_models):
            weight_accumulator, single_wa = self.update_weight_accumulator(net, weight_accumulator,
                                                                           net_freq[net_idx])  
            weight_accumulator_by_client.append(single_wa)

        return weight_accumulator, weight_accumulator_by_client

    def flatten_model(self, model):
        def flatten_tensors(tensors):
            """
            Reference: https://github.com/facebookresearch/stochastic_gradient_push
            Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
            same dense type.
            Since inputs are dense, the resulting tensor will be a concatenated 1D
            buffer. Element-wise operation on this buffer will be equivalent to
            operating individually.
            Arguments:
                tensors (Iterable[Tensor]): dense tensors to flatten.
            Returns:
                A 1D buffer containing input tensors.
            """
            if len(tensors) == 1:
                return tensors[0].view(-1).clone()
            flat = torch.cat([t.view(-1) for t in tensors], dim=0)
            return flat

        ten = torch.cat([flatten_tensors(i) for i in model.parameters()])
        return ten

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    

    def if_adversary(self, epoch, participant_id, sampled_participants):
        if self.helper.config.is_poison and epoch < self.helper.config.end_poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random' and participant_id < self.helper.config.num_adversaries:
                return True
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    for idx in self.advs[epoch]:
                        if sampled_participants[idx] == participant_id:
                            return True
            elif self.helper.config.sample_method == 'attack_rounds_frequency':
                if participant_id < self.helper.config.num_adversaries:
                    return True
        else:
            return False

    def create_local_model_copy(self, model):
        model_copy = dict()
        for name, param in model.named_parameters():
            model_copy[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
        return model_copy

    def create_global_model_copy(self):
        global_model_copy = dict()
        for name, param in self.helper.global_model.named_parameters():
            global_model_copy[name] = self.helper.global_model.state_dict()[name].clone().detach().requires_grad_(False)
        return global_model_copy

    def create_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.helper.global_model.state_dict().items():
            
            if name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data).float()
        return weight_accumulator

    def update_weight_accumulator(self, client_model, weight_accumulator, net_freq=1):
        single_weight_accumulator = dict()
        for name, data in client_model.state_dict().items():
            if name == 'decoder.weight' or '__' in name:
                continue

            diff = (data - self.helper.global_model.state_dict()[name]).float()
            
            weight_accumulator[name].add_(diff * net_freq)
            
            single_weight_accumulator[name] = diff
        return weight_accumulator, single_weight_accumulator

    
    
    
    
    
    
    
    

    def train_benign(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.helper.config.momentum,
                                    weight_decay=self.helper.config.decay)

        for internal_epoch in range(self.helper.config.retrain_times):
            total_loss = 0.0
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = self.criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def scale_up(self, model, curren_num_adv):
        clip_rate = 2 / curren_num_adv
        for key, value in model.state_dict().items():
            
            if key == 'decoder.weight' or '__' in key:
                continue
            target_value = self.helper.global_model.state_dict()[key]
            new_value = target_value + (value - target_value) * clip_rate

            model.state_dict()[key].copy_(new_value)
        return model

    def train_malicious(self, participant_id, model, epoch):
        time_start = time.time()
        self.attacker.update_trigger(model, self.helper.train_data[participant_id], 'inner', participant_id, epoch)
        time_update_trigger = time.time()
        logger.info(f"Attacker {participant_id} update trigger time: {time_update_trigger - time_start:.3f}s")

        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.helper.config.momentum,
                                    weight_decay=self.helper.config.decay)
        clean_model = copy.deepcopy(model)
        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            total_loss = 0.0
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = self.attacker.poison_input(inputs, labels)
                output = model(inputs)
                loss = self.attacker_criterion(output, labels)
                if self.helper.config.attacker == 'dba':
                    distance_loss = self.attacker.model_dist_norm_var(model, clean_model)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        time_end = time.time()
        logger.info(f"Attacker {participant_id} training time: {time_end - time_start:.3f}s")
        self.visualize_tsne(model,epoch, participant_id, poison=True)
        self.visualize_tsne(model, epoch, participant_id, poison=False)

    def get_lr(self, epoch):
        if self.helper.config.lr_method == 'exp':
            tmp_epoch = epoch
            if self.helper.config.is_poison and self.helper.config.load_benign_model:
                tmp_epoch += self.helper.config.poison_start_epoch
            lr = self.helper.config.lr * (self.helper.config.gamma ** tmp_epoch)
        elif self.helper.config.lr_method == 'linear':
            if self.helper.config.is_poison or epoch > 1900:
                lr = 0.002
            else:
                lr_init = self.helper.config.lr
                target_lr = self.helper.config.target_lr
                
                if epoch <= self.helper.config.epochs / 2.:
                    lr = epoch * (target_lr - lr_init) / (self.helper.config.epochs / 2. - 1) + lr_init - (
                            target_lr - lr_init) / (self.helper.config.epochs / 2. - 1)
                else:
                    lr = (epoch - self.helper.config.epochs / 2) * (-target_lr) / (
                            self.helper.config.epochs / 2) + target_lr

                if lr <= 0.002:
                    lr = 0.002
                
                
        return lr

    def calc_norm_diff(self, gs_model, vanilla_model, epoch, fl_round, mode="bad"):
        norm_diff = 0
        for p_index, p in enumerate(gs_model.parameters()):
            norm_diff += torch.norm(
                list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
        norm_diff = torch.sqrt(norm_diff).item()
        if mode == "bad":
            
            logger.info(
                "===> ND `|w_bad-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
        elif mode == "normal":
            logger.info("===> ND `|w_normal-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round,
                                                                                                     norm_diff))
        elif mode == "avg":
            logger.info(
                "===> ND `|w_avg-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))

        return norm_diff

    def sample_participants(self, epoch):
        if self.helper.config.sample_method in ['random', 'random_updates']:
            sampled_participants = random.sample(
                range(self.helper.config.num_total_participants),
                self.helper.config.num_sampled_participants)
        elif self.helper.config.sample_method == 'fix-rate':
            start_index = (
                                  epoch * self.helper.config.num_sampled_participants) % self.helper.config.num_total_participants
            sampled_participants = list(range(start_index, start_index + self.helper.config.num_sampled_participants))
        elif self.helper.config.sample_method == 'attack_rounds_frequency':
            if self.helper.config.attack_rounds and epoch in self.helper.config.attack_rounds:
                
                sampled_participants = self.sample_attack_participants()
            elif epoch % self.helper.config.attack_frequency == 0:
                
                sampled_participants = self.sample_attack_participants()
            else:
                
                sampled_participants = random.sample(
                    range(self.helper.config.num_adversaries, self.helper.config.num_total_participants),
                    self.helper.config.num_sampled_participants)
        else:
            raise NotImplementedError
        assert len(sampled_participants) == self.helper.config.num_sampled_participants
        return sampled_participants

    def sample_attack_participants(self):
        
        sampled_participants = random.sample(
            range(self.helper.config.num_adversaries, self.helper.config.num_total_participants),
            self.helper.config.num_sampled_participants)
        
        for i in range(self.helper.config.num_adversaries):
            sampled_participants[i] = i
        return sampled_participants

    def copy_params(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            layer.data = copy.deepcopy(target_params_variables[name])
