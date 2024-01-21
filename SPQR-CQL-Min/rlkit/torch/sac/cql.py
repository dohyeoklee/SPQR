from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd

#=============================================================================================
def build_random_matrix(q_list):
    batch_size = q_list[0].size(dim=0)
    N = len(q_list)
    mat_dim = int((np.sqrt(1+8*N)-1)/2)
    len_idxs = int((mat_dim**2+mat_dim)/2)
    expand_q_list = []
    upper_tri_pointer = 0
    lower_tri_pointer = 0
    filling_pointer = mat_dim
    upper_idx = 0
    lower_idx = 1
    while upper_idx < len_idxs:
        if upper_tri_pointer < filling_pointer:
            expand_q_list.append(q_list[upper_idx])
            upper_idx += 1
            upper_tri_pointer += 1
        else:
            upper_tri_pointer = 0
            filling_pointer -= 1
            if filling_pointer < 0:
                break
            lower_tri_pointer += 1
            _pointer = 1
            expand_q_list.append(q_list[lower_idx])
            _lower_pointer = lower_idx
            while _pointer < lower_tri_pointer:
                _lower_pointer += mat_dim-_pointer
                expand_q_list.append(q_list[_lower_pointer])
                _pointer += 1
            lower_idx += 1
    q_tensor = torch.squeeze(torch.stack(expand_q_list))
    q_mean = torch.mean(q_tensor,dim=0,keepdim=True)
    q_std = torch.std(q_tensor,dim=0,keepdim=True)
    q_tensor = (q_tensor - q_mean) / q_std
    q_tensor = q_tensor.view(mat_dim,mat_dim,batch_size)
    return q_tensor

def get_wishart_eigen(q_tensor):
    N = q_tensor.shape[1]
    q_tensor = torch.squeeze(q_tensor).transpose(2,0)*1/np.sqrt(N)
    eig_val_tensor = torch.linalg.eigvalsh(q_tensor)
    eig_val_tensor.requires_grad_(True)
    return eig_val_tensor

def get_wishart_loss(eig_val_tensor):
    N = eig_val_tensor.shape[1]
    eps = 0.01
    rho = 0.5
    wig_pdf = wigner_semi_pdf(eig_val_tensor,eps,rho)
    log_dis = -torch.log(N*wig_pdf) / N
    kl_dis = torch.sum(log_dis,dim=1)
    rmt_loss = torch.mean(kl_dis,dim=0)
    return rmt_loss

def wigner_semi_pdf(eig_val,eps,rho):
    in_val_tensor = torch.sqrt(4-torch.pow(eig_val,2)*(abs(eig_val)<2))/(2*np.pi) 
    out_val_tensor = torch.full_like(eig_val,eps)*(abs(eig_val)>=2)
    return rho*in_val_tensor + (1-rho)*out_val_tensor

#=============================================================================================

class CQLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            qfs,
            target_qf1,
            target_qf2,
            target_qfs,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            num_qs=2,

            #my code
            beta=-1.0,
            device=1,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.qfs = qfs
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.target_qfs = target_qfs
        self.soft_target_tau = soft_target_tau
        self.beta = beta #my code
        self.device = device #my code

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item() 
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.qfs_optimizer = [optimizer_class(qf.parameters(),lr=qf_lr,) for qf in self.qfs]

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start
        
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1
        
        self.num_qs = num_qs

        ## min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        # For implementation on the 
        self.discrete = False
    
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp, reparameterize=True, return_log_prob=True,
        )
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        if self.num_qs > 1:
            q2_pred = self.qf2(obs, actions)
            if len(self.qfs) > 0:
                qs_pred = [qf(obs, actions) for qf in self.qfs]
        
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        if not self.max_q_backup:
            if self.num_qs == 1:
                target_q_values = self.target_qf1(next_obs, new_next_actions)
            else:
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                )
                if len(self.qfs) > 0:
                    target_qs_pred = [target_qf(next_obs, new_next_actions) for target_qf in self.target_qfs]
                    target_qs_pred_compute = [self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),*target_qs_pred]
                    q_prediction_next_cat = torch.cat(target_qs_pred_compute, 1)
                    #print(q_prediction_next_cat.shape)
                    target_q_values, _ = torch.min(q_prediction_next_cat,dim=1,keepdim=True)
                    #print(target_q_values.shape)
            
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        
        if self.max_q_backup:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
            if len(self.qfs) > 0:
                target_qs_pred = [self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf).max(1)[0].view(-1, 1) \
                for target_qf in self.target_qfs]
                target_qs_pred_compute = [target_qf1_values,target_qf2_values,*target_qs_pred]
                q_prediction_next_cat = torch.cat(target_qs_pred_compute, 1)
                #print(q_prediction_next_cat.shape)
                target_q_values, _ = torch.min(q_prediction_next_cat,dim=1,keepdim=True)
            else:
                target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
            
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        if self.num_qs > 1:
            qf2_loss = self.qf_criterion(q2_pred, q_target)
            if len(self.qfs) > 0:
                qfs_loss = [self.qf_criterion(q_pred, q_target) for q_pred in qs_pred]
        
        ## add CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).cuda() # .cuda()
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        if len(self.qfs) > 0:
            qs_rand = [self._get_tensor_values(obs, random_actions_tensor, network=qf) for qf in self.qfs]
            qs_curr_actions = [self._get_tensor_values(obs, curr_actions_tensor, network=qf) for qf in self.qfs]
            qs_next_actions = [self._get_tensor_values(obs, new_curr_actions_tensor, network=qf) for qf in self.qfs]
            cat_qs = [torch.cat([q_rand, q_pred.unsqueeze(1), q_next_actions, q_curr_actions], 1) \
            for q_rand,q_pred,q_next_actions,q_curr_actions in zip(qs_rand,qs_pred,qs_next_actions,qs_curr_actions)]
            std_qs = [torch.std(cat_q, dim=1) for cat_q in cat_qs]

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            if len(self.qfs) > 0:
                cat_qs = [torch.cat([q_rand - random_density, q_next_actions - new_log_pis.detach(), q_curr_actions - curr_log_pis.detach()], 1) \
                for q_rand,q_next_actions,q_curr_actions in zip(qs_rand,qs_next_actions,qs_curr_actions)]
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
        
        if len(self.qfs) > 0:
            min_qfs_loss = [torch.logsumexp(cat_q / self.temp, dim=1,).mean() * self.min_q_weight * self.temp for cat_q in cat_qs]
            min_qfs_loss = [min_qf_loss - q_pred.mean() * self.min_q_weight for min_qf_loss,q_pred in zip(min_qfs_loss,qs_pred)]

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)
            min_qfs_loss = [alpha_prime * (min_qf_loss - self.target_action_gap) for min_qf_loss in min_qfs_loss]

            self.alpha_prime_optimizer.zero_grad()
            #alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5
            min_qs_loss_sum = sum(min_qfs_loss)
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss - min_qs_loss_sum)*1/self.num_qs
            #alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        if len(self.qfs) > 0:
            qfs_loss = [qf_loss + min_qf_loss for qf_loss,min_qf_loss in zip(qfs_loss,min_qfs_loss)]

        ## add SPQR
        if self.beta > 0 and len(self.qfs) > 0:
            #next_actions = new_next_actions.detach().clone().cpu().to(self.device)
            if not self.max_q_backup:
                qs_pred = [qf(next_obs, new_next_actions) for qf in self.qfs]
                qs_pred_list = [self.qf1(next_obs, new_next_actions),
                    self.qf2(next_obs, new_next_actions),*qs_pred]
            if self.max_q_backup:
                qs_pred = [
                    self._get_tensor_values(next_obs, next_actions_temp, network=qf).max(1)[0].view(-1, 1)
                    for qf in self.qfs
                ]
                qs_pred_list = [qf1_values,qf2_values,*qs_pred]
            
            q_tensor = build_random_matrix(qs_pred_list)
            eig_val = get_wishart_eigen(q_tensor)
            rmt_loss = get_wishart_loss(eig_val)
            qf1_loss = qf1_loss + self.beta*rmt_loss
            qf2_loss = qf2_loss + self.beta*rmt_loss
            qfs_loss = [qf_loss + self.beta*rmt_loss for qf_loss in qfs_loss]


        """
        Update networks
        """
        # Update the Q-functions iff 
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        #self.qf1_optimizer.step()

        if self.num_qs > 1:
            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            #self.qf2_optimizer.step()
            if len(self.qfs) > 0:
                for i in range(len(self.qfs)):
                    self.qfs_optimizer[i].zero_grad()
                    qfs_loss[i].backward(retain_graph=True)
                    #self.qfs_optimizer[i].step()
        
        self.qf1_optimizer.step()
        if self.num_qs > 1:
            self.qf2_optimizer.step()
            if len(self.qfs) > 0:
                for i in range(len(self.qfs)):
                    self.qfs_optimizer[i].step()
        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if self.num_qs == 1:
            q_new_actions = self.qf1(obs, new_obs_actions)
        else:
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )
            if len(self.qfs) > 0:
                qs_value = [qf(obs, new_obs_actions) for qf in self.qfs]
                qs_new_compute = [self.qf1(obs, new_obs_actions),self.qf2(obs, new_obs_actions),*qs_value]
                q_cat = torch.cat(qs_new_compute, 1)
                q_new_actions, _ = torch.min(q_cat,dim=1,keepdim=True)

        policy_loss = (alpha*log_pi - q_new_actions).mean()

        if self._current_epoch < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.log_prob(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        if self.num_qs > 1:
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
            for i in range(len(self.qfs)):
                ptu.soft_update_from_to(
                self.qfs[i], self.target_qfs[i], self.soft_target_tau
            )
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            if self.num_qs > 1:
                self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))

            if not self.discrete:
                self.eval_statistics['Std QF1 values'] = np.mean(ptu.get_numpy(std_q1))
                self.eval_statistics['Std QF2 values'] = np.mean(ptu.get_numpy(std_q2))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 in-distribution values',
                    ptu.get_numpy(q1_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 in-distribution values',
                    ptu.get_numpy(q2_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 random values',
                    ptu.get_numpy(q1_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 random values',
                    ptu.get_numpy(q2_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 next_actions values',
                    ptu.get_numpy(q1_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 next_actions values',
                    ptu.get_numpy(q2_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'actions', 
                    ptu.get_numpy(actions)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'rewards',
                    ptu.get_numpy(rewards)
                ))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.num_qs > 1:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            
            if self.with_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(min_qf2_loss).mean()
                self.eval_statistics['threshold action gap'] = self.target_action_gap
                self.eval_statistics['alpha prime loss'] = alpha_prime_loss.item()
            
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            *self.qfs,
            self.target_qf1,
            self.target_qf2,
            *self.target_qfs,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )

