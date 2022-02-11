import glob
import os
import abc_sr.evogym_utils as evoutils

import torch
import torch.nn as nn

from ppo.envs import VecNormalize

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):

    try:
        os.makedirs(log_dir)
    except:
        pass
        # files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        # for f in files:
        #     os.remove(f)


def init_input(obs, robot_structure, robot_shape, voxel_ids,  n_proc, device):

    mass_matrix = evoutils.mass_pos_matrix(robot_structure[0]).to(device)
    sa_matrix = [evoutils.init_state_action_matrix(robot_structure[0], 16, 1).to(device) for _ in range(n_proc)]

    #print(torch.sum(mass_matrix), obs.shape)
    #if(torch.sum(mass_matrix).item() != obs.shape[-1]):
    #    print("Bad dims\n", mass_matrix, "\n------------\n", robot_structure, "\n------------\n", obs[0], "\n------------\n")
    obs_mat = [evoutils.mass_obs_matrix(mass_matrix, ob, device).to(device) for ob in obs]

    voxel_input = [evoutils.get_voxel_input(robot_shape, obs_i, sa_i, voxel_ids).to(device) for obs_i, sa_i in zip(obs_mat, sa_matrix)]
    voxel_input = torch.cat(voxel_input)

    return mass_matrix, sa_matrix, obs_mat, voxel_input

def update_input(obs, obs_mat, mass_matrix, sa_matrix, action, robot_shape, voxel_ids, n_proc, n_actuators, device):

    sa_pairs = torch.vstack([evoutils.get_voxel_obs(ob_mat, voxel_ids) for ob_mat in obs_mat])
    sa_pairs = torch.hstack([sa_pairs, action]).reshape((n_proc, n_actuators, -1)).to(device)

    obs_mat = [evoutils.mass_obs_matrix(mass_matrix, obs[i], device).to(device) for i in range(n_proc)]
    [evoutils.update_state_action_matrix(sa_i, sa, voxel_ids) for sa, sa_i in zip(sa_pairs, sa_matrix)]

    voxel_input = torch.cat([evoutils.get_voxel_input(robot_shape, obs_i, sa_i, voxel_ids) for obs_i, sa_i in
                             zip(obs_mat, sa_matrix)]).to(device)

    return obs_mat, voxel_input
