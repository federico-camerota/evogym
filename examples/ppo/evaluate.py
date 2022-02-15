import numpy as np
import torch
from ppo import utils
from ppo.envs import make_vec_envs
from ppo.utils import init_input, update_input

import abc_sr.evogym_utils as evoutils

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

def evaluate(
    num_evals,
    actor_critic,
    obs_rms,
    env_name,
    robot_structure,
    seed,
    num_processes,
    eval_log_dir,
    device, n_actuators):

    num_processes = min(num_processes, num_evals)

    eval_envs = make_vec_envs(env_name, robot_structure, seed + num_processes, num_processes,
                              None, eval_log_dir, device, False)

    #vec_norm = utils.get_vec_normalize(eval_envs)
    #if vec_norm is not None:
    #    vec_norm.eval()
    #    vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    robot_structure = (torch.Tensor(robot_structure[0]), torch.Tensor(robot_structure[1]))
    robot_shape = robot_structure[0].shape
    n_proc = num_processes
    voxel_ids = evoutils.actuators_ids(robot_structure[0])
    n_actuators = len(voxel_ids[0])

    #mass_matrix, sa_matrix, obs_mat, voxel_input = init_input(obs, robot_structure, robot_shape, voxel_ids,  n_proc, device)
    voxel_masses, sa_matrix, voxel_input = init_input(obs, robot_structure, robot_shape, voxel_ids, n_proc, device)


    eval_recurrent_hidden_states = torch.zeros(
        num_processes*n_actuators, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes*n_actuators, 1, device=device)

    while len(eval_episode_rewards) < num_evals:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                voxel_input,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        actions = action.reshape((n_proc, -1))

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(actions)

        #obs_mat, voxel_input = update_input(obs, obs_mat, mass_matrix, sa_matrix, action, robot_shape, voxel_ids, n_proc, n_actuators, device)
        voxel_input = update_input(obs, voxel_masses, sa_matrix, action, robot_shape, voxel_ids, n_proc, n_actuators,
                                   device)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device).repeat(1,n_actuators).reshape((-1, 1))

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return np.mean(eval_episode_rewards)
