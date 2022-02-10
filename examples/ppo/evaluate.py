import numpy as np
import torch
from ppo import utils
from ppo.envs import make_vec_envs

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
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    robot_shape = robot_structure[0].shape
    voxel_ids = torch.nonzero((robot_structure[1] == evoutils.VoxelType.H_ACT.value) + (robot_structure[1] == evoutils.VoxelType.V_ACT.value), as_tuple=True)[0]
    mass_matrix = evoutils.mass_pos_matrix(robot_structure[0])
    sa_matrix = [evoutils.init_state_action_matrix(robot_structure[0], 3, 1) for _ in range(num_processes)]

    obs_mat = [evoutils.mass_obs_matrix(mass_matrix, obs[i], device).unsqueeze(0) for i in range(num_processes)]

    voxel_input = torch.cat([evoutils.get_voxel_input(robot_shape, obs_i, sa_i, voxel_ids) for obs_i, sa_i in zip(obs_mat, sa_matrix)])

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

        sa_pairs = torch.hstack([voxel_input, action]).reshape((num_processes, n_actuators, -1))
        actions = action.reshape((num_processes, -1))

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        obs_mat = [evoutils.mass_obs_matrix(mass_matrix, obs[i], device).unsqueeze(0) for i in range(num_processes)]
        sa_matrix = [evoutils.update_state_action_matrix(robot_shape, sa_i, sa, voxel_ids) for sa, sa_i in zip(sa_pairs, sa_matrix)]

        voxel_input = torch.cat([evoutils.get_voxel_input(robot_shape, obs_i, sa_i, voxel_ids) for obs_i, sa_i in
                                 zip(obs_mat, sa_matrix)])

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device).repeat(1,n_actuators).flatten()

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return np.mean(eval_episode_rewards)
