import numpy as np
import torch
from ppo import utils
from ppo.envs import make_vec_envs

import abc_sr.evogym_utils as evoutils
from abc_sr.agents import MLPAgent, MLPAgentGlobals

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
    device, voxel_ob_len, action_len, has_globals):

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

    if has_globals:
        global_obs = eval_envs.env_method('global_obs')
        global_obs = torch.from_numpy(np.array(global_obs)).float().to(device)



        globals_len = global_obs.shape[-1]
        mlp_agent = MLPAgentGlobals(robot_structure, n_proc, device, voxel_ob_len, action_len, globals_len)

        voxel_input = mlp_agent.init(obs, global_obs)
    else:
        mlp_agent = MLPAgent(robot_structure, n_proc, device, voxel_ob_len, action_len)
        voxel_input = mlp_agent.init(obs)

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

        if has_globals:
            global_obs = eval_envs.env_method('global_obs')
            global_obs = torch.from_numpy(np.array(global_obs)).float().to(device)

            voxel_input = mlp_agent.step(obs, global_obs, action)
        else:
            voxel_input = mlp_agent.step(obs, action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device).repeat(1,n_actuators).reshape((-1, 1))

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return np.mean(eval_episode_rewards)
