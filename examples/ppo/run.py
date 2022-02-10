import os, sys
sys.path.insert(1, os.path.join(sys.path[0], 'externals', 'pytorch_a2c_ppo_acktr_gail'))

import numpy as np
import time
from collections import deque
import torch

from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate import evaluate
from ppo.envs import make_vec_envs

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import abc_sr.evogym_utils as evoutils

import evogym.envs

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

def run_ppo(
    structure,
    termination_condition,
    saving_convention,
    override_env_name = None,
    verbose = True):

    assert (structure == None) == (termination_condition == None) and (structure == None) == (saving_convention == None)

    print(f'Starting training on \n{structure}\nat {saving_convention}...\n')

    args = get_args()
    if override_env_name:
        args.env_name = override_env_name

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention != None:
        log_dir = os.path.join(saving_convention[0], log_dir, "robot_" + str(saving_convention[1]))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    robot_structure = (torch.from_numpy(structure[0]).to(device), torch.from_numpy(structure[1]).to(device))

    envs = make_vec_envs(args.env_name, structure, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    robot_shape = robot_structure[0].shape
    n_proc = args.num_processes
    voxel_ids = torch.nonzero((robot_structure[1] == evoutils.VoxelType.H_ACT.value) + (robot_structure[1] == evoutils.VoxelType.V_ACT.value), as_tuple=True)[0]
    mass_matrix = evoutils.mass_pos_matrix(robot_structure[0]).to(device)
    sa_matrix = [evoutils.init_state_action_matrix(robot_structure[0], 3, 1).to(device) for _ in range(n_proc)]


    obs = envs.reset()
    print(f"obs shape{obs.shape}")
    n_actuators = obs.shape[-1]
    obs_mat = [evoutils.mass_obs_matrix(mass_matrix, obs[i], device).unsqueeze(0).to(device) for i in range(n_proc)]

    voxel_input = torch.cat([evoutils.get_voxel_input(robot_shape, obs_i, sa_i, voxel_ids).to(device) for obs_i, sa_i in zip(obs_mat, sa_matrix)])

    rollouts = RolloutStorage(args.num_steps, args.num_processes * n_actuators,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)


    print(n_proc, n_actuators, rollouts.obs[0].shape, voxel_input.shape)
    rollouts.obs[0].copy_(voxel_input)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    rewards_tracker = []
    avg_rewards_tracker = []
    sliding_window_size = 10
    max_determ_avg_reward = float('-inf')

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            sa_pairs = torch.hstack([voxel_input, action]).reshape((n_proc, n_actuators, -1)).to(device)
            actions = action.reshape((n_proc, -1))

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(actions)
            obs_mat = [evoutils.mass_obs_matrix(mass_matrix, obs[i], device).unsqueeze(0).to(device) for i in range(n_proc)]
            sa_matrix = [evoutils.update_state_action_matrix(sa_i, sa, voxel_ids).to(device) for sa, sa_i in zip(sa_pairs, sa_matrix)]
            reward = reward.repeat(1, n_actuators).flatten().to(device)

            voxel_input = torch.cat([evoutils.get_voxel_input(robot_shape, obs_i, sa_i, voxel_ids) for obs_i, sa_i in
                                     zip(obs_mat, sa_matrix)]).to(device)

            # track rewards
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    rewards_tracker.append(info['episode']['r'])
                    if len(rewards_tracker) < 10:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker)))
                    else:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker[-10:])))

            # If done then clean the history of observations.
            masks = torch.FloatTensor( [[0.0] if done_ else [1.0] for done_ in done]).repeat(1,n_actuators).flatten()
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos]).repeat(1,n_actuators).flatten()

            rollouts.insert(voxel_input, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # print status
        if j % args.log_interval == 0 and len(episode_rewards) > 1 and verbose:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

        # evaluate the controller and save it if it does the best so far
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):

            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(args.num_evals, actor_critic, obs_rms, args.env_name, robot_structure, args.seed,
                     args.num_processes, eval_log_dir, device, n_actuators)

            if verbose:
                if saving_convention != None:
                    print(f'Evaluated {saving_convention[1]} using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')
                else:
                    print(f'Evaluated using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward

                temp_path = os.path.join(args.save_dir, args.algo, args.env_name + ".pt")
                if saving_convention != None:
                    temp_path = os.path.join(saving_convention[0], "robot_" + str(saving_convention[1]) + "_controller" + ".pt")

                if verbose:
                    print(f'Saving {temp_path} with avg reward {max_determ_avg_reward}\n')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], temp_path)

        # return upon reaching the termination condition
        if not termination_condition == None:
            if termination_condition(j):
                if verbose:
                    print(f'{saving_convention} has met termination condition ({j})...terminating...\n')
                return max_determ_avg_reward

def init_data(n_actuators, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):

    data = {'obs': torch.zeros(num_steps + 1, num_processes* n_actuators, *obs_shape),
            'recurrent_hidden_states': torch.zeros(num_steps + 1, num_processes* n_actuators, recurrent_hidden_state_size),
            'rewards': torch.zeros(num_steps, num_processes* n_actuators, 1),
            'value_preds': torch.zeros(num_steps + 1, num_processes* n_actuators,  1),
            'returns': torch.zeros(num_steps + 1, num_processes* n_actuators,  1),
            'action_log_probs': torch.zeros(num_steps, num_processes* n_actuators,  1)}
    if action_space.__class__.__name__ == 'Discrete':
        action_shape = 1
    else:
        action_shape = action_space.shape[0]
    data['actions'] = torch.zeros(num_steps, num_processes* n_actuators,  action_shape)
    if action_space.__class__.__name__ == 'Discrete':
        data['actions'] = data['actions'].long()
    data['masks'] = torch.ones(num_steps + 1, num_processes* n_actuators,  1)

    # Masks that indicate whether it's a true terminal state
    # or time limit end state
    data['bad_masks'] = torch.ones(num_steps + 1, num_processes* n_actuators,  1)

    return data

def insert_data(data, step, obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks):
    data['obs'][step + 1].copy_(obs)
    data['recurrent_hidden_states'][step + 1].copy_(recurrent_hidden_states)
    data['actions'][step].copy_(action)
    data['action_log_probs'][step].copy_(action_log_prob)
    data['value_preds'][step].copy_(value)
    data['rewards'][step].copy_(reward)
    data['masks'][step + 1].copy_(masks)
    data['bad_masks'][step + 1].copy_(bad_masks)



#python ppo_main_test.py --env-name "roboticgamedesign-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
#python ppo.py --env-name "roboticgamedesign-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir "logs/"
