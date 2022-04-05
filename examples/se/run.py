import os
import numpy as np
import shutil
import random
import math

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure

from sklearn.cluster import KMeans

def compute_ranks(structures, shape_features, cluster_idxs, cluster_centers):
    dists = [np.linalg.norm(shape_features[cluster_idxs[c]] - cluster_centers[c]) for c in range(len(cluster_centers))]
    reprs_idxs = [np.argmin(d) for d in dists]
    reprs = [c_idxs[r] for c_idxs, r in zip(cluster_idxs, reprs_idxs)]
    fitness = np.array([structures[r].fitness for r in reprs])
    fit_ranks = np.argsort(fitness)
    idxs = np.arange(len(fit_ranks))
    return idxs[fit_ranks]

def run_se(experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores):
    print()

    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###
    tc = TerminationCondition(train_iters)

    is_continuing = False
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")

        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' +
            f'max evals: {max_evaluations}, train iters {train_iters}.')

        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0

    #generate a population
    if not is_continuing:
        for i in range (pop_size):

            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(Structure(*temp_structure, i))
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1

    #read status from file
    else:
        for g in range(start_gen+1):
            for i in range(pop_size):
                save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(g), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                # only a current structure if last gen
                if g == start_gen:
                    structures.append(Structure(*structure_data, i))
        num_evaluations = len(list(population_structure_hashes.keys()))
        generation = start_gen


    while True:

        ### UPDATE NUM SURVIORS ###
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")

        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")

                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error coppying controller for {save_path_controller_part}.\n')
            else:
                ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label))
                group.add_job(run_ppo, ppo_args, callback=structure.set_reward)

        group.run_jobs(num_cores)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()

         ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])

        ### CLUSTERING ###
        n_clusters = 10
        n_elite = 5

        shape_features = np.array([s.body.flatten() for s in structures]) / 4
        kmeans = KMeans(n_clusters=n_clusters)

        cluster_labels = kmeans.fit_predict(shape_features)
        cluster_centers = kmeans.cluster_centers_
        cluster_idxs = [np.where(cluster_labels == c)[0].astype(int) for c in range(n_clusters)]

        ### CROSSOVER AND MUTATION ###
        # Save best structure and best for every cluster
        survivors = structures[:1]
        survivors += [structures[ids[0]] for ids in cluster_idxs if len(ids) > n_elite]
        for i, s in enumerate(survivors):
            s.is_survivor = True
            s.prev_gen_label = s.label
            s.label = i


        n_children = pop_size - len(survivors)

        alpha = 0.75
        ranks = compute_ranks(structures, shape_features, cluster_idxs, cluster_centers)
        n_k = sum([alpha ** r for r in ranks])
        n_k = [n_children * (alpha ** r) / n_k for r in ranks]

        # Produce children with mutations
        for i in range(n_clusters):
            c = 0
            while c < n_k[i]:
                idx = c % len(cluster_idxs[i])
                idx = cluster_idxs[i][idx]
                child = mutate(structures[idx].body.copy(), mutation_rate = 0.1, num_attempts=50)

                if child != None and hashable(child[0]) not in population_structure_hashes:
                    population_structure_hashes[hashable(child[0])] = True
                    num_evaluations += 1
                    survivors.append(Structure(*child, len(survivors)))
                    c += 1

        structures = survivors

        generation += 1
