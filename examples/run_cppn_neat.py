import random
import sys

import numpy as np

from cppn_neat.run import run_cppn_neat

from abc_sr.evogym_utils import get_number_evaluations, get_robot_shape

if __name__ == '__main__':
    seed = int(sys.argv[2])
    env = sys.argv[4]
    task = sys.argv[4].split("-")[0].lower()
    random.seed(seed)
    np.random.seed(seed)

    best_robot, best_fitness = run_cppn_neat(
        experiment_name="-".join([task, str(seed)]),
        structure_shape=get_robot_shape(env),
        pop_size=50,
        max_evaluations=get_number_evaluations(env),
        train_iters=1000,
        num_cores=25
    )
