import random
import numpy as np
import sys

from ga.run import run_ga

from abc_sr.evogym_utils import get_number_evaluations, get_robot_shape

if __name__ == "__main__":
    seed = int(sys.argv[2])
    env = sys.argv[4]
    task = sys.argv[4].split("-")[0].lower()
    random.seed(seed)
    np.random.seed(seed)

    run_ga(
        pop_size=1,
        structure_shape=get_robot_shape(env),
        experiment_name="-".join([task, str(seed)]),
        max_evaluations=get_number_evaluations(env),
        train_iters=100,
        num_cores=1
    )
