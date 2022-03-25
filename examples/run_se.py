import random
import sys

import numpy as np
from abc_sr.evogym_utils import get_robot_shape, get_number_evaluations

from se.run import run_se

if __name__ == "__main__":
    seed = int(sys.argv[2])
    env = sys.argv[4]
    task = sys.argv[4].split("-")[0].lower()
    random.seed(seed)
    np.random.seed(seed)

    run_se(
        pop_size=25,
        structure_shape=get_robot_shape(env),
        experiment_name="-".join([task, str(seed)]),
        max_evaluations=get_number_evaluations(env),
        train_iters=1000,
        num_cores=25
    )
