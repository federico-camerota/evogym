import random
import numpy as np

from ga.run import run_ga

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    run_ga(
        pop_size = 3,
        structure_shape = (3,3),
        experiment_name = "test_ga",
        max_evaluations = 5,
        train_iters = 50,
        num_cores = 4,
    )
