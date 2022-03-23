import random
import numpy as np

from se.run import run_se

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    run_se(
        pop_size=4,
        structure_shape=(3, 3),
        experiment_name="se_test",
        max_evaluations=4,
        train_iters=20,
        num_cores=2
    )
