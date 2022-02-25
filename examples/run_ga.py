import random
import numpy as np
import sys

from ga.run import run_ga

if __name__ == "__main__":
    seed = int(sys.argv[1])
    task = sys.argv[3].split("-")[0].lower()
    random.seed(seed)
    np.random.seed(seed)

    run_ga(
        pop_size=25,
        structure_shape=(5, 5),
        experiment_name="-".join([task, str(seed)]),
        max_evaluations=250 if "soft" in task else 500,
        train_iters=1000,
        num_cores=1
    )
