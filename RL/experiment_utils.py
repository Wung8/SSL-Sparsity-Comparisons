import csv
import json
import os
import random
import time

import numpy as np
import torch


def set_global_seed(seed, deterministic_torch=False):
    '''
    Seeds python / numpy / torch in the calling process. Environment worker processes
    are seeded separately -- see ParallelEnvManager(seed=...) -- because they are spawned
    and do not inherit this process's RNG state on Windows.

    deterministic_torch=True makes cuDNN deterministic at a noticeable speed cost.
    Leave it off for sweeps; turn it on only when chasing a reproducibility bug.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


class RunLogger:
    '''
    Append-only CSV logger, one row per training iteration, plus a JSON sidecar holding
    the run config. Flushes every row so an interrupted run keeps its data.

    Columns are inferred from the first log() call, so any metric added later (effective
    rank, dormant ratio, measured sparsity, SSL loss) needs no changes here -- but it
    must be present in the FIRST call or it will be dropped. Pass the full metric set
    from iteration one, using None for values not yet available.
    '''

    def __init__(self, log_dir, run_name, config=None):
        self.dir = os.path.join(log_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, "progress.csv")
        self.writer = None
        self.file = None
        self.fields = None
        self.start_time = time.time()

        if config is not None:
            with open(os.path.join(self.dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)

    def log(self, **metrics):
        metrics.setdefault("wall_time", round(time.time() - self.start_time, 3))

        if self.writer is None:
            self.fields = list(metrics.keys())
            self.file = open(self.csv_path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()

        row = {k: metrics.get(k) for k in self.fields}
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None
