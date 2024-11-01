import os
import ray
import sys
sys.path.append('')
import sastvd
import sastvd.linevd.run as lvdrun
from ray import tune

#------------------------------comment the below line----------------------------------------------
#os.environ["SLURM_JOB_NAME"] = "bash"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"]="" #add to suppress the warning

# Adjust the memory usage threshold (e.g., set to 90%)
os.environ['RAY_memory_usage_threshold'] = '0.9'

# Disable worker killing by setting the refresh interval to zero
os.environ['RAY_memory_monitor_refresh_ms'] = '0'

ray.init(num_cpus=8, num_gpus=0, runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}})
config = {
    "hfeat": tune.choice([512]),
    "embtype": tune.choice(["codebert"]),
    "stmtweight": tune.choice([1]),
    "hdropout": tune.choice([0.3]),
    "gatdropout": tune.choice([0.2]),
    "modeltype": tune.choice(["mlponly"]),
    "gnntype": tune.choice(["gat2layer"]),
    "loss": tune.choice(["ce"]),
    "scea": tune.choice([0.5]),
    "gtype": tune.choice(["pdg+raw"]),
    "batch_size": tune.choice([32]),
    "multitask": tune.choice(["linemethod"]),
    "splits": tune.choice(["default"]),
    "lr": tune.choice([1e-4]),#
}

samplesz = -1
run_id = sastvd.get_run_id()
sp = sastvd.get_dir(sastvd.processed_dir() / f"raytune_best_{samplesz}" / run_id)
trainable = tune.with_parameters(
    lvdrun.train_linevd, max_epochs=50, samplesz=samplesz, savepath=sp
)

analysis = tune.run(
    trainable,
    resources_per_trial={"cpu": 6, "gpu": 0},
    metric="val_loss",
    mode="min",
    config=config,
    num_samples=1,
    name="tune_linevd",
    #local_dir=sp,
    keep_checkpoints_num=1,
    checkpoint_score_attr="min-val_loss",
    storage_path=sp,#fix bug here
    max_concurrent_trials=1#add this to prevent something
)

