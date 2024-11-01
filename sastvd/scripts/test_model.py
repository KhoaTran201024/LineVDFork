import sys
import os
import ray
sys.path.append('')
import pytorch_lightning as pl
from sastvd.linevd import LitGNN
import sastvd.linevd as lvds
from ray import tune
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = {
        "hfeat": tune.choice([512]),
        "embtype": tune.choice(["codebert"]),
        "stmtweight": tune.choice([1]),
        "hdropout": tune.choice([0.3]),
        "gatdropout": tune.choice([0.2]),
        "modeltype": tune.choice(["mlponly"]),
        "gnntype": tune.choice(["gat"]),
        "loss": tune.choice(["ce"]),
        "scea": tune.choice([0.5]),
        "gtype": tune.choice(["pdg+raw"]),
        "batch_size": tune.choice([1024]),
        "multitask": tune.choice(["gat2layer"]),
        "splits": tune.choice(["default"]),
        "lr": tune.choice([1e-3]),#
        "config/gtype":"pdg+raw",
        "config/splits":"default",
        "config/embtype":"codebert"
}
# Load the model from the checkpoint
model = lvds.LitGNN.load_from_checkpoint(
    checkpoint_path="/home/teamq-g2-no2/testauto/LineVDFork/storage/processed/raytune_best_-1/selectedgat2layers50/lightning_logs/version_0/checkpoints/epoch=47-step=753312.ckpt",
    hparams_file="/home/teamq-g2-no2/testauto/LineVDFork/storage/processed/raytune_best_-1/selectedgat2layers50/lightning_logs/version_0/hparams.yaml",
    map_location=None,
)

# Create a trainer instance
trainer = pl.Trainer(
    devices=1, 
    accelerator="cpu"
)
data = lvds.BigVulDatasetLineVDDataModule(
    batch_size=32,
    nsampling_hops=2,
    methodlevel=False,
    sample=-1,
    nsampling=True,
    gtype=config["config/gtype"],
    splits=config["config/splits"],
    feat=config["config/embtype"],
)# Test the model
trainer.test(model, datamodule=data)