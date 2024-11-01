import pytorch_lightning as pl
import sastvd.linevd.__init__ as lvd
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from pytorch_lightning.callbacks import EarlyStopping

def train_linevd(                                   #fix some bugs here
    config, savepath, samplesz=-1, max_epochs=1, num_gpus=1, checkpoint_dir=None
):
    print("ENTER TRAIN_LINEVD FUNCTION")
    """Wrap Pytorch Lightning to pass to RayTune."""
    model = lvd.LitGNN(
        hfeat=config["hfeat"],
        embtype=config["embtype"],
        methodlevel=False,
        nsampling=True,
        model=config["modeltype"],
        loss=config["loss"],
        hdropout=config["hdropout"],
        gatdropout=config["gatdropout"],
        num_heads=4,
        multitask=config["multitask"],
        stmtweight=config["stmtweight"],
        gnntype=config["gnntype"],
        scea=config["scea"],
        lr=config["lr"],
        lstm_layers = 2,  # Number of LSTM layers
        lstm_dropout= 0.2,  # Dropout rate for LSTM
    )
    # Load data
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )
    print("LINE 41 TRAINER")
    # # Train model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss",save_top_k = -1)
    metrics = ["train_loss", "val_loss", "val_auroc"]
    #raytune_callback  = TuneReportCheckpointCallback(metrics, on="validation_end")
    #raytune_callback = TuneReportCallback(metrics, on="validation_end")
    rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")
    # trainer = pl.Trainer(
    #     gpus=0,#fix somebug here
    #     auto_lr_find=False,
    #     default_root_dir=savepath,
    #     num_sanity_val_steps=0,
    #     callbacks=[checkpoint_callback, raytune_callback, rtckpt_callback],
    #     max_epochs=max_epochs,
    # )
    # Initialize the PyTorch Lightning Trainer
    #lastest version update

    trainer = pl.Trainer(
        #devices=0,  # Use 'devices' in newer versions, set to '0' for CPU
	#gpus=1,
        devices=1, 
        accelerator="gpu",  # auto choose cpu or gpu
        #auto_lr_find=False,
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, rtckpt_callback],
        max_epochs=max_epochs,
    )

    print("LINE 57 run.py")
    trainer.fit(model,data,ckpt_path="/home/teamq-g2-no2/testauto/LineVDFork/storage/processed/raytune_best_-1/202410292030_e715a0c_happy_happy/lightning_logs/version_0/checkpoints/epoch=1-step=31388.ckpt")
    trainer.test(ckpt_path="best",datamodule=data)
