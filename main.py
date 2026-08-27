import os
import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--config_path", default="./configs/semantic_kitti.py")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--seed", type=int, default=7240, help="random seed point")
    parser.add_argument("--log_folder", default="semantic_kitti")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--test_mapping", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=1000)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg


if __name__ == "__main__":
    args, config = parse_config()
    log_folder = config["log_folder"]
    misc.check_path(log_folder)

    misc.check_path(os.path.join(log_folder, "tensorboard"))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder, name="tensorboard", version=0
    )

    config.dump(os.path.join(log_folder, "config.py"))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    model = pl_model(config)

    data_dm = DataModule(config)

    checkpoint_callback = ModelCheckpoint(  #* 保存验证集 mIoU 最好的模型以及最近一次训练状态
        monitor="val/mIoU",  # 监控验证阶段记录的平均 IoU
        mode="max",  # mIoU 越大越好，数值变大时更新最佳模型
        save_last=True,  # 额外保存最近一次训练状态为 last.ckpt，便于断点续训
        filename="best",  # 最佳模型保存为 best.ckpt；save_top_k 默认为 1
    )

    checkpoint_callback_mIoU = ModelCheckpoint(  #* 每次验证后都保存一份历史模型
        monitor="val/mIoU",  # 同样监控验证集平均 IoU
        save_last=False,  # 不重复保存 last.ckpt，由上面的回调负责
        save_top_k=-1,  # -1 表示保留每一次验证产生的 checkpoint
        auto_insert_metric_name=False,  # 禁止 Lightning 自动给文件名字段添加指标名称
        filename="epoch={epoch:03d}-mIoU={val/mIoU:.5f}-IoU={val/IoU:.5f}",  # 文件名记录轮次和指标
    )

    if not config.eval:
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(accelerator="gpu", find_unused_parameters=True),
            max_steps=config.training_steps,
            callbacks=[
                checkpoint_callback,
                checkpoint_callback_mIoU,
                LearningRateMonitor(logging_interval="step"),
            ],
            logger=tb_logger,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config["log_every_n_steps"],
            check_val_every_n_epoch=config["check_val_every_n_epoch"],
        )
        trainer.fit(model=model, datamodule=data_dm)
    else:
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(accelerator="gpu", find_unused_parameters=True),
            logger=tb_logger,
            profiler=profiler,
        )
        trainer.test(model=model, datamodule=data_dm, ckpt_path=config["ckpt_path"])
