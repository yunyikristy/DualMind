import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import utils
from arguments import parser
from datamodules.multi_domains_datamodules import MultiDomainDataModule
from models.multidomain_dt_module import MultiTaskDTLitModule


def main(args):
    # set seed for reproducibility, although the trainer does not allow deterministic for this implementation
    pl.seed_everything(args.seed, workers=True)

    # init data module
    dataset = MultiDomainDataModule.from_argparse_args(args)

    # init training module
    dict_args = vars(args)
    model = MultiTaskDTLitModule(**dict_args)

    output_dir = args.output_dir
    # init root dir
    os.makedirs(output_dir, exist_ok=True)
    print("output dir", output_dir)

    # checkpoint saving metrics

    logger = TensorBoardLogger(os.path.join(args.output_dir, "tb_logs"), name="train")

    # init trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="checkpoint_{epoch:02d}-{global_step}",
        mode="max",
        save_top_k=-1,
        monitor="epoch",
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        callbacks=[checkpoint_callback],
        default_root_dir=args.output_dir,
        min_epochs=1,
        max_epochs=args.epochs,
        strategy="ddp",
        fast_dev_run=False,
        logger=[logger],
        precision=16,
    )

    # start training
    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":

    parser = MultiDomainDataModule.add_argparse_args(parser)
    parser = MultiTaskDTLitModule.add_model_specific_args(parser)

    args = parser.parse_args()

    args.timesteps = 1000
    if args.training_phase == 1:
        args.model_type = 'gpt'
    elif args.training_phase == 2:
        args.model_type = 'multitask'
    utils.init_distributed_mode(args)
    main(args)
