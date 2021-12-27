import pytorch_lightning as pl
from pl_bolts.models.autoencoders import VAE
from pytorch_lightning.callbacks import ModelCheckpoint

from fabric import make_datamodule
from utils import parse_config

if __name__ == '__main__':
    config = parse_config()
    datamodule = make_datamodule(config['type'], config['data_path'], batch_size=512)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config['weights_save_path'],
        filename="exp-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    vae = VAE(input_height=32)
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         callbacks=[checkpoint_callback],
                         gpus=config['gpus'],
                         num_processes=config['num_processes']
                         )
    trainer.fit(vae, datamodule)