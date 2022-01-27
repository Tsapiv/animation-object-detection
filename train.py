import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from fabric import make_datamodule, make_model
from utils import parse_config

if __name__ == '__main__':
    config = parse_config()
    datamodule = make_datamodule(config['type'], config['data_path'], batch_size=config['batch_size'],
                                 num_workers=config['num_workers'])
    model = make_model(config['type'])
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mapk",
        dirpath=config['weights_save_path'],
        filename="exp-{epoch:02d}-{val_mapk:.2f}",
        save_top_k=3,
        mode="max",
    )
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         callbacks=[checkpoint_callback],
                         gpus=config['gpus']
                         )
    trainer.fit(model, datamodule)
