import pytorch_lightning as pl
from pl_bolts.models.autoencoders import VAE
from fabric import make_datamodule
from utils import parse_config

if __name__ == '__main__':
    config = parse_config()
    mnist = make_datamodule(config['type'], config['data_path'])
    vae = VAE(input_height=32)
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         weights_save_path=config['weights_save_path'],
                         gpus=config['gpus'],
                         num_processes=config['num_processes']
                         )
    trainer.fit(vae, mnist)
