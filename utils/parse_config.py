import argparse
import json


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, required=False)
    parser.add_argument('--weights_save_path', type=str, required=False)
    parser.add_argument('--gpus', type=int, required=False)
    parser.add_argument('--num_processes', type=int, required=False)
    parser.add_argument('--data_path', type=str, required=False)
    parser.add_argument('--type', type=str, required=False)
    parser.add_argument('--config', type=str, required=True)

    args = vars(parser.parse_args())
    with open(args['config'], "r") as f:
        config = json.load(f)
    for key in args:
        config[key] = config.get(key, None) if args[key] is None else args[key]

    return config
