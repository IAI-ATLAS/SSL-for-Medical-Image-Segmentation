import argparse
import logging


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_files',
        '-cfg',
        type=str,
        help='Path to config file(s)'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default="./results",
        help='Path to result_dir'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--raw_data_path',
        type=str,
        help='Path to data needs to be processed'
    )

    args, _ = parser.parse_known_args()
    if 'config_files' not in args:
        raise ValueError('config_files in parameters missing, aborting!')
    logging.info(f"Reading parameter file(s) {args.config_files}")
    if 'result_dir' in args:
        logging.info(f"Result directory: {args.result_dir}")
    if 'ckpt_path' in args:
        logging.info(f"Checkpoint path: {args.ckpt_path}")
    
    return vars(args)