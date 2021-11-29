"""
@date: 2021/11/22
@description: Conversion training ckpt into inference ckpt
"""
import argparse
import os

import torch

from config.defaults import merge_from_file


def parse_option():
    parser = argparse.ArgumentParser(description='Conversion training ckpt into inference ckpt')
    parser.add_argument('--cfg',
                        type=str,
                        required=True,
                        metavar='FILE',
                        help='path of config file')

    parser.add_argument('--output_path',
                        type=str,
                        help='path of output ckpt')

    args = parser.parse_args()

    print("arguments:")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print("-" * 50)
    return args


def convert_ckpt():
    args = parse_option()
    config = merge_from_file(args.cfg)
    ck_dir = os.path.join("checkpoints", f"{config.MODEL.ARGS[0]['decoder_name']}_{config.MODEL.ARGS[0]['output_name']}_Net",
                          config.TAG)
    print(f"Processing {ck_dir}")
    model_paths = [name for name in os.listdir(ck_dir) if '_best_' in name]
    if len(model_paths) == 0:
        print("Not find best ckpt")
        return
    model_path = os.path.join(ck_dir, model_paths[0])
    print(f"Loading {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))
    net = checkpoint['net']
    output_path = None
    if args.output_path is None:
        output_path = os.path.join(ck_dir, 'best.pkl')
    else:
        output_path = args.output_path
    if output_path is None:
        print("Output path is invalid")
    print(f"Save on: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(net, output_path)


if __name__ == '__main__':
    convert_ckpt()
