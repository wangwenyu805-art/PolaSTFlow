#!/usr/bin/env python3
import argparse
from polastflow.engine.predictor import predict


def parser():
    p = argparse.ArgumentParser(description='Generate predictions without ground-truth targets')
    p.add_argument('--model-config', default='configs/model.yaml')
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', default='predictions')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--amp', action='store_true')
    return p


if __name__ == '__main__':
    predict(parser().parse_args())
