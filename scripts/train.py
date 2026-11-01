#!/usr/bin/env python3
import argparse
from polastflow.engine.trainer import train


def parser():
    p = argparse.ArgumentParser(description='Train the full PolaSTFlow model')
    p.add_argument('--model-config', default='configs/model.yaml')
    p.add_argument('--train-data', required=True)
    p.add_argument('--val-data', required=True)
    p.add_argument('--output-dir', default='runs/default')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8, help='Global batch size for DDP')
    p.add_argument('--val-batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=0.3)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--amp', action='store_true', help='Original default: off')
    p.add_argument('--resume', default='', help='Path to last.pth')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
