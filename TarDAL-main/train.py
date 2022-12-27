import argparse
import logging
from argparse import Namespace

# import wandb # 屏蔽wandb

from pipeline.train import Train
from utils.environment_probe import EnvironmentProbe


def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--id', default='a1', help='train process identifier')
    parser.add_argument('--folder', default='data/train', help='data root path')
    parser.add_argument('--size', default=224, help='resize image to the specified size')
    parser.add_argument('--cache', default='weights', help='weights cache folder')              # 默认路径修改为weights

    # TarDAL opt
    parser.add_argument('--depth', default=3, type=int, help='network dense depth')
    parser.add_argument('--dim', default=32, type=int, help='network features dimension')
    parser.add_argument('--mask', default='m1', help='mark index')
    parser.add_argument('--weight', nargs='+', type=float, default=[1, 20, 0.1], help='loss weight')
    parser.add_argument('--adv_weight', nargs='+', type=float, default=[1, 1], help='discriminator balance')

    # checkpoint opt
    # parser.add_argument('--epochs', type=int, default=200, help='epoch to train')
    parser.add_argument('--epochs', type=int, default=5, help='epoch to train')
    # optimizer opt
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=8, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers number')

    # experimental opt
    parser.add_argument('--debug', action='store_true', help='debug mode (default: off)')

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    # print(config, '超参数信息')
    logging.basicConfig(level='INFO')

    # # wandb settings 已屏蔽
    # wandb.login(key='f751c3b454d75cd63b1350s76a88da7f2fc5102f')  # enter yourself wandb api key
    # runs = wandb.init(
    #     project='tardal',
    #     entity="lovepre",  # enter yourself entity
    #     config=config,
    #     mode='disabled' if config.debug else 'online',
    #     name=config.id,
    # )
    # config = wandb.config

    environment_probe = EnvironmentProbe()  # 显示运行环境
    # print('环境检查完毕----------------------------------------------------------------')
    train_process = Train(environment_probe, config)
    # print('配置导入完毕----------------------------------------------------------------')
    train_process.run()
