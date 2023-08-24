import os
import subprocess
import random
import datetime
import shutil
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from config import config_parser
from tensorboardX import SummaryWriter
from loaders.create_training_dataset import get_training_dataset
from trainer import BaseTrainer
torch.manual_seed(1234)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(args):
    seq_name = os.path.basename(args.data_dir.rstrip('/'))
    out_dir = os.path.join(args.save_dir, '{}_{}'.format(args.expname, seq_name))
    os.makedirs(out_dir, exist_ok=True)
    print('optimizing for {}...\n output is saved in {}'.format(seq_name, out_dir))

    args.out_dir = out_dir

    # save the args and config files
    f = os.path.join(out_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            if not arg.startswith('_'):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

    if args.config:
        f = os.path.join(out_dir, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    log_dir = 'logs/{}_{}'.format(args.expname, seq_name)
    writer = SummaryWriter(log_dir)

    g = torch.Generator()
    g.manual_seed(args.loader_seed)
    dataset, data_sampler = get_training_dataset(args, max_interval=args.start_interval)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.num_pairs,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              num_workers=args.num_workers,
                                              sampler=data_sampler,
                                              shuffle=True if data_sampler is None else False,
                                              pin_memory=True)

    # get trainer
    trainer = BaseTrainer(args)

    start_step = trainer.step + 1
    step = start_step
    epoch = 0
    while step < args.num_iters + start_step + 1:
        for batch in data_loader:
            trainer.train_one_step(step, batch)
            trainer.log(writer, step)

            step += 1

            dataset.set_max_interval(args.start_interval + step // 2000)

            if step >= args.num_iters + start_step + 1:
                break

        epoch += 1
        if args.distributed:
            data_sampler.set_epoch(epoch)


if __name__ == '__main__':
    args = config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)

