# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import numpy as np
from tqdm import tqdm
import logging

import torch
# see: https://github.com/pytorch/pytorch/issues/973
#  needed when I was using DataParallel, but I don't think its necessary
#  for DistributedDataParallel
#torch.multiprocessing.set_sharing_strategy('file_system')
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from eend.pytorch_backend.models import TransformerModel, NoamScheduler, TransformerEDAModel
from eend.pytorch_backend.diarization_dataset import KaldiDiarizationDataset, my_collate
from eend.pytorch_backend.loss import batch_pit_loss, eda_batch_pit_loss, report_diarization_error


def init_process(rank, size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend, rank=rank, world_size=size)


def train_runner(args, ddp=True):
    # will spawn the mutliple processes for DDP
    if ddp:
        if torch.cuda.is_available():
            if args.gpu == 'auto-detect':
                world_size = torch.cuda.device_count()
            else:
                world_size = int(args.gpu)
        else:
            raise Exception('Need CUDA to use DDP')
    else:
        # TODO: add support?
        raise Exception("Unsupported")

    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

def train(rank, world_size, args):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """
    #torch.set_num_threads(16)  

    # Logger settings====================================================
    formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(args.model_save_dir + "/train_" + str(rank) + ".log", mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # ===================================================================
    logger.info(str(args))

    init_process(rank, world_size)
    logger.info(
        f"Rank {rank + 1}/{world_size} process initialized.\n"
    )

    #os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    """
    if torch.cuda.is_available():
        if args.gpu == 'auto-detect':
            num_gpu_arg = torch.cuda.device_count()
        else:
            num_gpu_arg = int(args.gpu)
        device = torch.device('cuda')
    else:
        num_gpu_arg = 1
        device = torch.device('cpu')
    """

    train_set = KaldiDiarizationDataset(
        data_dir=args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        #n_gpu=num_gpu_arg,
        logger=logger,
        )
    dev_set = KaldiDiarizationDataset(
        data_dir=args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        #n_gpu=num_gpu_arg,
        logger=logger
        )

    # Prepare model
    Y, T = next(iter(train_set))
    
    if args.model_type.lower() == 'transformer':
        model = TransformerModel(
            n_speakers=args.num_speakers,
            in_size=Y.shape[1],
            n_units=args.hidden_size,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout=args.transformer_encoder_dropout,
            has_pos=False
        )
    elif args.model_type.lower() == 'transformer_eda':
        model = TransformerEDAModel(
            n_speakers=args.num_speakers,
            in_size=Y.shape[1],
            n_units=args.hidden_size,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            xformer_dropout=args.transformer_dropout,
            attractor_encoder_dropout=args.attractor_encoder_dropout,
            attractor_decoder_dropout=args.attractor_decoder_dropout,
            attractor_loss_ratio=args.attractor_loss_ratio,
            has_pos=False
        )
    else:
        raise ValueError('Possible model_type are: ["Transformer". "Transformer_EDA"]')
  
    """ 
    if device.type == "cuda":
        # TODO: convert to DistributedDataParallel
        logger.info('Using %d GPUs' % (num_gpu_arg, ))
        model = nn.DataParallel(model, list(range(num_gpu_arg)))
    """
    model.cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    #model = model.to(device)
    logger.info('Prepared model')
    logger.info(model)

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr * world_size)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr * world_size)
    elif args.optimizer == 'noam':
        # for noam, lr refers to base_lr (i.e. scale), suggest lr=1.0
        optimizer = optim.Adam(model.parameters(), lr=args.lr * world_size, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(args.optimizer)

    # For noam, we use noam scheduler
    if args.optimizer == 'noam':
        scheduler = NoamScheduler(optimizer,
                                  args.hidden_size,
                                  warmup_steps=args.noam_warmup_steps)

    # Init/Resume
    if args.initmodel:
        logger.info(f"Load model from {args.initmodel}")
        model.load_state_dict(torch.load(args.initmodel))

    train_sampler = DistributedSampler(train_set, rank=rank, num_replicas=world_size)
    dev_sampler = DistributedSampler(dev_set, rank=rank, num_replicas=world_size)
    train_iter = DataLoader(
            train_set,
            batch_size=int(args.batchsize/world_size),
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=my_collate,
            sampler=train_sampler
    )

    dev_iter = DataLoader(
            dev_set,
            batch_size=int(args.batchsize/world_size),
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=my_collate,
            sampler=dev_sampler
    )

    # Training
    # y: feats, t: label
    # grad accumulation is according to: https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        # zero grad here to accumualte gradient
        optimizer.zero_grad()
        loss_epoch = 0
        num_total = 0
        for step, (y, t) in tqdm(enumerate(train_iter), ncols=100, total=len(train_iter)):
            y = [yi.cuda(rank) for yi in y]
            t = [ti.cuda(rank) for ti in t]

            output = model(y)

            """
            ####  DEBUGGING  ####
            for ii in range(len(output)):
                if output[ii].size()[0] != y[ii].size()[0]:
                    import pdb; pdb.set_trace()
                    output = model(y)
            #####################
            """
            if args.model_type.lower() == 'transformer' or isinstance(model, TransformerModel):
                loss, label = batch_pit_loss(output, t)
            elif args.model_type.lower() == 'transformer_eda' or isinstance(model, TransformerEDAModel):
                y, attractor_logits = output
                loss, attractor_loss, pit_loss = eda_batch_pit_loss(y, t, attractor_logits,
                                                                    attractor_loss_ratio=model.attractor_loss_ratio)
            else:
                raise Exception("Unknown model type!")
            # clear graph here
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # noam should be updated on step-level
                if args.optimizer == 'noam':
                    scheduler.step()
                if args.gradclip > 0:
                    nn.utils.clip_grad_value_(model.parameters(), args.gradclip)
            loss_epoch += loss.item()
            num_total += 1
        
        loss_epoch /= num_total
        model.eval()
        with torch.no_grad():
            stats_avg = {}
            cnt = 0
            for y, t in dev_iter:
                y = [yi.cuda(rank) for yi in y]
                t = [ti.cuda(rank) for ti in t]

                output = model(y)
                _, label = batch_pit_loss(output, t)
                stats = report_diarization_error(output, label)
                for k, v in stats.items():
                    stats_avg[k] = stats_avg.get(k, 0) + v
                cnt += 1
            stats_avg = {k:v/cnt for k,v in stats_avg.items()}
            stats_avg['DER'] = stats_avg['diarization_error'] / stats_avg['speaker_scored'] * 100
            for k in stats_avg.keys():
                stats_avg[k] = round(stats_avg[k], 2)

        if rank == 0 and epoch % 5 == 0:
            model_filename = os.path.join(args.model_save_dir, f"transformer{epoch}.th")
            torch.save(model.state_dict(), model_filename)

        logger.info(f"Epoch: {epoch:3d}, LR: {optimizer.param_groups[0]['lr']:.7f},\
            Training Loss: {loss_epoch:.5f}, Dev Stats: {stats_avg}")

    logger.info('Finished!')
