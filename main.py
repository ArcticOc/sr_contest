import torch
import torch.distributed as dist

from train.compare import calc_and_print_PSNR
from train.config import args
from train.inference import Qinference
from train.train import train

if __name__ == "__main__":
    # DDP Initialization
    if torch.cuda.is_available():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if not args.only_eval:
        train(rank, world_size)

    if rank == 0:
        Qinference().inference_onnxruntime()
        calc_and_print_PSNR()

    if args.only_eval:
        dist.destroy_process_group()
