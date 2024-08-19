import importlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange

from .config import args
from .dataset import get_dataset

warmup_epochs = args.num_epoch * args.warmup_factor


def load_model(module_name, class_name='ESPCN4x', relative_to=None):
    if relative_to:
        module_name = f'{relative_to}.{module_name}'
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not load {class_name} from {module_name}: {e}") from None


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return max(
            0.5 * (1 + math.cos((epoch - warmup_epochs) / (args.num_epoch - warmup_epochs) * math.pi)),
            args.min_lr / args.learning_rate,
        )


def train(rank, world_size):
    torch.backends.cudnn.benchmark = True
    to_image = transforms.ToPILImage()
    ESPCN4x = load_model(f'{args.model_type}', 'ESPCN4x', 'train.model')

    def calc_psnr(image1: Tensor, image2: Tensor):
        image1 = cv2.cvtColor((np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor((np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.PSNR(image1, image2)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        model = ESPCN4x().to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_model = DDP(model, device_ids=[rank])
    else:
        device = torch.device("cpu")
        model = ESPCN4x().to(device)

    writer = SummaryWriter(f"log/{args.writer_name}")

    train_data_loader, validation_data_loader, train_sampler = get_dataset(world_size, rank)

    optimizer = AdamW(ddp_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01 * args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in trange(args.num_epoch, desc="EPOCH"):
        if epoch < warmup_epochs:
            lr_scale = min(1.0, float(epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * args.learning_rate
        train_sampler.set_epoch(epoch)
        try:
            # 学習
            ddp_model.train()
            train_loss = 0.0
            train_psnr = 0.0

            if rank == 0:
                data_loader = tqdm(train_data_loader, desc=f"EPOCH[{epoch}] TRAIN", total=len(train_data_loader))
            else:
                data_loader = train_data_loader
            for idx, (low_resolution_image, high_resolution_image) in enumerate(data_loader):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                optimizer.zero_grad()

                output = ddp_model(low_resolution_image)
                loss = criterion(output, high_resolution_image)

                loss.backward()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image, strict=False):
                    train_psnr += calc_psnr(image1, image2)
                optimizer.step()

            scheduler.step()

            # 検証
            ddp_model.eval()
            validation_loss = 0.0
            validation_psnr = 0.0
            with torch.no_grad():
                if rank == 0:
                    data_loader = tqdm(
                        validation_data_loader, desc=f"EPOCH[{epoch}] VAL", total=len(validation_data_loader)
                    )
                else:
                    data_loader = validation_data_loader

                for idx, (low_resolution_image, high_resolution_image) in enumerate(data_loader):
                    low_resolution_image = low_resolution_image.to(device)
                    high_resolution_image = high_resolution_image.to(device)

                    output = ddp_model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image, strict=False):
                        validation_psnr += calc_psnr(image1, image2)
            if rank == 0:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/psnr", train_psnr, epoch)
                writer.add_scalar("validation/loss", validation_loss, epoch)
                writer.add_scalar("validation/psnr", validation_psnr, epoch)
                # writer.add_image("output", output[0], epoch)
        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

    writer.close()

    # モデル生成
    if rank == 0:
        output_model_dir = Path(f"output/{args.model_type}/{args.model_type}")
        output_model_dir.mkdir(parents=True, exist_ok=True)
        # torch.save(ddp_model.module.state_dict(), f"output/{args.model_type}.pth")

        ddp_model.module.to(torch.device("cpu"))
        dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
        torch.onnx.export(
            model,
            dummy_input,
            f"{output_model_dir}.onnx",
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {2: "height", 3: "width"}},
        )

    dist.destroy_process_group()
