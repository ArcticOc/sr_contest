import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch import GradScaler, Tensor, autocast, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange

from .config import args
from .dataset import get_dataset
from .model import ESPCN4x


def train(rank, world_size):
    to_image = transforms.ToPILImage()

    def calc_psnr(image1: Tensor, image2: Tensor):
        image1 = cv2.cvtColor((np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor((np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.PSNR(image1, image2)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        model = ESPCN4x().to(device)
        ddp_model = DDP(model, device_ids=[rank])
        scaler = GradScaler("cuda")
    else:
        device = torch.device("cpu")
        model = ESPCN4x().to(device)
        scaler = GradScaler("cpu")

    writer = SummaryWriter("log")

    train_dataset, validation_dataset = get_dataset()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    validation_data_loader = data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    optimizer = Adam(ddp_model.parameters(), lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 65, 80, 90], gamma=0.7)
    criterion = nn.MSELoss()

    for epoch in trange(args.num_epoch, desc="EPOCH"):
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

                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    output = ddp_model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image, strict=False):
                    train_psnr += calc_psnr(image1, image2)

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
                    with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        output = ddp_model(low_resolution_image)
                        loss = criterion(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image, strict=False):
                        validation_psnr += calc_psnr(image1, image2)
            if rank == 0:
                writer.add_scalar("train/loss", train_loss / len(train_dataset), epoch)
                writer.add_scalar("train/psnr", train_psnr / len(train_dataset), epoch)
                writer.add_scalar("validation/loss", validation_loss / len(validation_dataset), epoch)
                writer.add_scalar("validation/psnr", validation_psnr / len(validation_dataset), epoch)
                writer.add_image("output", output[0], epoch)
        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

    writer.close()

    # モデル生成
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "output/model.pth")

        ddp_model.module.to(torch.device("cpu"))
        dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
        torch.onnx.export(
            model,
            dummy_input,
            "output/model.onnx",
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {2: "height", 3: "width"}},
        )

    dist.destroy_process_group()
