import os 
import random
import gc
import argparse
from tqdm import tqdm
from PIL import Image


import matplotlib.pyplot as plt 

from dataclasses import dataclass

from torchvision import transforms
from torchvision.utils import make_grid
from torchmetrics import MeanMetric

import torch
import torch.nn as nn
from torch.cuda import amp

from DM_model import SimpleDiffusion, UNet, TrainingConfig, BaseConfig, forward_diffusion, inverse_transform
from utils import get_dataloader, display, get, setup_log_directory, frames2vid, seed_everything
from image_dataset import load_dataset

@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 256
    BASE_CH_MULT = (1, 2, 4, 4) # 32, 16, 8, 8 
    APPLY_ATTENTION = (False, True, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128


# Algorithm 1: Training

def train_one_epoch(model, sd, loader, optimizer, loss_fn, training_epochs=800,
                   base_config=BaseConfig(), training_config=TrainingConfig(), device='cuda'):
    
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_epochs}")
         
        for x0s, _ in loader:
            tq.update(1)
            
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=device)
            xts, gt_noise = forward_diffusion(sd, x0s.to(device), ts)

            with torch.amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()
    
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
    
    return mean_loss 


# Algorithm 2: Sampling
    
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=8, device="cpu", **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        frames2vid(outs, kwargs['save_path'])
        display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.
        return None

    else: # Display and save the image at the final timestep of the reverse process. 
        x = inverse_transform(x).type(torch.uint8)
        grid = make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = transforms.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())
        display(pil_image)
        return None
    



if __name__ == '__main__':

    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser('encoder decoder examiner')
    parser.add_argument('--TIMESTEPS', type=int, default=1000, help='')        
    parser.add_argument('--DATASET', type=str, default="CIFAR10" , help='CIFAR10, MNIST, Cifar-100, Flowers')
    parser.add_argument('--NUM_EPOCHS', type=int, default=30, help='number of training epochs')
    parser.add_argument('--BATCH_SIZE', type=int, default=128, help='batch size per GPU')
    parser.add_argument('--LR', type=float, default=2e-4, help='')
    parser.add_argument('--NUM_WORKERS', type=int, default=2, help='')
    #parser.add_argument('--', type=, default=, help='')
    parser.add_argument('--BASE_CH',type=int, default=64, help='# 64, 128, 256, 256')
    parser.add_argument('--BASE_CH_MULT',type=tuple, default=(1, 2, 4, 4) , help='# 32, 16, 8, 8 ')
    parser.add_argument('--APPLY_ATTENTION',type=tuple, default=(False, True, True, False), help='')
    parser.add_argument('--DROPOUT_RATE',type=float, default=0.1, help='')
    parser.add_argument('--TIME_EMB_MULT',type=int, default=4, help='# 128')
    ##parser.add_argument('--',type=, default=, help='')
    #parser.add_argument('--',type=, default=, help='')



    args = parser.parse_args()


    sd = SimpleDiffusion(num_diffusion_timesteps=args.TIMESTEPS, device=device)
    
    """
    loader = iter(  # converting dataloader into an iterator for now.
        get_dataloader(
            dataset_name=BaseConfig.DATASET,
            batch_size=6,
            device="cpu",
        )
    )


    x0s, _ = next(loader)

    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = forward_diffusion(sd, x0s, timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)
    
    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()
   
    

    x0s, _ = next(loader)

    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long)

        xts, _ = forward_diffusion(sd, x0s, timestep)
        xts = inverse_transform(xts) / 255.0
        xts = make_grid(xts, nrow=1, padding=1)

        noisy_images.append(xts)

    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")
    
    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    plt.show()
    """

    model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = args.BASE_CH,
        base_channels_multiples = args.BASE_CH_MULT,
        apply_attention         = args.APPLY_ATTENTION,
        dropout_rate            = args.DROPOUT_RATE,
        time_multiple           = args.TIME_EMB_MULT,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

    """
    dataloader = get_dataloader(
        dataset_name  = BaseConfig.DATASET,
        batch_size    = TrainingConfig.BATCH_SIZE,
        device        = BaseConfig.DEVICE,
        pin_memory    = True,
        num_workers   = TrainingConfig.NUM_WORKERS,
    )
    """

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps = args.TIMESTEPS,
        img_shape               = TrainingConfig.IMG_SHAPE,
        device                  = device,
    )

    scaler = torch.amp.GradScaler()

    total_epochs = args.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    generate_video = False
    ext = ".mp4" if generate_video else ".png"

    train_dataloader, test_dataloader = load_dataset(args.DATASET)

    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Algorithm 1: Training
        train_one_epoch(model, sd, train_dataloader, optimizer, scaler, loss_fn, training_epochs=args.NUM_EPOCHS)

        if epoch % 20 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")
            
            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32, generate_video=generate_video,
                save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
            )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict
    
    reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32, generate_video=generate_video,
                save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
            )
        