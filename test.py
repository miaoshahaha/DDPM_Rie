import gc
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


from utils import setup_log_directory, get_dataloader
from DM_model import UNet, forward_diffusion
from train import BaseConfig, TrainingConfig, ModelConfig, reverse_diffusion, SimpleDiffusion

def train_one_epoch(model, sd, loader, optimizer, loss_fn, epoch=800, 
                   base_config=BaseConfig(), training_config=TrainingConfig()):
    
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
         
        for x0s, _ in loader:
            #tq.update(1)
            
            x0s = x0s.to(base_config.DEVICE)
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = forward_diffusion(sd, x0s, ts)

            pred_noise = model(xts, ts)
            loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #scaler.step(optimizer)
            optimizer.step()
            #scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

        # tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")
            print(loss_value)

        mean_loss = loss_record.compute().item()

        #tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss 


model = UNet(
        input_channels          = TrainingConfig.IMG_SHAPE[0],
        output_channels         = TrainingConfig.IMG_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
model.to(BaseConfig.DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

dataloader = get_dataloader(
    dataset_name  = BaseConfig.DATASET,
    batch_size    = TrainingConfig.BATCH_SIZE,
    device        = BaseConfig.DEVICE,
    pin_memory    = True,
    num_workers   = TrainingConfig.NUM_WORKERS,
)

loss_fn = nn.MSELoss()

sd = SimpleDiffusion(
    num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
    img_shape               = TrainingConfig.IMG_SHAPE,
    device                  = BaseConfig.DEVICE,
)



total_epochs = TrainingConfig.NUM_EPOCHS + 1
log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

generate_video = False
ext = ".mp4" if generate_video else ".png"

train_dataset = CIFAR10(root='cifar10_ds', download=True, train=True,  transform=transforms.ToTensor())
test_dataset = CIFAR10(root='cifar10_ds', download=True, train=False,  transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)


for epoch in range(1, total_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    


    # Algorithm 1: Training
    train_one_epoch(model, sd, train_dataloader, optimizer, loss_fn, epoch=epoch) #scaler, loss_fn, epoch=epoch)

    if epoch % 20 == 0:
        save_path = os.path.join(log_dir, f"{epoch}{ext}")
        
        
        # Algorithm 2: Sampling
        reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32, generate_video=generate_video,
            save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
        )
        
        # clear_output()
        checkpoint_dict = {
            "opt": optimizer.state_dict(),
            "model": model.state_dict()
        }
        torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
        del checkpoint_dict

reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=32, generate_video=generate_video,
            save_path=save_path, img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE,
        )