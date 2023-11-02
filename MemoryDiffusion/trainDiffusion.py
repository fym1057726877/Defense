import os
import torch
from torch import optim
from models import GaussianDiffusion, UNetModel, ModelMeanType
from respace import SpacedDiffusion, iter_denoise
from data.data_600_classes import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import get_project_path, draw_img_groups


class TrainDiffusion:
    def __init__(
            self,
            unet_pred=ModelMeanType.EPSILON,
            num_ddim_timesteps=100,
            lr=5e-4,
    ):
        super(TrainDiffusion, self).__init__()
        self.device = "cuda"
        self.lr = lr
        self.num_ddim_timesteps = num_ddim_timesteps
        self.unet_pred = ModelMeanType.EPSILON
        self.diffsuion = GaussianDiffusion(mean_type=unet_pred)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(self.device)
        if unet_pred == ModelMeanType.EPSILON:
            self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "diffusion.pth")
        elif unet_pred == ModelMeanType.START_X:
            self.save_path = os.path.join(
                get_project_path(project_name="Defense"), "pretrained", "diffusion_pred_x0.pth")
        else:
            raise NotImplementedError(f"{unet_pred} not implemented")

        self.unet.load_state_dict(torch.load(self.save_path))

    def train(self, epochs):
        start = time()
        optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
        for epoch in range(epochs):
            count = len(trainloader)
            epoch_loss = 0
            self.unet.train()
            for step, (img, _) in tqdm(enumerate(trainloader), desc=f"train step {epoch+1}/{epochs}", total=count):
                optimizer.zero_grad()
                img = img.to(self.device)
                batch_size = img.shape[0]

                t = self.diffsuion.get_rand_t(batch_size, self.device, min_t=20, max_t=100)

                loss = self.diffsuion.training_losses(model=self.unet, x_start=img, t=t)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            epoch_loss /= count
            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

        end = time()
        seconds = int(end - start)
        minutes = seconds // 60
        remain_second = seconds % 60
        print(f"time consumed: {minutes}min{remain_second}s")

    def test_ddim(self):
        assert self.unet_pred == ModelMeanType.EPSILON
        self.unet.eval()
        spacediffusion = SpacedDiffusion(num_ddpm_timesteps=1000, num_ddim_timesteps=self.num_ddim_timesteps)
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(self.device)
        noise = torch.randn_like(imgs)
        final_sample = spacediffusion.ddim_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)
        draw_img_groups([noise, final_sample])

    def test_ddpm(self):
        assert self.unet_pred == ModelMeanType.EPSILON
        self.unet.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(self.device)
        noise = torch.randn_like(imgs)
        final_sample = self.diffsuion.p_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)
        draw_img_groups([noise, final_sample])

    def test_denoise(self, t=30):
        self.unet.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(self.device)
        restore_imgs = self.diffsuion.restore_img(model=self.unet, x_start=imgs, t=t)
        draw_img_groups([imgs, restore_imgs])

    def test_iter_denoise(self, t=50):
        assert self.unet_pred == ModelMeanType.EPSILON and t % 10 == 0
        self.unet.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(self.device)
        final_sample = iter_denoise(self.unet, imgs=imgs, t=t)
        draw_img_groups([imgs, final_sample])


def main():
    # trainer = TrainDiffusion(unet_pred=ModelMeanType.START_X)
    trainer = TrainDiffusion(unet_pred=ModelMeanType.EPSILON)
    # trainer.train(100)
    # trainer.test_ddim()
    # trainer.test_ddpm()
    # trainer.test_denoise(t=40)
    trainer.test_iter_denoise(t=140)


if __name__ == "__main__":
    main()



