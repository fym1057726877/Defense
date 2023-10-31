import os
import torch
from torch import optim
from gaussiandiffusion import GaussianDiffusion, UNetModel, ModelMeanType
from respace import SpacedDiffusion
from data.mydataset import trainloader, testloader
from tqdm import tqdm
from time import time
from utils import get_project_path, draw_img_groups


device = "cuda"


class TrainDiffusion:
    def __init__(
            self,
            unet_pred=ModelMeanType.EPSILON,
            num_ddim_timesteps=100,
            lr=5e-5,
    ):
        super(TrainDiffusion, self).__init__()
        self.lr = lr
        self.num_ddim_timesteps = num_ddim_timesteps
        self.diffsuion = GaussianDiffusion(mean_type=unet_pred)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(device)
        self.save_path = os.path.join(get_project_path(), "pretrained", "ddim_eps_64.pth")
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
                img = img.to(device)
                batch_size = img.shape[0]
                # t = self.diffsuion.get_rand_t(batch_size, device)
                t = self.diffsuion.get_ddim_rand_t(batch_size, device, self.num_ddim_timesteps)
                loss = self.diffsuion.training_losses(model=self.unet, x_start=img, t=t)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            # self.test()
            epoch_loss /= count
            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

        end = time()
        seconds = int(end - start)
        minutes = seconds // 60
        remain_second = seconds % 60
        print(f"time consumed: {minutes}min{remain_second}s")

    def test_ddim(self):
        self.unet.eval()
        spacediffusion = SpacedDiffusion(self.num_ddim_timesteps)
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(device)
        noise = torch.randn_like(imgs)
        final_sample = spacediffusion.ddim_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)
        draw_img_groups([noise, final_sample])

    def test_ddpm(self):
        self.unet.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(device)
        noise = torch.randn_like(imgs)
        final_sample = self.diffsuion.p_sample_loop(self.unet, shape=imgs.shape, noise=noise, progress=True)
        draw_img_groups([noise, final_sample])

    def test_restore(self):
        self.unet.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(device)
        noise_imgs, restore_imgs = self.diffsuion.restore_img(model=self.unet, x_start=imgs, t=30)
        draw_img_groups([imgs, noise_imgs, restore_imgs])


class TrainDiffusion_x0:
    def __init__(
            self,
            unet_pred=ModelMeanType.START_X,
            num_ddim_timesteps=100,
            lr=5e-5,
    ):
        super(TrainDiffusion_x0, self).__init__()
        self.lr = lr
        self.num_ddim_timesteps = num_ddim_timesteps
        self.diffsuion = GaussianDiffusion(mean_type=unet_pred)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(device)
        self.save_path = os.path.join(get_project_path(), "pretrained", "ddim_x0_64.pth")
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
                img = img.to(device)
                batch_size = img.shape[0]
                # t = self.diffsuion.get_rand_t(batch_size, device)
                t = torch.LongTensor([50] * batch_size).to(device)
                loss = self.diffsuion.training_losses(model=self.unet, x_start=img, t=t)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                scheduler.step()

            torch.save(self.unet.state_dict(), self.save_path)

            # self.test()
            epoch_loss /= count
            print(f"Epoch:{epoch+1}/{epochs}  Loss:{epoch_loss:.8f}")

        end = time()
        seconds = int(end - start)
        minutes = seconds // 60
        remain_second = seconds % 60
        print(f"time consumed: {minutes}min{remain_second}s")

    def test_restore(self):
        self.unet.eval()
        imgs, _ = next(iter(testloader))
        imgs = imgs.to(device)
        noise_imgs, restore_imgs = self.diffsuion.restore_img_0(self.unet, imgs, t=10)
        draw_ori_noise_recon_images16(imgs, noise_imgs, restore_imgs)


def main():
    # trainer = TrainDiffusion(num_ddim_timesteps=100)
    # trainer.train(100)
    # trainer.test_ddim()
    # trainer.test_ddpm()
    # trainer.test_restore()
    trainer = TrainDiffusion_x0()
    # trainer.train(100)
    trainer.test_restore()


if __name__ == "__main__":
    main()



