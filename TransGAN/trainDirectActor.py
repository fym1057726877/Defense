import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from ActorModel import Resnet34Actor
from TransGAN.TransGanModel import SwinTransGenerator


# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"

is_local = True
# is_local = False
# is_peg = True
is_peg = False

device = "cuda"
batch_size = 32
lr = 5e-5
total_epochs = 1000

loss_type = 'mse_loss'


def getDataLoader():
    # customize this function according to the path of your own data
    pass


class DirectTrainActorReconstruct:
    def __init__(self):
        super(DirectTrainActorReconstruct, self).__init__()
        self.trainActor_ckp_path = os.path.join(
            "DirectActor",
            "actor.ckp"
        )
        self.actor = Resnet34Actor().to(device)
        self.generator = SwinTransGenerator(is_local=is_local, is_peg=is_peg)
        self.generator.load_state_dict(torch.load("path of g"))
        self.generator.to(device)

    def train(self):
        trainDataloader = getDataLoader()
        optimizer_actor = optim.Adam(self.actor.parameters(), self.lr)
        lr_scheduler_actor = optim.lr_scheduler.StepLR(optimizer_actor, step_size=1, gamma=0.99)
        mse_loss_fun = nn.MSELoss()
        for epoch in range(self.total_epochs):
            iter_object = trainDataloader
            self.actor.train()
            for index, (img, label) in enumerate(iter_object):
                img = img.to(device)
                label = label.to(device)
                optimizer_actor.zero_grad()
                z = self.actor(img)
                recImg = self.generator(z)
                train_loss = mse_loss_fun(img, recImg)
                train_loss.backward()
                optimizer_actor.step()
                reward = F.mse_loss(img, recImg, reduction="none")
                reward = reward.view(reward.size(0), -1)
                reward = reward.sum(1).unsqueeze(1).mean()
                print("[Train loss: %f] [Reward: %f]" % (train_loss.item(), reward.item()))
            if optimizer_actor.state_dict()["param_groups"][0]["lr"] > self.lr / 1e2:
                lr_scheduler_actor.step()
            torch.save(self.actor.state_dict(), self.trainActor_ckp_path)


def trainDirectActor():
    # seed = 1  # seed for random function
    directActorRec = DirectTrainActorReconstruct()
    directActorRec.train()


if __name__ == "__main__":
    trainDirectActor()
