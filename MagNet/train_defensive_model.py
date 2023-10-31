import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from defensive_models import DefensiveModel1
from utils import get_project_path
from data.data_600_classes import trainloader, testloader
from utils import draw_img_groups
from classifier.models import TargetedModelB


class Trainer:
    def __init__(
            self,
            lr=5e-4,
            device="cuda",
    ):
        super(Trainer, self).__init__()
        self.device = device
        self.train_loader, self.test_loader = trainloader, testloader
        self.num_classes = 600
        self.weight_noise = 0.1

        # magnet
        self.model = DefensiveModel1(in_channels=1).to(self.device)
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "magnet.pth")
        self.model.load_state_dict((torch.load(self.save_path)))

        # loss function
        self.loss_fun = nn.MSELoss()

        # optimizer
        self.lr = lr
        self.optimer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self, epochs):
        for e in range(epochs):
            train_num = 0
            epoch_loss = 0
            self.model.train()
            batch_count = len(self.train_loader)
            for index, (img, label) in tqdm(enumerate(self.train_loader), desc=f"train {e+1}/{epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img_ori, label = img.to(self.device), label.to(self.device)
                img_noise = img_ori + self.weight_noise * torch.randn_like(img, device=self.device)
                # img_noise = torch.clamp(img_noise, 0, 1)

                rec_x = self.model(img_noise)
                loss = self.loss_fun(rec_x, img_ori)

                loss.backward()
                self.optimer.step()

                train_num += label.size(0)
                epoch_loss += loss

                # if index % 200 == 0 and index != 0:
                #     draw_img_groups([img_ori, rec_x], imgs_every_row=8, block=False)

            if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                self.scheduler.step()

            epoch_loss /= batch_count
            print(f"[Epoch {e+1}/{epochs}   Loss:{epoch_loss:.6f}]")
            torch.save(self.model.state_dict(), self.save_path)

    def eval_rec(self):
        self.model.eval()
        imgs, labels = next(iter(self.test_loader))
        imgs = imgs.to(self.device)
        x_recon = self.model(imgs)
        draw_img_groups([imgs, x_recon])

    def eval_classifier(self):
        target_classifier = TargetedModelB(num_classes=self.num_classes).to(self.device)
        target_classifier_path = os.path.join(
            get_project_path(project_name="Defense"),
            "pretrained",
            "ModelB.pth"
        )
        target_classifier.load_state_dict(torch.load(target_classifier_path))
        target_classifier.eval()

        def accurary(y_pred, y):
            return (y_pred.max(dim=1)[1] == y).sum()

        total = 0
        acc_ori, acc_rec = 0., 0.
        for i, (img, label) in tqdm(enumerate(self.test_loader), total=len(testloader), desc="eval classifier"):
            total += img.shape[0]
            img, label = img.to(self.device), label.to(self.device)

            y_ori = target_classifier(img)
            acc_ori += accurary(y_ori, label)

            rec_x = self.model(img)
            y_rec = target_classifier(rec_x)
            acc_rec += accurary(y_rec, label)

        acc_ori /= total
        acc_rec /= total
        print(f"acc_ori:{acc_ori:.8f}\n"
              f"acc_rec:{acc_rec:.8f}")


def main(device="cuda"):
    train_model = Trainer(lr=5e-4, device=device)
    train_model.train(50)
    train_model.eval_rec()
    train_model.eval_classifier()


if __name__ == "__main__":
    main()
