import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from memorydef import MemoryDefense
from utils import get_project_path
from data.data_600_classes import trainloader, testloader
from utils import draw_img_groups
from classifier.models import Resnet50, TargetedModelB


class EntropyLoss(nn.Module):
    def __init__(self, entropy_loss_coef=0.0002):
        super(EntropyLoss, self).__init__()
        self.entropy_loss_coef = entropy_loss_coef

    def forward(self, mem_weight):
        entropy_loss = -mem_weight * torch.log(mem_weight + 1e-12)
        entropy_loss = entropy_loss.sum()
        entropy_loss *= self.entropy_loss_coef
        return entropy_loss


class Trainer:
    def __init__(
            self,
            lr=5e-4,
            device="cuda",
            beta=1e-4,

    ):
        super(Trainer, self).__init__()
        self.device = device
        self.train_loader, self.test_loader = trainloader, testloader
        self.num_memories = 1200
        self.num_classes = 600

        # MemoryDefense
        self.memory = MemoryDefense(
            MEM_DIM=self.num_memories,
            num_classes=self.num_classes,
            device=self.device
        ).to(self.device)
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "memorydefense.pth")
        self.memory.load_state_dict((torch.load(self.save_path)))

        self.classifer_path = os.path.join(
            get_project_path(project_name="Defense"), "pretrained", "resnet50_classifier.pth")
        self.classifierA = Resnet50(num_classes=self.num_classes).to(self.device)
        self.classifierA.load_state_dict(torch.load(self.classifer_path))

        # loss function
        self.loss_fun1 = nn.MSELoss()
        self.loss_fun2 = EntropyLoss()
        self.beta = beta

        # optimizer
        self.lr = lr
        self.optimer = optim.AdamW(self.memory.parameters(), lr=self.lr, weight_decay=0.005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self, epochs):
        for e in range(epochs):
            train_num = 0
            epoch_loss = 0
            self.memory.train()
            batch_count = len(self.train_loader)
            for index, (img, label) in tqdm(enumerate(self.train_loader), desc=f"train {e+1}/{epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                out = self.memory(img, label)
                rec_x, mem_weight = out["rec_x"], out["mem_weight"]
                rec_x_hat, mem_weight_hat = out["rec_x_hat"], out["mem_weight_hat"]
                loss_target = self.loss_fun1(rec_x, img) + self.loss_fun2(mem_weight)
                loss_non_target = self.loss_fun1(rec_x_hat, img) + self.loss_fun2(mem_weight_hat)
                loss = loss_target - self.beta * loss_non_target.sigmoid()
                loss.backward()
                self.optimer.step()

                train_num += label.size(0)
                epoch_loss += loss

                if index % 100 == 0:
                    draw_img_groups([img, rec_x], imgs_every_row=8, block=False)

            if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                self.scheduler.step()

            epoch_loss /= batch_count
            print(f"[Epoch {e+1}/{epochs}   Loss:{epoch_loss:.6f}]")
            torch.save(self.memory.state_dict(), self.save_path)

    def eval_rec(self):
        self.memory.eval()
        imgs, labels = next(iter(self.test_loader))
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        label_pred = self.classifierA(imgs).max(dim=1)[1]
        x_recon = self.memory(imgs, label_pred)["rec_x"]
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
        self.classifierA.eval()

        def accurary(y_pred, y):
            return (y_pred.max(dim=1)[1] == y).sum()

        total = 0
        acc_ori, acc_rec = 0., 0.
        for i, (img, label) in tqdm(enumerate(self.test_loader), total=len(testloader), desc="eval classifier"):
            total += img.shape[0]
            img, label = img.to(self.device), label.to(self.device)

            y_ori = target_classifier(img)
            acc_ori += accurary(y_ori, label)

            label_pred = self.classifierA(img).max(dim=1)[1]
            rec_x = self.memory(img, label_pred)["rec_x"]
            y_rec = target_classifier(rec_x)
            acc_rec += accurary(y_rec, label)

        acc_ori /= total
        acc_rec /= total
        print(f"acc_ori:{acc_ori:.8f}\n"
              f"acc_rec:{acc_rec:.8f}")


def main(device="cuda"):
    train_model = Trainer(lr=5e-4, device=device)
    # train_model.train(100)
    # train_model.eval_rec()
    train_model.eval_classifier()


if __name__ == "__main__":
    main()
