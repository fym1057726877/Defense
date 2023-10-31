import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import get_project_path
from data.data_600_classes import trainloader, testloader
from models import getDefinedClsModel

# lr = 1e-5
# lr = 5e-5
# lr = 1e-4
# batchsize = 48
# total_epochs = 1000
# device = "cuda"


# dataset_name = "Handvein"
# dataset_name = "Handvein3"
# dataset_name = "Fingervein2"

# model_name = "Resnet18"
# model_name = "GoogleNet"
# model_name = "ModelB"
# model_name = "MSMDGANetCnn_wo_MaxPool"   # PV-CNN
# model_name = "Tifs2019Cnn_wo_MaxPool"   # FV-CNN
# model_name = "FVRASNet_wo_Maxpooling"
# model_name = "LightweightDeepConvNN"


def accurary(y_pred, y):
    return (y_pred.max(dim=1)[1] == y).sum()


class TrainClassifier:
    def __init__(
            self,
            lr=5e-5,
            device="cuda",
            dataset_name="Fingervein2",
            model_name="ModelB",
    ):
        super(TrainClassifier, self).__init__()
        # customize these object according to the path of your own data
        self.device = device
        self.train_loader, self.test_loader = trainloader, testloader
        self.classifier = getDefinedClsModel(
            model_name=model_name,
            dataset_name=dataset_name,
            num_classes=600,
            device=device
        )
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", f"{model_name}.pth")
        self.classifier.load_state_dict(torch.load(self.save_path))

        # loss function
        self.loss_fun = nn.CrossEntropyLoss()
        self.lr = lr

        self.optimer = optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=10, gamma=0.99)

    def train(self, epochs):
        for e in range(epochs):
            correct_num = 0
            train_num = 0
            epoch_loss = 0
            self.classifier.train()
            # start train
            batch_count = len(self.train_loader)
            for index, (img, label) in tqdm(enumerate(self.train_loader), desc=f"train {e+1}/{epochs}",
                                            total=batch_count):
                self.optimer.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                pred = self.classifier(img)
                cls_loss = self.loss_fun(pred, label.long())

                cls_loss.backward()
                self.optimer.step()

                correct_num += (pred.max(dim=1)[1] == label).sum()
                train_num += label.size(0)
                epoch_loss += cls_loss

            if self.optimer.state_dict()['param_groups'][0]['lr'] > self.lr / 1e2:
                self.scheduler.step()

            train_acc = correct_num / train_num
            test_acc = self.eval(show=False)
            epoch_loss /= batch_count
            print(
                f"[Epoch {e}/{epochs}   Loss:{epoch_loss:.6f}   "
                f"Train_acc: {train_acc:.6f}   Test_acc: {test_acc:.6f}]\n"
            )
            torch.save(self.classifier.state_dict(), self.save_path)

    def eval(self, show=True):
        correct_num, eval_num = 0, 0
        self.classifier.eval()
        for index, (x, label) in tqdm(enumerate(self.test_loader), desc="test step", total=len(self.test_loader)):
            x, label = x.to(self.device), label.to(self.device)
            pred = self.classifier(x)
            correct_num += accurary(pred, label)
            eval_num += label.size(0)
        acc = correct_num / eval_num
        if show:
            print(f"acc:{acc:.6f}")
        return acc


def main_Cls(dataset_name="Handvein3", model_name="ModelB", device="cuda"):
    # seed = 1  # the seed for random function
    train_Classifier = TrainClassifier(
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
    )
    # train_Classifier.train(20)
    train_Classifier.eval()


if __name__ == "__main__":
    main_Cls()
