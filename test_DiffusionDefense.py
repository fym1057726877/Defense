import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from utils import get_project_path
from classifier.attackClassifier import generateAdvImage
from classifier.models import getDefinedClsModel
from data.data_600_classes import testloader
from MemoryDiffusion.models import UNetModel, GaussianDiffusion, ModelMeanType
from MemoryDiffusion.respace import iter_denoise

# dataset_name = "Handvein"
# dataset_name = "Handvein3"
dataset_name = "Fingervein2"


class DirectTrainActorReconstruct:
    def __init__(
            self,
            target_classifier_name="ModelB",
            attack_type="FGSM",   # "RandFGSM", "FGSM", "PGD", "HSJA"
            eps=0.03,
    ):
        super(DirectTrainActorReconstruct, self).__init__()
        self.batch_size = 20
        self.device = "cuda"
        self.attack_dataloder = testloader
        self.attack_type = attack_type
        self.eps = eps

        self.num_classes = 600

        self.diffsuion = GaussianDiffusion(mean_type=ModelMeanType.EPSILON)
        # self.diffsuion = GaussianDiffusion(mean_type=ModelMeanType.START_X)
        self.unet = UNetModel(
            in_channels=1,
            model_channels=64,
            out_channels=1,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
        ).to(self.device)
        self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "diffusion.pth")
        # self.save_path = os.path.join(get_project_path(project_name="Defense"), "pretrained", "diffusion_pred_x0.pth")
        self.unet.load_state_dict(torch.load(self.save_path))

        # classifier B
        self.target_classifier_name = target_classifier_name
        self.target_classifier = getDefinedClsModel(
            dataset_name=dataset_name,
            model_name=self.target_classifier_name,
            num_classes=self.num_classes,
            device=self.device
        )
        self.target_classifier_path = os.path.join(
            get_project_path(project_name="Defense"), "pretrained", f"{self.target_classifier_name}.pth")
        self.target_classifier.load_state_dict(torch.load(self.target_classifier_path))

        # adversial
        self.adv_path = os.path.join(
            get_project_path(project_name="Defense"),
            "data",
            "adv_imgs",
            f"600_{self.target_classifier_name}_{self.attack_type}_{self.eps}.pth"
        )

    def defense(self, img, t=50):
        # rec = self.diffsuion.restore_img(model=self.unet, x_start=img, t=t)
        rec = iter_denoise(unet_model=self.unet, imgs=img, t=t)
        return rec

    def test(self, progress=False):
        advDataloader = self.getAdvDataLoader(
            progress=True,
            shuffle=True
        )

        self.target_classifier.eval()

        def accuracy(y, y1):
            return (y.max(dim=1)[1] == y1).sum()

        normal_acc, rec_acc, adv_acc, rec_adv_acc, diff_adv_acc, num = 0, 0, 0, 0, 0, 0
        total_num = len(advDataloader)

        iterObject = enumerate(advDataloader)
        if progress:
            iterObject = tqdm(iterObject, total=total_num)

        for i, (img, adv_img, label) in iterObject:
            img, label = img.to(self.device), label.to(self.device)

            normal_y = self.target_classifier(img)
            normal_acc += accuracy(normal_y, label)

            rec_img = self.defense(img)
            rec_y = self.target_classifier(rec_img)
            rec_acc += accuracy(rec_y, label)

            adv_img = adv_img.to(self.device)
            adv_y = self.target_classifier(adv_img)
            adv_acc += accuracy(adv_y, label)

            rec_adv_img = self.defense(adv_img)
            rec_adv_y = self.target_classifier(rec_adv_img)
            rec_adv_acc += accuracy(rec_adv_y, label)

            num += label.size(0)

        print(f"-------------------------------------------------\n"
              f"test result:\n"
              f"NorAcc:{torch.true_divide(normal_acc, num).item():.4f}\n"
              f"RecAcc:{torch.true_divide(rec_acc, num).item():.4f}\n"
              f"AdvAcc:{torch.true_divide(adv_acc, num).item():.4f}\n"
              f"RAvAcc:{torch.true_divide(rec_adv_acc, num).item():.4f}\n"
              f"-------------------------------------------------")

    def getAdvDataLoader(
            self,
            progress=False,
            shuffle=False
    ):
        if os.path.exists(self.adv_path):
            data_dict = torch.load(self.adv_path)
        else:
            data_dict = generateAdvImage(
                classifier=self.target_classifier,
                attack_dataloder=self.attack_dataloder,
                attack_type=self.attack_type,
                eps=self.eps,
                progress=progress,
                savepath=self.adv_path
            )
        normal_data = data_dict["normal"]
        adv_data = data_dict["adv"]
        label = data_dict["label"]
        dataloder = DataLoader(TensorDataset(normal_data, adv_data, label), batch_size=self.batch_size, shuffle=shuffle)
        return dataloder


def testDirectActor():
    # seed = 1  # the seed for random function
    torch.manual_seed(10)
    # attack_type = 'FGSM'
    attack_type = 'PGD'
    # attack_type = 'RandFGSM'
    # attack_type = 'HSJA'
    if attack_type == 'RandFGSM':
        # eps = 0.03
        # eps = 0.1
        eps = 0.3
    elif attack_type == "PGD":
        # eps = 0.03
        # eps = 0.1
        eps = 0.3
    else:
        # eps = 0.03
        # eps = 0.1
        eps = 0.3
    # classifier_name = "Resnet18"
    # classifier_name = "GoogleNet"
    # classifier_name = "ModelB"
    # classifier_name = "MSMDGANetCnn_wo_MaxPool"
    # classifier_name = "Tifs2019Cnn_wo_MaxPool"
    # classifier_name = "FVRASNet_wo_Maxpooling"
    # classifier_name = "LightweightDeepConvNN"

    directActorRec = DirectTrainActorReconstruct(
        target_classifier_name="ModelB",
        attack_type=attack_type,
        eps=eps
    )
    directActorRec.test(progress=True)


if __name__ == "__main__":
    testDirectActor()
