import os
import torch
from ConvGanModel import ConvGenerator
from reconstruct import GradientDescentReconstruct
from utils import get_project_path
from data.data_600_classes import testloader
from utils import draw_img_groups


if __name__ == '__main__':
    model = ConvGenerator(latent_dim=256).to("cuda")
    save_path = os.path.join(
        get_project_path(project_name="Defense"),
        "pretrained",
        "ConvGenerator.pth"
    )
    model.load_state_dict(torch.load(save_path))
    img, _ = next(iter(testloader))
    img = img.to("cuda")
    rec_img = GradientDescentReconstruct(
        img=img,
        generator=model,
        device="cuda",
        L=200,
        R=20,
        lr=0.001,
        latent_dim=256
    )
    draw_img_groups([img, rec_img])

