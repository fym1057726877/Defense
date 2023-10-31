import time
import torch
from tqdm import tqdm
import numpy as np
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack


def advAttack(classifier, x, attack_type, eps):
    if attack_type == "RandFGSM":
        alpha = 0.005
        x = x + alpha * torch.sign(torch.randn(x.shape).to(x.device))
        # x = torch.clip(x + alpha * torch.sign(torch.randn(x.shape).to(device)), 0, 1)
        eps2 = eps - alpha
        x_adv = fast_gradient_method(classifier, x, eps2, np.inf)
    elif attack_type == "FGSM":
        x_adv = fast_gradient_method(classifier, x, eps, np.inf)
    elif attack_type == "PGD":
        x_adv = projected_gradient_descent(classifier, x, eps=eps, eps_iter=1 / 255,
                                           nb_iter=min(255 * eps + 4, 1.25 * (eps * 255)), norm=np.inf)
    elif attack_type == "HSJA":
        x_adv = hop_skip_jump_attack(classifier, x, norm=2, initial_num_evals=1, max_num_evals=50,
                                     num_iterations=30, batch_size=20, verbose=False)
    else:
        raise RuntimeError(f"The attack type {attack_type} is invalid!")
    return x_adv


def generateAdvImage(
        classifier,
        attack_dataloder,
        savepath=None,
        attack_type="FGSM",
        eps=0.03,
        device="cuda",
        progress=False
):
    print(f"-------------------------------------------------\n"
          f"Generating Adversarial Examples ...\n"
          f"eps = {eps} attack = {attack_type}")
    time.sleep(1)

    def accuracy(y, y1):
        return (y.max(dim=1)[1] == y1).sum()

    dataloader = attack_dataloder
    train_acc, adv_acc, train_n = 0, 0, 0
    normal_data, adv_data, label_data = None, None, None
    if progress:
        indice = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        indice = enumerate(dataloader)
    for index, (x, label) in indice:
        x, label = x.to(device), label.to(device)
        pred = classifier(x)
        train_acc += accuracy(pred, label)

        x_adv = advAttack(classifier=classifier, x=x, attack_type=attack_type, eps=eps)
        y_adv = classifier(x_adv)
        adv_acc += accuracy(y_adv, label)

        train_n += label.size(0)

        x, x_adv, label = x.data, x_adv.data, label.data
        if normal_data is None:
            normal_data, adv_data, label_data = x, x_adv, label
        else:
            normal_data = torch.cat((normal_data, x))
            adv_data = torch.cat((adv_data, x_adv))
            label_data = torch.cat((label_data, label))

    print(f"Accuracy(normal) {torch.true_divide(train_acc, train_n):.6f}\n"
          f"Accuracy({attack_type}) {torch.true_divide(adv_acc, train_n):.6f}\n"
          f"-------------------------------------------------")

    adv_data = {"normal": normal_data, "adv": adv_data, "label": label_data}
    torch.save(adv_data, savepath)
    return adv_data


