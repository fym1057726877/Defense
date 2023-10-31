import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryDefense(nn.Module):
    def __init__(self, MEM_DIM=1200, num_classes=600, device="cpu"):
        super(MemoryDefense, self).__init__()
        assert MEM_DIM % num_classes == 0
        self.device = device
        self.num_classes = num_classes

        self.image_height = 64
        self.image_width = 64
        self.image_channel_size = 1

        self.num_memories = MEM_DIM
        self.addressing = 'sparse'

        self.conv_channel_size = 4
        self.feature_size = self.conv_channel_size * 16 * 9 * 9
        self.drop_rate = 0.5
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

        # Encoder
        self.encoder = Encoder()

        # Memory
        init_mem = torch.zeros(self.num_memories, self.feature_size)
        nn.init.kaiming_uniform_(init_mem)
        self.memory = nn.Parameter(init_mem)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        self.hardshrink = nn.Hardshrink(lambd=1e-12)  # 1e-12

        # Decoder
        self.decoder = Decoder()

        if self.addressing == 'sparse':
            self.threshold = 1 / self.memory.size(0)
            self.epsilon = 1e-12

    def forward(self, x, labels):
        # Encoder
        x = self.encoder(x)
        batch, _, _, _ = x.size()
        z = x.view(batch, -1)

        # Memory
        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)
        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)

        # Masking using one hot encoding scheme over memory slots.
        m1, m2 = self.masking(labels)  # Generating Mask
        masked_mem_weight = mem_weight * m1  # Masking target class
        masked_mem_weight_hat = mem_weight * m2  # Masking non-target class

        z_hat_target = z_hat_non_target = None
        if self.addressing == 'soft':
            z_hat_target = torch.mm(masked_mem_weight, self.memory)  # Matrix Multiplication:  Att_W x Mem
            z_hat_non_target = torch.mm(masked_mem_weight_hat, self.memory)  # Matrix Multiplication:  Att_W x Mem

        elif self.addressing == 'sparse':
            # Unmask Weight Target Class
            masked_mem_weight = self.hardshrink(masked_mem_weight)
            masked_mem_weight = masked_mem_weight / masked_mem_weight.norm(p=1, dim=1).unsqueeze(1).expand(
                batch, self.num_memories)
            z_hat_target = torch.mm(masked_mem_weight, self.memory)
            # Mask Weight Non-target Class
            masked_mem_weight_hat = self.hardshrink(masked_mem_weight_hat)
            masked_mem_weight_hat = masked_mem_weight_hat / masked_mem_weight_hat.norm(p=1, dim=1).unsqueeze(1).expand(
                batch, self.num_memories)
            z_hat_non_target = torch.mm(masked_mem_weight_hat, self.memory)

        # Decoder
        rec_x = self.decoder(z_hat_target)
        rec_x_hat = self.decoder(z_hat_non_target)

        return dict(encoded=z_hat_target, rec_x=rec_x, rec_x_hat=rec_x_hat,
                    mem_weight=masked_mem_weight, mem_weight_hat=masked_mem_weight_hat)

    def masking(self, label):
        memoryPerClass = self.num_memories // self.num_classes
        batch_size = len(label)

        mask1 = torch.zeros(batch_size, self.num_memories)
        mask2 = torch.ones(batch_size, self.num_memories)
        ones = torch.ones(memoryPerClass)
        zeros = torch.zeros(memoryPerClass)

        for i in range(batch_size):
            lab = torch.arange(memoryPerClass * label[i], memoryPerClass * (label[i] + 1), dtype=torch.long)
            if lab.nelement() == 0:
                print("Label tensor empty in the memory module.")
            else:
                mask1[i, lab] = ones
                mask2[i, lab] = zeros
        return mask1.to(self.device), mask2.to(self.device)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        self.en = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.25, True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.25, True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.en(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_channel_size = 16
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.25, True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.25, True),

            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=1)
        )

    def forward(self, x):
        x = x.view(-1, self.conv_channel_size * 4, 9, 9)
        x = self.dec(x)
        return x


def test_encoder():
    e = Encoder()
    x = torch.randn((16, 1, 64, 64))
    x = e(x)
    print(x.shape)


def test_decoder():
    d = Decoder()
    x = torch.randn((16, 64, 9, 9))
    x = d(x)
    print(x.shape)


def test_memory():
    m = MemoryDefense(MEM_DIM=1200)
    x = torch.randn((16, 1, 64, 64))
    l = torch.randint(0, 600, (16,))
    out = m(x, l)
    rec_x = out["rec_x"]
    w = out["mem_weight"]
    print(rec_x.shape)
    print(w.shape)


if __name__ == '__main__':
    test_encoder()
    test_decoder()
    test_memory()
