import torch
from torch import nn
import torch.nn.functional as F


# memory
class MemoryUnit(nn.Module):
    def __init__(self, MEM_DIM, features, addressing="sparse"):
        super(MemoryUnit, self).__init__()

        self.MEM_DIM = MEM_DIM
        self.features = features

        self.memory = torch.zeros((self.MEM_DIM, self.features))
        nn.init.kaiming_uniform_(self.memory)
        self.memory = nn.Parameter(self.memory)

        self.Cosine_Similiarity = nn.CosineSimilarity(dim=2)
        self.addressing = addressing
        if self.addressing == 'sparse':
            self.threshold = 1 / self.MEM_DIM
            self.epsilon = 1e-12

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shape = x.shape
        dim = len(shape)

        if dim == 2:
            b, features = shape
            z = x
        elif dim == 3:
            b, features = shape[0], shape[1] * shape[2]
            z = x.view(b, features).contiguous()
        elif dim == 4:
            b, features = shape[0], shape[1] * shape[2] * shape[3]
            z = x.view(b, features).contiguous()
        else:
            raise NotImplementedError

        assert self.features == features

        ex_mem = self.memory.unsqueeze(0).repeat(b, 1, 1)  # [b, mem_dim, fea]
        ex_z = z.unsqueeze(1).repeat(1, self.MEM_DIM, 1)  # [b, mem_dim, fea]

        mem_logit = self.Cosine_Similiarity(ex_z, ex_mem)  # [b, mem_dim]
        mem_weight = mem_logit.softmax(dim=1)  # [b, num_mem]

        # soft寻址和稀疏寻址
        z_hat = None
        if self.addressing == "soft":
            z_hat = torch.matmul(mem_weight, self.memory)
        elif self.addressing == "sparse":
            mem_weight = (self.relu(mem_weight - self.threshold) * mem_weight) \
                         / (torch.abs(mem_weight - self.threshold) + self.epsilon)
            mem_weight = F.normalize(mem_weight, p=1, dim=1)
            z_hat = torch.matmul(mem_weight, self.memory)

        assert z_hat is not None, "model parameter：addressing is wrong"

        out = z_hat.view(shape).contiguous()

        return out

def test_memory():
    x = torch.randn((16, 192, 7, 7))
    model = MemoryUnit(MEM_DIM=200, features=192 * 7 * 7, addressing="sparse")
    out = model(x)
    print(out.shape)

