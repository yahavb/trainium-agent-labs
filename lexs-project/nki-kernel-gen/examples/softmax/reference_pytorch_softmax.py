import torch


def reference(x):
    return torch.softmax(x, dim=-1)


def cases(seed):
    generator = torch.Generator().manual_seed(seed)
    for shape in [(128, 512), (256, 1024), (129, 513)]:
        yield (torch.randn(shape, generator=generator),)
    yield (torch.zeros(128, 512),)
    yield (torch.randn(128, 512, generator=generator) * 50,)


RTOL = 1e-4
ATOL = 1e-6
