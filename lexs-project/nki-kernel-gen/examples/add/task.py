import torch


def reference(a, b):
    return a + b


def cases(seed):
    g = torch.Generator().manual_seed(seed)
    for shape in [(128, 512), (256, 1024), (129, 513)]:
        yield tuple(torch.randn(shape, generator=g) for _ in range(2))
    yield (torch.zeros(128, 512), torch.zeros(128, 512))


RTOL = 1e-5
ATOL = 1e-6
