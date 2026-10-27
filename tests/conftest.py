"""Small tensor kernels are faster and more reproducible with one CPU thread."""

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)
