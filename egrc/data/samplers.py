import torch
import numpy as np


class CategoriesSampler:
    def __init__(
        self,
        label,
        n_batch,
        n_cls,
        n_per,
        seed=None,
        repeat_manifest=True,
    ):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.seed = None if seed is None else int(seed)
        self.repeat_manifest = bool(repeat_manifest)
        self.generator = None
        if self.seed is not None and not self.repeat_manifest:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        generator = None
        if self.seed is not None:
            if self.repeat_manifest:
                generator = torch.Generator()
                generator.manual_seed(self.seed)
            else:
                generator = self.generator
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(
                len(self.m_ind),
                generator=generator,
            )[: self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(
                    len(l),
                    generator=generator,
                )[: self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
