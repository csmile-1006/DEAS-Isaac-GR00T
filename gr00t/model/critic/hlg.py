import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        support = torch.linspace(self.min_value, self.max_value, self.num_bins + 1, device=target.device)
        cdf_evals = torch.special.erf(
            (support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0, device=target.device)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        support = torch.linspace(self.min_value, self.max_value, self.num_bins + 1, device=probs.device)
        centers = (support[:-1] + support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)
