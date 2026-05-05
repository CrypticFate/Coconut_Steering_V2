import sys
from typing import Callable, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, "external/coconut")
from coconut import Coconut  # noqa: E402


class SteeredCoconut(Coconut):
    """
    Coconut wrapper with latent steering: h_t + alpha * gamma^t * std(h_t) * v_truth.
    """

    def __init__(
        self,
        *args,
        alpha_fn: Optional[Callable[[], torch.Tensor]] = None,
        v_truth: Optional[torch.Tensor] = None,
        gamma: float = 0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_fn = alpha_fn
        self.v_truth = v_truth
        self.gamma = gamma
        self._h_seq = []

    def _steer(self, h_t: torch.Tensor, t: int):
        if self.alpha_fn is None or self.v_truth is None:
            return h_t
        v = self.v_truth.to(h_t.device)
        v = v / (v.norm() + 1e-12)
        sigma = h_t.std(dim=-1, keepdim=True)
        delta = self.alpha_fn() * (self.gamma**t) * sigma * v
        h_out = h_t + delta
        self._h_seq.append(h_out)
        return h_out

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        logits = []
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if len(latent_lists) else 0
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None
        self._h_seq = []

        for pass_idx in range(max_n_latents):
            out = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                attention_mask=attention_mask[:, : next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=True,
            )
            logits.append(out.logits)
            kv_cache = out.past_key_values
            h_last = out.hidden_states[-1][:, -1, :]
            h_steered = self._steer(h_last, pass_idx)

            for b in range(input_ids.shape[0]):
                if pass_idx < len(latent_lists[b]):
                    pos = latent_lists[b][pass_idx]
                    inputs_embeds[b, pos, :] = h_steered[b]

            if pass_idx + 1 < max_n_latents:
                next_start = next_compute_range[1]
                next_end = min(
                    latent_indices[:, 1][latent_indices[:, 1] > next_start].min().item(),
                    input_ids.shape[1],
                )
                next_compute_range = (next_start, next_end)
            else:
                next_compute_range = (next_compute_range[1], input_ids.shape[1])

        if next_compute_range[0] < next_compute_range[1]:
            out = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                attention_mask=attention_mask,
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=False,
            )
            logits.append(out.logits)

        full_logits = torch.cat(logits, dim=1)
        shift_logits = full_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return type("Out", (), {"loss": loss, "logits": full_logits})()
