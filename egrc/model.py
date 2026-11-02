"""Main EGRC model, extracted from the established full-method computation.

Parameter names, the zero message slot, and diagnostic parameters are retained
for checkpoint and seeded numerical equivalence with the original experiments.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class EGRCConfig:
    dim: int = 384
    relation_dim: int = 96
    relation_hidden_dim: int = 128
    task_hidden_dim: int = 64
    num_steps: int = 2
    positive_topk: int = 10
    evidence_max: float = 20.0
    update_scale_max: float = 0.2
    proto_temperature: float = 0.1
    dropout: float = 0.1
    classifier_dropout: float = 0.0

    def __post_init__(self):
        if min(self.dim, self.relation_dim, self.relation_hidden_dim, self.task_hidden_dim) < 1:
            raise ValueError("Feature dimensions must be positive")
        if self.num_steps < 1 or self.positive_topk < 1:
            raise ValueError("num_steps and positive_topk must be positive")
        if self.evidence_max <= 0 or self.update_scale_max < 0 or self.proto_temperature <= 0:
            raise ValueError("Invalid evidence, update, or temperature setting")
        if not 0 <= self.dropout < 1 or not 0 <= self.classifier_dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


def _normalized_entropy(probabilities):
    probabilities = probabilities.clamp_min(1e-08)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    if probabilities.size(-1) <= 1:
        return torch.zeros_like(entropy)
    return entropy / math.log(probabilities.size(-1))


def _pairwise_js_divergence(probabilities):
    left = probabilities[:, None, :].clamp_min(1e-08)
    right = probabilities[None, :, :].clamp_min(1e-08)
    mixture = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - mixture.log())).sum(dim=-1)
        + (right * (right.log() - mixture.log())).sum(dim=-1)
    )


class TaskContextEncoder(nn.Module):
    """Permutation-invariant support geometry encoder."""

    RAW_DIM = 9

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.RAW_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def geometry_statistics(support, prototypes, way, shot):
        gram = prototypes @ prototypes.transpose(0, 1)
        off_diagonal_mask = ~torch.eye(way, dtype=torch.bool, device=gram.device)
        off_diagonal = gram[off_diagonal_mask]
        if off_diagonal.numel() == 0:
            inter_stats = gram.new_zeros(4)
            margin_stats = gram.new_zeros(2)
        else:
            inter_stats = torch.stack(
                [
                    off_diagonal.mean(),
                    off_diagonal.std(unbiased=False),
                    off_diagonal.min(),
                    off_diagonal.max(),
                ]
            )
            masked_gram = gram.masked_fill(~off_diagonal_mask, -1.0)
            nearest_margin = 1.0 - masked_gram.max(dim=-1).values
            margin_stats = torch.stack([nearest_margin.mean(), nearest_margin.std(unbiased=False)])
        support_by_shot = support.reshape(shot, way, support.size(-1))
        if shot > 1:
            scatter = (support_by_shot - prototypes.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=0)
            scatter_stats = torch.stack([scatter.mean(), scatter.std(unbiased=False)])
        else:
            scatter_stats = gram.new_zeros(2)
        shot_indicator = gram.new_tensor([float(shot == 1)])
        return torch.cat([inter_stats, margin_stats, scatter_stats, shot_indicator])

    def forward(self, support, prototypes, way, shot):
        statistics = self.geometry_statistics(support, prototypes, way, shot)
        return (self.network(statistics), statistics)


class TaskConditionedRelation(nn.Module):
    """Task-conditioned pairwise relation tendencies and evidential strengths."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relation_projection = nn.Linear(config.dim, config.relation_dim, bias=False)
        pair_dim = config.relation_dim * 2 + 8
        self.pair_input = nn.Sequential(
            nn.Linear(pair_dim, config.relation_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.relation_hidden_dim),
        )
        self.task_film = nn.Linear(config.relation_hidden_dim, config.relation_hidden_dim * 2)
        nn.init.zeros_(self.task_film.weight)
        nn.init.zeros_(self.task_film.bias)
        self.pair_geometry_scale = None
        self.output = nn.Sequential(
            nn.Linear(config.relation_hidden_dim, config.relation_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.relation_hidden_dim // 2, 2),
        )

    @staticmethod
    def _pair_types(node_count, support_count, dtype, device):
        support = torch.arange(node_count, device=device) < support_count
        support_left = support[:, None]
        support_right = support[None, :]
        pair_index = torch.where(
            support_left & support_right,
            torch.zeros((), dtype=torch.long, device=device),
            torch.where(
                support_left ^ support_right,
                torch.ones((), dtype=torch.long, device=device),
                torch.full((), 2, dtype=torch.long, device=device),
            ),
        )
        return F.one_hot(pair_index, num_classes=3).to(dtype=dtype)

    def forward(self, nodes, beliefs, task_context, support_count, prototype_gram=None):
        projected = self.relation_projection(nodes)
        difference = torch.abs(projected[:, None, :] - projected[None, :, :])
        product = projected[:, None, :] * projected[None, :, :]
        visual_similarity = nodes @ nodes.transpose(0, 1)
        belief_agreement = beliefs @ beliefs.transpose(0, 1)
        js_divergence = _pairwise_js_divergence(beliefs)
        entropy = _normalized_entropy(beliefs)
        entropy_sum = entropy[:, None] + entropy[None, :]
        entropy_difference = torch.abs(entropy[:, None] - entropy[None, :])
        pair_types = self._pair_types(nodes.size(0), support_count, nodes.dtype, nodes.device)
        scalar_features = torch.stack(
            [visual_similarity, belief_agreement, js_divergence, entropy_sum, entropy_difference],
            dim=-1,
        )
        pair_features = torch.cat([difference, product, scalar_features, pair_types], dim=-1)
        hidden = self.pair_input(pair_features)
        gamma, beta = self.task_film(task_context).chunk(2, dim=-1)
        hidden = (1.0 + 0.1 * torch.tanh(gamma)) * hidden + 0.1 * beta
        relation_logit, evidence_logit = self.output(hidden).unbind(dim=-1)
        same_probability = torch.sigmoid(relation_logit)
        total_evidence = self.config.evidence_max * torch.sigmoid(evidence_logit)
        positive_evidence = total_evidence * same_probability
        negative_evidence = total_evidence * (1.0 - same_probability)
        denominator = total_evidence + 2.0
        membership = positive_evidence / denominator
        nonmembership = negative_evidence / denominator
        hesitation = 2.0 / denominator
        diagonal = torch.eye(nodes.size(0), dtype=torch.bool, device=nodes.device)
        membership = membership.masked_fill(diagonal, 0.0)
        nonmembership = nonmembership.masked_fill(diagonal, 0.0)
        hesitation = hesitation.masked_fill(diagonal, 1.0)
        alpha = torch.stack([positive_evidence + 1.0, negative_evidence + 1.0], dim=-1)
        state = {
            "total_evidence": total_evidence.masked_fill(diagonal, 0.0),
            "same_probability": same_probability,
            "alpha": alpha,
            "pair_types": pair_types,
        }
        state.update(membership=membership, nonmembership=nonmembership, hesitation=hesitation)
        return state


def _symmetric_topk_mask(scores, topk):
    node_count = scores.size(0)
    diagonal = torch.eye(node_count, dtype=torch.bool, device=scores.device)
    if topk <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    if topk >= node_count - 1:
        return ~diagonal
    ranking_scores = scores.detach().masked_fill(diagonal, float("-inf"))
    indices = ranking_scores.topk(k=topk, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return (mask | mask.transpose(0, 1)) & ~diagonal


class GraphPropagation(nn.Module):
    """Credibility-selected positive aggregation and residual node updates."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.positive_projection = nn.Linear(config.dim, config.dim, bias=False)
        self.negative_projection = nn.Linear(config.dim, config.dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(config.dim * 3, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.dim),
        )
        self.update_norm = nn.LayerNorm(config.dim)
        self.step_scales = nn.Parameter(torch.full((config.num_steps,), -3.0))

    @staticmethod
    def _normalize_edges(edges):
        return edges / edges.sum(dim=-1, keepdim=True).clamp_min(1e-08)

    def forward(
        self, nodes, edge_state, support_count, step, query_update_weight=None, adjacency=None
    ):
        membership = edge_state["membership"]
        nonmembership = edge_state["nonmembership"]
        hesitation = edge_state["hesitation"]
        selection_scores = membership
        aggregation_scores = membership
        positive_mask = _symmetric_topk_mask(selection_scores, self.config.positive_topk)
        negative_mask = _symmetric_topk_mask(nonmembership, 10)
        positive_edges = aggregation_scores * positive_mask.to(aggregation_scores.dtype)
        negative_edges = nonmembership * negative_mask.to(nonmembership.dtype)
        positive_weights = self._normalize_edges(positive_edges)
        negative_weights = self._normalize_edges(negative_edges)
        positive_values = self.positive_projection(nodes)
        negative_values = self.negative_projection(nodes)
        positive_message = positive_weights @ positive_values
        negative_mass = negative_weights.sum(dim=-1, keepdim=True)
        negative_message = negative_mass * negative_values - negative_weights @ negative_values
        selected_mask = positive_mask | negative_mask
        negative_message = torch.zeros_like(negative_message)
        confidence = ((1.0 - hesitation) * selected_mask.to(hesitation.dtype)).sum(
            dim=-1
        ) / selected_mask.sum(dim=-1).clamp_min(1)
        confidence = torch.ones_like(confidence)
        raw_delta = self.update(torch.cat([nodes, positive_message, negative_message], dim=-1))
        delta = self.update_norm(raw_delta)
        scale = self.config.update_scale_max * torch.sigmoid(self.step_scales[step])
        updated = nodes.clone()
        query_before_update = nodes[support_count:]
        query_count = nodes.size(0) - support_count
        if query_update_weight is None:
            query_update_weight = nodes.new_ones(query_count, 1)
        else:
            query_update_weight = query_update_weight.reshape(-1, 1).to(
                dtype=nodes.dtype, device=nodes.device
            )
            if query_update_weight.size(0) != query_count:
                raise ValueError("query_update_weight must have one value per query")
        node_update_weight = nodes.new_ones(nodes.size(0), 1)
        node_update_weight[support_count:] = query_update_weight
        updated = F.normalize(
            nodes + scale * confidence[:, None] * node_update_weight * delta, dim=-1
        )
        diagnostics = {}
        return (
            updated,
            {
                "mechanism_diagnostics": diagnostics,
                "confidence": confidence,
                "positive_mask": positive_mask,
                **{"negative_mask": negative_mask},
                "update_scale": scale,
                "query_update_norm": (updated[support_count:] - query_before_update)
                .norm(dim=-1)
                .mean(),
            },
        )


class EGRCHead(nn.Module):
    """Main episodic EGRC head with the established parameter layout."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or EGRCConfig()
        self.adapter = nn.Identity()
        self.task_encoder = TaskContextEncoder(
            self.config.task_hidden_dim, self.config.relation_hidden_dim
        )
        self.edge_builder = TaskConditionedRelation(self.config)
        self.message_block = GraphPropagation(self.config)
        self.gate_network = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Dropout(self.config.dropout), nn.Linear(16, 1)
        )
        self.classifier_dropout = nn.Dropout(self.config.classifier_dropout)
        self.residual_alpha_raw = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, -3.0)

    def _prototypes(self, support, way, shot):
        expected = way * shot
        if support.size(0) != expected:
            raise ValueError(f"Expected {expected} support CLS tokens, got {support.size(0)}")
        return F.normalize(support.reshape(shot, way, support.size(-1)).mean(dim=0), dim=-1)

    def _prototype_logits(self, query, prototypes):
        query = F.normalize(self.classifier_dropout(query), dim=-1)
        return query @ prototypes.transpose(0, 1) / self.config.proto_temperature

    def _prepare_features(self, features):
        return F.normalize(features, dim=-1)

    def forward(self, support_cls, query_cls, way, shot, external_base_logits=None):
        if support_cls.ndim != 2 or query_cls.ndim != 2:
            raise ValueError("support_cls and query_cls must be [samples, dim]")
        if support_cls.size(-1) != self.config.dim:
            raise ValueError(f"Expected CLS dim {self.config.dim}, got {support_cls.size(-1)}")
        support = self._prepare_features(support_cls)
        query = self._prepare_features(query_cls)
        support_count = support.size(0)
        prototypes = self._prototypes(support, way, shot)
        internal_base_logits = self._prototype_logits(query, prototypes)
        base_logits = internal_base_logits if external_base_logits is None else external_base_logits
        if base_logits.shape != internal_base_logits.shape:
            raise ValueError("external_base_logits must match the episodic logit shape")
        base_probabilities = F.softmax(base_logits, dim=-1)
        base_uncertainty = _normalized_entropy(base_probabilities)
        support_labels = torch.arange(way, device=support.device).repeat(shot)
        support_beliefs = F.one_hot(support_labels, num_classes=way).to(dtype=support.dtype)
        query_beliefs = base_probabilities.detach()
        nodes = torch.cat([support, query], dim=0)
        task_context, task_statistics = self.task_encoder(support, prototypes, way, shot)
        prototype_gram = None
        edge_states = []
        message_states = []
        for step in range(self.config.num_steps):
            beliefs = torch.cat([support_beliefs, query_beliefs], dim=0)
            edge_state = self.edge_builder(
                nodes, beliefs, task_context, support_count, prototype_gram=prototype_gram
            )
            nodes, message_state = self.message_block(
                nodes, edge_state, support_count, step, query_update_weight=None
            )
            prototypes = self._prototypes(nodes[:support_count], way, shot)
            query_beliefs = F.softmax(
                self._prototype_logits(nodes[support_count:], prototypes), dim=-1
            )
            edge_states.append(edge_state)
            message_states.append(message_state)
        candidate_logits = self._prototype_logits(nodes[support_count:], prototypes)
        final_confidence = message_states[-1]["confidence"][support_count:]
        base_entropy = base_uncertainty
        if way > 1:
            top_two = base_probabilities.topk(k=2, dim=-1).values
            base_margin = top_two[:, 0] - top_two[:, 1]
        else:
            base_margin = torch.ones_like(base_entropy)
        gate_features = torch.stack([base_entropy, base_margin, final_confidence], dim=-1)
        residual_alpha = torch.zeros_like(self.residual_alpha_raw)
        correction_logits = candidate_logits - base_logits
        graph_logits = candidate_logits
        fusion_gate = (
            0.0 * final_confidence * torch.sigmoid(self.gate_network(gate_features).squeeze(-1))
        )
        logits = (1.0 - fusion_gate[:, None]) * base_logits + fusion_gate[:, None] * graph_logits
        edge_alpha = torch.stack([state["alpha"] for state in edge_states], dim=0)
        aux = {
            "probability_edges": not True or False,
            "base_logits": base_logits,
            "graph_logits": graph_logits,
            "candidate_logits": candidate_logits,
            "correction_logits": correction_logits,
            "base_uncertainty": base_uncertainty,
            "residual_alpha": residual_alpha,
            "fusion_gate": fusion_gate,
            "edge_alpha": edge_alpha,
            "edge_pair_types": edge_states[-1]["pair_types"],
            "edge_total_evidence": edge_states[-1]["total_evidence"],
            "task_statistics": task_statistics,
            "support_anchor_drift": (nodes[:support_count] - support).abs().max(),
            "query_update_norm": message_states[-1]["query_update_norm"],
            "graph_confidence": final_confidence,
        }
        aux.update(
            edge_membership=edge_states[-1]["membership"],
            edge_nonmembership=edge_states[-1]["nonmembership"],
            edge_hesitation=edge_states[-1]["hesitation"],
        )
        return (logits, aux)


class EGRCModel(nn.Module):
    """ViT-S backbone and the main EGRC episodic head."""

    def __init__(self, backbone, config=None):
        super().__init__()
        self.backbone = backbone
        self.head = EGRCHead(config)
        self.proto_temperature = float(0.1)
        if self.proto_temperature <= 0.0:
            raise ValueError("proto_temperature must be positive")
        self.backbone_gradient_checkpointing = False

    def set_backbone_gradient_checkpointing(self, enabled=True):
        self.backbone_gradient_checkpointing = bool(enabled)

    def _forward_backbone_checkpointed(self, images, chunk_size):
        """vendored ViT forward with block-level activation recomputation."""
        encoder = getattr(self.backbone, "encoder", None)
        required = ("prepare_tokens", "blocks", "norm")
        if encoder is None or not all((hasattr(encoder, name) for name in required)):
            raise TypeError("Backbone gradient checkpointing requires the vendored ViT encoder.")
        outputs = []
        for start in range(0, images.size(0), chunk_size):
            chunk = images[start : start + chunk_size]
            tokens = encoder.prepare_tokens(chunk)
            for block in encoder.blocks:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            tokens = encoder.norm(tokens)
            if getattr(encoder, "fc_norm", None) is not None:
                pooled = encoder.fc_norm(tokens[:, 1:, :].mean(dim=1))
                tokens = tokens.clone()
                tokens[:, 0, :] = pooled
            outputs.append(tokens)
        return torch.cat(outputs, dim=0)

    def encode_cls(self, images, chunk_size=32):
        backbone_trainable = any(
            (parameter.requires_grad for parameter in self.backbone.parameters())
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and backbone_trainable):
            use_checkpointing = (
                self.backbone_gradient_checkpointing
                and backbone_trainable
                and torch.is_grad_enabled()
            )
            if use_checkpointing:
                tokens = self._forward_backbone_checkpointed(images, chunk_size)
            elif hasattr(self.backbone, "forward_chunked"):
                tokens = self.backbone.forward_chunked(images, chunk_size=chunk_size)
            else:
                outputs = []
                for start in range(0, images.size(0), chunk_size):
                    outputs.append(self.backbone(images[start : start + chunk_size]))
                tokens = torch.cat(outputs, dim=0)
        if tokens.ndim == 2:
            cls = tokens
        elif tokens.ndim == 3:
            cls = tokens[:, 0, :]
        else:
            raise ValueError(f"Backbone must return [N,D] or [N,T,D], got {tokens.shape}")
        return F.normalize(cls, dim=-1)

    def forward_episode(self, support, query, way, shot, chunk_size=32):
        if self.head is None:
            raise RuntimeError("EGRC head is unavailable")
        support_cls = self.encode_cls(support, chunk_size=chunk_size)
        query_cls = self.encode_cls(query, chunk_size=chunk_size)
        _base_logits = self.prototype_logits_from_cls(
            support_cls,
            query_cls,
            way=way,
            shot=shot,
            temperature=self.head.config.proto_temperature,
        )
        graph_support_cls = support_cls
        graph_query_cls = query_cls
        return self.head(
            graph_support_cls, graph_query_cls, way=way, shot=shot, external_base_logits=None
        )

    @staticmethod
    def prototype_logits_from_cls(support_cls, query_cls, way, shot, temperature=0.1):
        """Parameter-free cosine ProtoNet logits from normalized CLS features."""
        expected = way * shot
        if support_cls.ndim != 2 or query_cls.ndim != 2:
            raise ValueError("support_cls and query_cls must be [samples, dim]")
        if support_cls.size(0) != expected:
            raise ValueError(f"Expected {expected} support CLS tokens, got {support_cls.size(0)}")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        support_cls = F.normalize(support_cls, dim=-1)
        query_cls = F.normalize(query_cls, dim=-1)
        prototypes = F.normalize(
            support_cls.reshape(shot, way, support_cls.size(-1)).mean(dim=0), dim=-1
        )
        return query_cls @ prototypes.transpose(0, 1) / temperature

    def forward_episode_from_cls(self, support_cls, query_cls, way, shot):
        if self.head is None:
            raise RuntimeError("EGRC head is unavailable")
        support_cls = F.normalize(support_cls, dim=-1)
        query_cls = F.normalize(query_cls, dim=-1)
        _base_logits = self.prototype_logits_from_cls(
            support_cls,
            query_cls,
            way=way,
            shot=shot,
            temperature=self.head.config.proto_temperature,
        )
        graph_support_cls = support_cls
        graph_query_cls = query_cls
        return self.head(
            graph_support_cls, graph_query_cls, way=way, shot=shot, external_base_logits=None
        )


def _dirichlet_kl_to_uniform(alpha):
    classes = alpha.size(-1)
    beta = torch.ones_like(alpha)
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    beta_sum = beta.sum(dim=-1, keepdim=True)
    log_normalizer_ratio = (
        torch.lgamma(alpha_sum)
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        - torch.lgamma(beta_sum)
        + torch.lgamma(beta).sum(dim=-1, keepdim=True)
    )
    expectation = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(alpha_sum))).sum(
        dim=-1, keepdim=True
    )
    result = log_normalizer_ratio + expectation
    if classes == 1:
        return torch.zeros_like(result.squeeze(-1))
    return result.squeeze(-1)


def evidential_edge_loss(edge_alpha, node_labels, pair_types=None, kl_weight=0.0):
    """Balanced evidential same/different loss over one or more graph steps.

    Query labels are supplied only to this loss.  They are not accepted by any
    EGRC forward method, which keeps the inference path label-leakage free.
    """
    if edge_alpha.ndim == 3:
        edge_alpha = edge_alpha.unsqueeze(0)
    if edge_alpha.ndim != 4 or edge_alpha.size(-1) != 2:
        raise ValueError("edge_alpha must be [steps, N, N, 2]")
    edge_alpha = edge_alpha.float()
    node_labels = node_labels.to(edge_alpha.device)
    same = node_labels[:, None].eq(node_labels[None, :])
    upper = torch.triu(torch.ones_like(same, dtype=torch.bool), diagonal=1)
    if pair_types is None:
        pair_type_index = torch.zeros_like(node_labels[:, None] + node_labels[None, :])
    else:
        pair_type_index = pair_types.argmax(dim=-1)
    selected_parts = []
    for pair_type in range(3 if pair_types is not None else 1):
        type_mask = upper & pair_type_index.eq(pair_type)
        positive = (type_mask & same).nonzero(as_tuple=False)
        negative = (type_mask & ~same).nonzero(as_tuple=False)
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        count = min(positive.size(0), negative.size(0))
        if positive.size(0) > count:
            positive = positive[torch.randperm(positive.size(0), device=positive.device)[:count]]
        if negative.size(0) > count:
            negative = negative[torch.randperm(negative.size(0), device=negative.device)[:count]]
        selected_parts.extend([positive, negative])
    if not selected_parts:
        raise ValueError("No balanced positive/negative edge pairs are available")
    selected = torch.cat(selected_parts, dim=0)
    selected_same = same[selected[:, 0], selected[:, 1]].to(edge_alpha.dtype)
    target = torch.stack([selected_same, 1.0 - selected_same], dim=-1)
    losses = []
    for step_alpha in edge_alpha:
        alpha = step_alpha[selected[:, 0], selected[:, 1]]
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        expected_nll = (target * (torch.digamma(alpha_sum) - torch.digamma(alpha))).sum(dim=-1)
        adjusted_alpha = target + (1.0 - target) * alpha
        kl = _dirichlet_kl_to_uniform(adjusted_alpha)
        losses.append((expected_nll + float(kl_weight) * kl).mean())
    return torch.stack(losses).mean()


def egrc_training_loss(
    logits, query_labels, aux, node_labels, edge_loss_weight=None, edge_kl_weight=0.0
):
    if edge_loss_weight is None:
        support_count = node_labels.numel() - query_labels.numel()
        shot, remainder = divmod(support_count, logits.size(-1))
        if remainder or shot not in (1, 5):
            raise ValueError("Set edge_loss_weight explicitly outside the 1/5-shot settings")
        edge_loss_weight = 0.01 if shot == 1 else 0.05
    final_loss = F.cross_entropy(logits, query_labels)
    graph_metric_logits = aux.get("candidate_logits", aux["graph_logits"])
    graph_loss = F.cross_entropy(graph_metric_logits.detach(), query_labels)
    base_loss = F.cross_entropy(aux["base_logits"].detach(), query_labels)
    edge_loss = evidential_edge_loss(
        aux["edge_alpha"],
        node_labels,
        pair_types=aux.get("edge_pair_types"),
        kl_weight=edge_kl_weight,
    )
    total = (
        final_loss
        + float(0.0) * graph_loss
        + float(0.0) * base_loss
        + float(edge_loss_weight) * edge_loss
    )
    metrics = {
        "loss_final": final_loss.detach(),
        "loss_graph": graph_loss.detach(),
        "loss_base": base_loss.detach(),
        "loss_edge": edge_loss.detach(),
    }
    return (total, metrics)
