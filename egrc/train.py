"""Main EGRC trainer: full ViT fine-tuning, validation selection, final test."""

from collections.abc import Mapping
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from egrc.config import ROOT, parse_args, resolve_paths
from egrc.model import EGRCConfig, EGRCModel, egrc_training_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested.startswith("cuda") and (not torch.cuda.is_available()):
        print("WARNING: CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def amp_context(args, device):
    if args.amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _load_local_checkpoint(path, map_location="cpu"):
    """Load one of this project's trusted local checkpoints across torch versions.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    The original pretrained ViT checkpoints contain NumPy scalar metadata, so the
    restricted loader rejects them even though the tensor payload is valid.
    The explicit fallback must therefore only be used with a checkpoint whose
    origin is trusted; the default paths point at the documented initialization files.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as restricted_error:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            raise restricted_error


def load_vit_pretrained_compatible(model, init_weights):
    """Map the original pretrained ViT tensors into ``model.backbone`` only."""
    checkpoint = _load_local_checkpoint(init_weights, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Invalid pretrained checkpoint: {init_weights}")
    state = checkpoint.get("teacher", checkpoint)
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Checkpoint teacher payload is not a state dict: {init_weights}")
    target_state = model.state_dict()
    mapped = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        candidates = [clean_key, f"backbone.encoder.{clean_key}"]
        if clean_key.startswith("backbone."):
            candidates.append(f"backbone.encoder.{clean_key[len('backbone.') :]}")
        if clean_key.startswith("encoder."):
            candidates.append(f"backbone.{clean_key}")
        for mapped_key in candidates:
            target = target_state.get(mapped_key)
            if (
                mapped_key.startswith("backbone.")
                and target is not None
                and (target.shape == value.shape)
            ):
                mapped[mapped_key] = value
                break
    target_state.update(mapped)
    model.load_state_dict(target_state, strict=True)
    return len(mapped)


def episode_labels(way, shot, query, device):
    support_labels = torch.arange(device=device, end=way).repeat(shot)
    query_labels = torch.arange(device=device, end=way).repeat(query)
    return (query_labels, torch.cat([support_labels, query_labels], dim=0))


def prediction_logits(fused_logits, aux, graph_only=False):
    """Select the reported/trained prediction without pruning diagnostics."""
    return aux["graph_logits"] if graph_only else fused_logits


def forward_episode(model, batch, args, device, train_mode):
    images, _ = batch
    images = images.to(device, non_blocking=True)
    way = args.way if train_mode else args.test_way
    support_count = way * args.shot
    support = images[:support_count]
    query = images[support_count:]
    logits, aux = model.forward_episode(
        support, query, way=way, shot=args.shot, chunk_size=args.backbone_chunk_size
    )
    logits = prediction_logits(logits, aux, graph_only=True)
    query_labels, node_labels = episode_labels(way, args.shot, args.query, device)
    if logits.size(0) != query_labels.size(0):
        raise RuntimeError(
            f"Query count mismatch: logits={logits.size(0)}, labels={query_labels.size(0)}"
        )
    return (logits, aux, query_labels, node_labels)


def accuracy(logits, labels):
    return logits.argmax(dim=-1).eq(labels).float().mean().item()


def confidence_interval(values):
    values = np.asarray(values, dtype=np.float64)
    return (float(values.mean()), float(1.96 * values.std() / np.sqrt(len(values))))


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu().item())
    return float(value)


def aux_summary(aux):
    if "edge_membership" not in aux:
        return {
            "gate": scalar(aux["fusion_gate"]),
            "graph_confidence": scalar(aux["graph_confidence"]),
            "update_norm": scalar(aux["query_update_norm"]),
            "support_drift": scalar(aux["support_anchor_drift"]),
        }
    summary = {
        "gate": scalar(aux["fusion_gate"]),
        "membership": scalar(aux["edge_membership"]),
        "nonmembership": scalar(aux["edge_nonmembership"]),
        "hesitation": scalar(aux["edge_hesitation"]),
        "evidence": scalar(aux["edge_total_evidence"]),
        "graph_confidence": scalar(aux["graph_confidence"]),
        "update_norm": scalar(aux["query_update_norm"]),
        "support_drift": scalar(aux["support_anchor_drift"]),
    }
    if "base_uncertainty" in aux:
        summary["base_uncertainty"] = scalar(aux["base_uncertainty"])
    if "residual_alpha" in aux:
        summary["residual_alpha"] = scalar(aux["residual_alpha"])
    return summary


def evaluate(model, loader, args, device):
    model.eval()
    model.backbone.eval()
    accuracies = []
    base_accuracies = []
    graph_accuracies = []
    candidate_accuracies = []
    summaries = []
    with torch.no_grad():
        for episode_index, batch in enumerate(loader, 1):
            with amp_context(args, device):
                logits, aux, labels, _ = forward_episode(
                    model, batch, args, device, train_mode=False
                )
            accuracies.append(accuracy(logits, labels))
            base_accuracies.append(accuracy(aux["base_logits"], labels))
            graph_accuracies.append(accuracy(aux["graph_logits"], labels))
            if "candidate_logits" in aux:
                candidate_accuracies.append(accuracy(aux["candidate_logits"], labels))
            summaries.append(aux_summary(aux))
            if args.log_every and (
                episode_index == 1
                or episode_index % args.log_every == 0
                or episode_index == len(loader)
            ):
                print(f"  evaluation {episode_index}/{len(loader)}", flush=True)
    final_mean, final_interval = confidence_interval(accuracies)
    base_mean, base_interval = confidence_interval(base_accuracies)
    graph_mean, graph_interval = confidence_interval(graph_accuracies)
    candidate_metrics = {}
    if candidate_accuracies:
        candidate_mean, candidate_interval = confidence_interval(candidate_accuracies)
        candidate_metrics = {"candidate_acc": candidate_mean, "candidate_ci95": candidate_interval}
    diagnostics = {
        key: float(np.mean([summary[key] for summary in summaries])) for key in summaries[0]
    }
    return {
        "final_acc": final_mean,
        "final_ci95": final_interval,
        "base_acc": base_mean,
        "base_ci95": base_interval,
        "graph_acc": graph_mean,
        "graph_ci95": graph_interval,
        **candidate_metrics,
        **diagnostics,
    }


def primary_accuracy(metrics, args):
    return metrics["final_acc"]


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, args):
    torch.save(
        {
            "method": "EGRC",
            "format": "egrc-vit-s16-v1",
            "head": model.head.state_dict(),
            "backbone": model.backbone.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "args": vars(args),
        },
        path,
    )


def load_egrc_checkpoint(path, model, device):
    checkpoint = _load_local_checkpoint(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("EGRC checkpoint must contain a metadata dictionary")
    method = checkpoint.get("method")
    metadata = checkpoint.get("args", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Invalid checkpoint configuration metadata")
    legacy_variant = metadata.get("ablation_variant")
    if method == "EGRC":
        if checkpoint.get("format") != "egrc-vit-s16-v1" or legacy_variant not in (None, "tc_eif"):
            raise ValueError("Unsupported EGRC checkpoint identity or format")
    else:
        components = metadata.get("ablation_components", {})
        expected = {"tc": True, "eif": True, "dmp": False, "cm": False}
        if (
            method is not None
            or legacy_variant != "tc_eif"
            or not isinstance(components, Mapping)
            or any(components.get(key) != value for key, value in expected.items())
            or any(value for key, value in components.items() if key not in expected)
        ):
            raise ValueError(
                "Checkpoint requires EGRC identity or validated legacy tc_eif metadata"
            )
        for key, value in (
            ("finetune_vit", True),
            ("graph_only", True),
            ("use_adapter", False),
            ("update_support", True),
            ("residual_graph", False),
            ("uncertainty_weighted_updates", False),
        ):
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Legacy checkpoint has incompatible {key}")
    if not isinstance(checkpoint.get("head"), Mapping) or not isinstance(
        checkpoint.get("backbone"), Mapping
    ):
        raise RuntimeError(
            "Full EGRC checkpoints must contain both the graph head and fine-tuned backbone"
        )
    incompatible = model.head.load_state_dict(checkpoint["head"], strict=False)
    unexpected_missing = set(incompatible.missing_keys) - {"residual_alpha_raw"}
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Incompatible EGRC head checkpoint: missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    model.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    return checkpoint


def build_model_and_data(args, device):
    from egrc.data.loaders import build_dataset, build_eval_loader, build_train_loaders
    from egrc.backbones.backbone import BackBone

    backbone = BackBone("small")
    feature_dim = 384
    config = EGRCConfig(
        dim=feature_dim,
        num_steps=args.num_steps,
        positive_topk=args.positive_topk,
        evidence_max=args.evidence_max,
        update_scale_max=args.update_scale_max,
        proto_temperature=args.proto_temperature,
        dropout=args.dropout,
        classifier_dropout=args.classifier_dropout,
    )
    model = EGRCModel(backbone, config).to(device)
    model.set_backbone_gradient_checkpointing(args.vit_gradient_checkpointing)
    loaded = load_vit_pretrained_compatible(model, args.init_weights)
    expected = len(model.backbone.state_dict())
    coverage = loaded / max(expected, 1)
    print(
        f"Loaded {loaded}/{expected} pretrained ViT-S tensors ({coverage:.2%} coverage).",
        flush=True,
    )
    if coverage < 0.95:
        raise RuntimeError(f"Pretrained backbone coverage {coverage:.2%} is below 95%.")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True
    model.backbone.train()
    trainable_backbone = sum(
        (parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad)
    )
    print(
        f"Backbone=ViT-S; mode=full-finetune; feature_dim={feature_dim}; trainable backbone parameters={trainable_backbone:,}; gradient_checkpointing={model.backbone_gradient_checkpointing}.",
        flush=True,
    )
    print("Preparing datasets and episodic loaders...", flush=True)
    dataset_cls = build_dataset(args)
    if args.eval_only:
        return (model, None, None, build_eval_loader(args, dataset_cls, split="test"))
    train_loader, val_loader = build_train_loaders(args, dataset_cls)
    test_loader = build_eval_loader(args, dataset_cls, split="test")
    return (model, train_loader, val_loader, test_loader)


def learning_rate_lambda(epoch, warmup_epochs, max_epoch):
    current = epoch + 1
    if warmup_epochs > 0 and current <= warmup_epochs:
        return current / warmup_epochs
    remaining = max(max_epoch - warmup_epochs, 1)
    progress = min(max((current - warmup_epochs) / remaining, 0.0), 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def main(args=None):
    args = resolve_paths(args if args is not None else parse_args())
    set_seed(args.seed)
    device = choose_device(args.device)
    args_file = "resolved_args_eval.json" if args.eval_only else "resolved_args.json"
    with open(Path(args.log_path) / args_file, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    model, train_loader, val_loader, test_loader = build_model_and_data(args, device)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (ROOT / checkpoint_path).resolve()
        checkpoint = load_egrc_checkpoint(checkpoint_path, model, device)
        print(
            f"Loaded EGRC checkpoint {checkpoint_path} (epoch={checkpoint.get('epoch')}, best_val={checkpoint.get('best_val')}).",
            flush=True,
        )
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        result = evaluate(model, test_loader, args, device)
        final_record = {"test": result}
        with open(Path(args.log_path) / "evaluation_result.json", "w", encoding="utf-8") as handle:
            json.dump(final_record, handle, indent=2, ensure_ascii=False)
        print(json.dumps(final_record, indent=2), flush=True)
        return
    parameter_groups = [
        {
            "params": list(model.head.parameters()),
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "group_name": "head",
        }
    ]
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.backbone_lr,
                "weight_decay": args.backbone_weight_decay,
                "group_name": "backbone",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rate_lambda(epoch, args.warmup_epochs, args.max_epoch),
    )
    best_val = -1.0
    best_path = Path(args.checkpoint_path) / "max_acc.pth"
    last_path = Path(args.checkpoint_path) / "last.pth"
    metrics_path = Path(args.log_path) / "metrics.jsonl"
    for epoch in range(1, args.max_epoch + 1):
        print(f"Epoch {epoch}/{args.max_epoch}: training {len(train_loader)} episodes", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        model.train()
        model.backbone.train()
        train_losses = []
        train_accuracies = []
        train_summaries = []
        edge_kl_weight = min(epoch / max(args.edge_kl_warmup_epochs, 1), 1.0)
        for episode_index, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args, device):
                logits, aux, query_labels, node_labels = forward_episode(
                    model, batch, args, device, train_mode=True
                )
                loss, loss_metrics = egrc_training_loss(
                    logits,
                    query_labels,
                    aux,
                    node_labels,
                    edge_loss_weight=args.edge_loss_weight,
                    edge_kl_weight=edge_kl_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu().item()))
            train_accuracies.append(accuracy(logits.detach(), query_labels))
            summary = {} if aux is None else aux_summary(aux)
            summary.update({key: scalar(value) for key, value in loss_metrics.items()})
            train_summaries.append(summary)
            if args.log_every and (
                episode_index == 1
                or episode_index % args.log_every == 0
                or episode_index == len(train_loader)
            ):
                print(
                    f"  train {episode_index}/{len(train_loader)} loss={train_losses[-1]:.4f} acc={train_accuracies[-1]:.2%}",
                    flush=True,
                )
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_end = time.time()
        train_peak_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(f"Epoch {epoch}/{args.max_epoch}: validating", flush=True)
        val_result = evaluate(model, val_loader, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_end = time.time()
        val_peak_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        )
        train_result = {
            "loss": float(np.mean(train_losses)),
            "acc": float(np.mean(train_accuracies)),
            **{
                key: float(np.mean([summary[key] for summary in train_summaries]))
                for key in train_summaries[0]
            },
        }
        record = {
            "epoch": epoch,
            "lr": {
                group.get("group_name", str(index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "seconds": val_end - start,
            "train_seconds": train_end - start,
            "val_seconds": val_end - train_end,
            "train_peak_allocated_mb": train_peak_mb,
            "val_peak_allocated_mb": val_peak_mb,
            "train": train_result,
            "val": val_result,
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        val_accuracy = primary_accuracy(val_result, args)
        if val_accuracy > best_val:
            best_val = val_accuracy
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val, args)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val, args)
    load_egrc_checkpoint(best_path, model, device)
    print("Testing the checkpoint selected by validation accuracy...", flush=True)
    test_result = evaluate(model, test_loader, args, device)
    final_record = {"best_val": best_val, "test": test_result, "checkpoint": str(best_path)}
    with open(Path(args.log_path) / "test_result.json", "w", encoding="utf-8") as handle:
        json.dump(final_record, handle, indent=2, ensure_ascii=False)
    print(json.dumps(final_record, indent=2, ensure_ascii=False), flush=True)
