import argparse
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
)
from tqdm import tqdm

from models import COS_Model
from pytorch_loss import FocalLossV1
from sessions import (
    build_roots_from_args,
    build_session,
    normalize_dataset_name,
    supported_session_pairs,
)
from uni3D.utils.tokenizer import SimpleTokenizer
from utils import EXIOStream


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts):
    return os.path.join(ROOT_DIR, *parts)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Point-UQ training and evaluation")
    parser.add_argument("--exp_name", type=str, default="point_uq")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch0", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cluster_num", type=int, default=5)
    parser.add_argument("--memory_shot", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=2e-5)
    parser.add_argument("--lri", type=float, default=1e-3)
    parser.add_argument("--lri_base", type=float, default=5e-4)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="frontloaded",
        choices=["none", "cosine", "frontloaded"],
        help="task-0 learning-rate schedule",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=0.0,
        help="optional warmup for task-0 scheduler; 0 disables warmup",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="minimum learning rate reached near the end of task-0 training",
    )
    parser.add_argument(
        "--early_decay_portion",
        type=float,
        default=0.35,
        help="fraction of post-warmup training spent on fast early LR decay",
    )
    parser.add_argument(
        "--early_decay_ratio",
        type=float,
        default=0.30,
        help="LR ratio reached after the fast early-decay stage",
    )
    parser.add_argument("--loss_fn", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--save_model_name", type=str, default=None)

    parser.add_argument("--base_dataset", type=str, default="shapenet")
    parser.add_argument("--incremental_dataset", type=str, default="co3d")
    parser.add_argument("--validate_dataset_prompt", type=str, default="modelnet40_640")

    parser.add_argument("--clip_model_name", type=str, default="EVA02-E-14-plus")
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default=project_path("uni3D", "trainedModel", "clip_model", "open_clip_pytorch_model.bin"),
    )
    parser.add_argument(
        "--uni3d_ckpt_path",
        type=str,
        default=project_path("uni3D", "trainedModel", "checkpoints", "model_b.pt"),
    )
    parser.add_argument("--shapenet_root", type=str, default=project_path("data", "shapenet"))
    parser.add_argument("--co3d_root", type=str, default=project_path("data", "co3d"))
    parser.add_argument("--modelnet_root", type=str, default=project_path("data", "modelnet"))
    parser.add_argument("--scanobjnn_root", type=str, default=project_path("data", "scanobjnn"))
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rgb(points):
    return torch.full_like(points, 0.4)


def unpack_model_output(output, device):
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, torch.zeros((), device=device)


def build_dataloader(dataset, args, shuffle):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=shuffle,
        persistent_workers=args.workers > 0,
    )


def build_scheduler(optimizer, args, steps_per_epoch):
    if args.lr_schedule == "none" or args.epoch0 <= 0 or steps_per_epoch <= 0:
        return None

    total_steps = max(1, args.epoch0 * steps_per_epoch)
    warmup_steps = int(max(0.0, args.warmup_epochs) * steps_per_epoch)
    warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0
    min_lr_ratio = max(0.0, min(1.0, args.min_lr / max(args.lr0, 1e-12)))

    if args.lr_schedule == "cosine":

        def lr_lambda(step_idx):
            if warmup_steps > 0 and step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            remaining = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step_idx - warmup_steps) / remaining))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    fast_portion = max(0.0, min(1.0, args.early_decay_portion))
    fast_target = max(min_lr_ratio, min(1.0, args.early_decay_ratio))

    def lr_lambda(step_idx):
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        post_warmup_steps = max(1, total_steps - warmup_steps)
        stage_step = min(post_warmup_steps, max(0, step_idx - warmup_steps))
        fast_steps = int(post_warmup_steps * fast_portion)

        if fast_steps > 0 and stage_step < fast_steps:
            fast_progress = stage_step / max(1, fast_steps)
            return 1.0 - (1.0 - fast_target) * fast_progress

        slow_steps = max(1, post_warmup_steps - fast_steps)
        slow_progress = (stage_step - fast_steps) / slow_steps
        slow_progress = min(1.0, max(0.0, slow_progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * slow_progress))
        return min_lr_ratio + (fast_target - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_prompt_features(id2name, clip_model, tokenizer, template_key, device):
    templates_path = project_path("uni3D", "data", "templates.json")
    with open(templates_path, "r", encoding="utf-8") as fin:
        templates = json.load(fin)[template_key]

    text_features = []
    clip_model.eval()
    with torch.no_grad():
        for label_name in id2name:
            texts = [template.format(label_name) for template in templates]
            tokenized = tokenizer(texts)
            if len(tokenized.shape) < 2:
                tokenized = tokenized[None, ...]
            tokenized = tokenized.to(device)
            class_embeddings = clip_model.encode_text(tokenized)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_embeddings = F.normalize(class_embeddings.mean(dim=0), dim=-1)
            text_features.append(class_embeddings)
    return torch.stack(text_features, dim=0)


def ensure_path_exists(path, label, expect_dir=False):
    if expect_dir:
        valid = os.path.isdir(path)
    else:
        valid = os.path.isfile(path)
    if not valid:
        kind = "directory" if expect_dir else "file"
        raise FileNotFoundError(f"Missing {label} {kind}: {path}")


def validate_runtime_inputs(args, roots, base_dataset, incremental_dataset):
    required_files = (
        ("CLIP pretrained checkpoint", args.clip_pretrained),
        ("Uni3D checkpoint", args.uni3d_ckpt_path),
        ("prompt template file", project_path("uni3D", "data", "templates.json")),
    )
    for label, path in required_files:
        ensure_path_exists(path, label, expect_dir=False)

    seen = set()
    for dataset_name in (base_dataset, incremental_dataset):
        if dataset_name in ("null",) or dataset_name in seen:
            continue
        seen.add(dataset_name)
        dataset_root = roots.path_for(dataset_name)
        if dataset_root is not None:
            ensure_path_exists(dataset_root, f"{dataset_name} dataset root", expect_dir=True)


def train_loop(dataloader, model, prompts_feats, optimizer, criterion, stat, device, io, scheduler=None):
    num_cls = prompts_feats.size(0)
    metrics = MetricCollection([MulticlassAccuracy(num_classes=num_cls, average="micro")]).to(device)
    train_loss = MeanMetric().to(device)

    model.train()
    for points, labels in tqdm(dataloader):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        rgb = make_rgb(points)

        logits, aux_loss = unpack_model_output(
            model(points, rgb, prompts_feats, stat["task_id"], labels),
            device,
        )
        loss = aux_loss if stat["task_id"] == 0 else criterion(logits, labels) + aux_loss
        preds = torch.max(logits, dim=1).indices

        metrics.update(preds, labels)
        train_loss.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    acc = metrics.compute()
    current_lr = optimizer.param_groups[0]["lr"]
    io.cprint(
        f"[train] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\tLoss:{train_loss.compute()}\t"
        f"LR:{current_lr:.6g}\t"
        f"Accuracy:{100 * acc['MulticlassAccuracy']:.1f}",
        name="epochs.log",
    )

def test_loop(dataloader, model, prompts_feats, stat, device, io, exp_dir):
    num_cls = prompts_feats.size(0)
    metrics = MetricCollection([MulticlassAccuracy(num_classes=num_cls, average="micro")]).to(device)

    model.eval()
    with torch.no_grad():
        for points, labels in tqdm(dataloader):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            rgb = make_rgb(points)
            logits, _ = unpack_model_output(model(points, rgb, prompts_feats, stat["task_id"], labels), device)
            preds = torch.max(logits, dim=1).indices

            metrics.update(preds, labels)

    acc = metrics.compute()
    io.cprint(
        f"[test] Task:{stat['task_id']}\tEpoch:{stat['epoch']}\t"
        f"Accuracy:{100 * acc['MulticlassAccuracy']:.1f}",
        name="epochs.log",
    )
    return num_cls


def evaluate_loader(dataloader, model, prompts_feats, task_id, device):
    preds_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        for points, labels in tqdm(dataloader):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            rgb = make_rgb(points)
            logits, _ = unpack_model_output(model(points, rgb, prompts_feats, task_id, labels), device)
            preds = torch.max(logits, dim=1).indices
            preds_all.extend(preds.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())
    return preds_all, labels_all
def test_loop_new(
    dataloader_base,
    dataloader_new,
    model,
    prompts_feats,
    stat,
    device,
    io,
):
    base_preds, base_labels = evaluate_loader(dataloader_base, model, prompts_feats, stat["task_id"], device)
    new_preds, new_labels = evaluate_loader(dataloader_new, model, prompts_feats, stat["task_id"], device)

    all_preds = base_preds + new_preds
    all_labels = base_labels + new_labels

    overall_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0

    io.cprint(
        f"[incremental-test] Task:{stat['task_id']}\tAccuracy:{100 * overall_acc:.1f}",
        name="epochs.log",
    )
    return prompts_feats.size(0)


def extract_and_average_features(
    model,
    train_loader,
    prompts_feats,
    device,
    n_clusters=5,
    progress_desc="extract",
):
    model.eval()
    all_features = defaultdict(list)
    with torch.no_grad():
        for points, labels in tqdm(train_loader, desc=progress_desc):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            rgb = make_rgb(points)
            features, _ = model.extract_point_features(points, rgb)
            for feature, label in zip(features.detach().cpu().numpy(), labels.detach().cpu().numpy()):
                all_features[int(label)].append(feature)

    weighted_prototypes = {}
    for label, features in all_features.items():
        if label < 0 or label >= len(prompts_feats):
            continue

        features_array = np.stack(features)
        if features_array.shape[0] < n_clusters:
            weighted_prototypes[label] = np.mean(features_array, axis=0)
            continue

        text_feat = prompts_feats[label].detach().cpu().numpy().reshape(1, -1)
        cluster_count = min(max(1, n_clusters), features_array.shape[0])

        if cluster_count == 1:
            weighted_prototypes[label] = np.mean(features_array, axis=0)
            continue

        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        kmeans.fit(features_array)

        center_samples = []
        for idx in range(cluster_count):
            cluster_features = features_array[kmeans.labels_ == idx]
            distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[idx], axis=1)
            center_samples.append(cluster_features[np.argmin(distances)])

        center_samples_array = np.stack(center_samples)
        similarities = cosine_similarity(center_samples_array, text_feat).reshape(-1)
        shifted = similarities - np.max(similarities)
        exp_scores = np.exp(shifted)
        denom = np.sum(exp_scores)
        if denom <= 0:
            weights = np.full((cluster_count,), 1.0 / cluster_count, dtype=np.float32)
        else:
            weights = exp_scores / denom
        weighted_prototypes[label] = np.sum(center_samples_array * weights[:, None], axis=0)

    return weighted_prototypes


def print_trainable_parameters(model):
    trainable_total = 0
    non_trainable_total = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_total += param.numel()
        else:
            non_trainable_total += param.numel()
    print(f"trainable params: {trainable_total}")
    print(f"frozen params: {non_trainable_total}")


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_dataset = normalize_dataset_name(args.base_dataset)
    incremental_dataset = normalize_dataset_name(args.incremental_dataset)
    roots = build_roots_from_args(args)
    validate_runtime_inputs(args, roots, base_dataset, incremental_dataset)

    exp_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_dir = project_path("exp_results", f"{exp_time}:{args.exp_name}")
    io = EXIOStream(exp_dir)

    io.cprint(f"Base dataset: {base_dataset}")
    io.cprint(f"Incremental dataset: {incremental_dataset}")
    io.cprint(f"Supported session pairs: {', '.join(supported_session_pairs())}")
    io.cprint(f"Dataset roots: {roots.as_dict()}")
    io.cprint(f"CLIP checkpoint: {args.clip_pretrained}")
    io.cprint(f"Uni3D checkpoint: {args.uni3d_ckpt_path}")
    io.cprint(
        f"LR schedule: {args.lr_schedule} | warmup_epochs={args.warmup_epochs} | "
        f"min_lr={args.min_lr} | early_decay_portion={args.early_decay_portion} | "
        f"early_decay_ratio={args.early_decay_ratio}"
    )

    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
    )
    clip_model = clip_model.to(device)
    tokenizer = SimpleTokenizer()

    session_maker = build_session(base_dataset, incremental_dataset, roots=roots)
    id2name = session_maker.get_id2name()
    io.cprint(session_maker.info())

    prompts_feats = build_prompt_features(
        id2name=id2name,
        clip_model=clip_model,
        tokenizer=tokenizer,
        template_key=args.validate_dataset_prompt,
        device=device,
    ).detach()
    io.cprint(f"Prompt features ready: {tuple(prompts_feats.shape)}")

    dataset_train_0, _, test_dataset_base_0, _ = session_maker.make_session(
        session_id=0,
        update_memory=args.memory_shot,
    )
    num_cat_0 = test_dataset_base_0.get_cat_num()
    train_loader_0 = build_dataloader(dataset_train_0, args, shuffle=True)
    test_loader_0 = build_dataloader(test_dataset_base_0, args, shuffle=False)
    io.cprint(
        f"Base session ready: train_samples={len(dataset_train_0)} "
        f"test_samples={len(test_dataset_base_0)} base_classes={num_cat_0}"
    )

    io.cprint(f"Loading Point-UQ model from {args.uni3d_ckpt_path}")
    model = COS_Model(
        args,
        device=device,
    ).to(device)
    io.cprint("Point-UQ model loaded")
    criterion = FocalLossV1() if args.loss_fn == "focal" else F.cross_entropy
    optimizer_0 = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr0,
        weight_decay=1e-4,
    )
    scheduler_0 = build_scheduler(optimizer_0, args, len(train_loader_0))

    os.makedirs(project_path("base_train"), exist_ok=True)
    base_model_name = args.save_model_name or f"{base_dataset}_to_{incremental_dataset}_base"
    base_model_path = project_path("base_train", f"{base_model_name}.pth")

    io.cprint("Extracting base cache features before task-0 training")
    avg_features = extract_and_average_features(
        model,
        train_loader_0,
        prompts_feats[:num_cat_0],
        device,
        n_clusters=args.cluster_num,
        progress_desc="base-cache",
    )
    model.update_classifier_cache(avg_features)
    io.cprint(f"Base cache ready: {len(avg_features)} classes")

    runtime_stat = {"task_id": 0, "epoch": 0}
    for epoch in range(args.epoch0):
        io.cprint(f"---------------Epoch {epoch + 1}-------------------")
        runtime_stat["epoch"] = epoch + 1
        train_loop(
            train_loader_0,
            model,
            prompts_feats[:num_cat_0],
            optimizer_0,
            criterion,
            runtime_stat,
            device,
            io,
            scheduler=scheduler_0,
        )
    io.cprint("Refreshing base cache features after task-0 training")
    avg_features = extract_and_average_features(
        model,
        train_loader_0,
        prompts_feats[:num_cat_0],
        device,
        n_clusters=args.cluster_num,
        progress_desc="base-cache-refresh",
    )
    model.update_classifier_cache(avg_features)
    io.cprint(f"Refreshed base cache ready: {len(avg_features)} classes")
    torch.save(model.state_dict(), base_model_path)

    print_trainable_parameters(model)
    test_loop(
        test_loader_0,
        model,
        prompts_feats[:num_cat_0],
        {"task_id": 0, "epoch": args.epoch0},
        device,
        io,
        exp_dir,
    )

    for task_id in range(1, session_maker.tot_session()):
        io.cprint("=" * 40, f"Task {task_id}", "=" * 40)
        dataset_train_i, _, test_dataset_base_i, test_dataset_new_i = session_maker.make_session(
            session_id=task_id,
            update_memory=args.memory_shot,
        )
        num_cat_i = test_dataset_base_i.get_cat_num() + test_dataset_new_i.get_cat_num()
        train_loader_i = build_dataloader(dataset_train_i, args, shuffle=True)
        test_loader_base_i = build_dataloader(test_dataset_base_i, args, shuffle=False)
        test_loader_new_i = build_dataloader(test_dataset_new_i, args, shuffle=False)

        avg_features = extract_and_average_features(
            model,
            train_loader_i,
            prompts_feats[:num_cat_i],
            device,
            n_clusters=args.cluster_num,
            progress_desc=f"task{task_id}-cache",
        )
        model.update_classifier_cache(avg_features)
        io.cprint(f"Task {task_id} cache ready: {len(avg_features)} classes")

        test_loop_new(
            test_loader_base_i,
            test_loader_new_i,
            model,
            prompts_feats[:num_cat_i],
            {"task_id": task_id, "epoch": args.epoch0},
            device,
            io,
        )


if __name__ == "__main__":
    parsed_args = build_argument_parser().parse_args()
    main(parsed_args)
