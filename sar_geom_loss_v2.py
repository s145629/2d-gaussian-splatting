"""Material-free SAR geometry losses for hybrid 2DGS training."""
from __future__ import annotations

import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class _RangeCapture:
    def __init__(self, original_splat, dctx):
        self.original_splat = original_splat
        self.dctx = dctx
        self.range_image = None
        self.valid = None

    def __call__(self, u, v, sigma, amplitude, height, width, win_r):
        raw = self.original_splat(u, v, sigma, amplitude, height, width, win_r)
        r_near = float(self.dctx["r_near"])
        r_far = float(self.dctx["r_far"])
        rg = r_near + v / float(height) * (r_far - r_near)
        weighted_range = self.original_splat(
            u, v, sigma, amplitude * rg, height, width, win_r
        )
        self.range_image = weighted_range / raw.clamp_min(1e-8)
        self.valid = raw > 1e-8
        return raw


@contextmanager
def _capture_probe_splat(probe, dctx):
    original = probe.local_window_splat
    capture = _RangeCapture(original, dctx)
    probe.local_window_splat = capture
    try:
        yield capture
    finally:
        probe.local_window_splat = original


def _import_probe(root):
    scripts_dir = Path(root) / "scripts"
    probe_path = scripts_dir / "sar_geom_grad_probe_v0_1.py"
    if not probe_path.is_file():
        raise FileNotFoundError(f"SAR probe not found: {probe_path}")
    path_text = str(scripts_dir)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    module = sys.modules.get("sar_geom_grad_probe_v0_1")
    if module is not None:
        module_file = Path(getattr(module, "__file__", "")).resolve()
        if module_file != probe_path.resolve():
            del sys.modules["sar_geom_grad_probe_v0_1"]
    return importlib.import_module("sar_geom_grad_probe_v0_1")


def _map_workspace_path(value, container_workspace, host_workspace):
    path = Path(str(value)).expanduser()
    container_root = Path(container_workspace)
    host_root = Path(host_workspace)
    if path.is_absolute() and path.exists():
        return path
    try:
        return container_root / path.relative_to(host_root)
    except ValueError:
        pass
    if path.is_absolute() and not path.exists():
        try:
            return host_root / path.relative_to(container_root)
        except ValueError:
            pass
    return path


def _host_display_path(path, container_workspace, host_workspace):
    value = Path(path)
    container_root = Path(container_workspace)
    host_root = Path(host_workspace)
    try:
        return host_root / value.relative_to(container_root)
    except ValueError:
        return value


def _load_range_file(path, direction, expected_shape):
    path = Path(path)
    if path.suffix == ".npy":
        payload = np.load(str(path), allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.ndim == 0:
            payload = payload.item()
    else:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if direction in payload:
            payload = payload[direction]
        elif isinstance(payload.get("range_gt"), dict) and direction in payload["range_gt"]:
            payload = payload["range_gt"][direction]
        else:
            raise KeyError(f"range GT file has no direction {direction!r}: {path}")
    target = torch.as_tensor(payload, dtype=torch.float32)
    if target.ndim != 2:
        raise ValueError(f"range GT for {direction!r} must be 2D, got {tuple(target.shape)}")
    if tuple(target.shape) != tuple(expected_shape):
        target = F.interpolate(
            target[None, None], size=expected_shape, mode="bilinear", align_corners=False
        )[0, 0]
    return target


def _select_dirs(active_dirs, spec, max_dirs):
    if spec:
        selected = [name.strip() for name in spec.split(",") if name.strip()]
        missing = [name for name in selected if name not in active_dirs]
        if missing:
            raise KeyError(f"SAR direction(s) not in session active_dirs: {missing}")
        return selected
    if max_dirs <= 0 or max_dirs >= len(active_dirs):
        return list(active_dirs)
    groups = {}
    group_order = []
    for name in active_dirs:
        prefix = name.split("_A", 1)[0]
        if prefix not in groups:
            groups[prefix] = []
            group_order.append(prefix)
        groups[prefix].append(name)
    selected = []
    index = 0
    while len(selected) < max_dirs:
        added = False
        for prefix in group_order:
            values = groups[prefix]
            if index < len(values):
                selected.append(values[index])
                added = True
                if len(selected) == max_dirs:
                    break
        if not added:
            break
        index += 1
    return selected


class SarGeometryLoss:
    def __init__(self, args, device):
        if float(getattr(args, "lambda_sar_int_geom", 0.0)) != 0.0:
            raise ValueError("lambda_sar_int_geom must remain 0; material intensity is disabled")
        self.args = args
        self.device = device
        root_value = str(getattr(args, "sar_root", "")).strip()
        if not root_value:
            raise ValueError("sar_root is required when SAR joint training is enabled")
        self.root = Path(root_value).expanduser()
        self.container_workspace = Path(
            getattr(args, "sar_container_workspace", "/local_workspace")
        )
        self.host_workspace = Path(
            getattr(args, "sar_host_workspace", "/home/sharp/github/3dgs_guan/workspace")
        )
        self.probe = _import_probe(self.root)
        session_path_arg = getattr(args, "sar_session_path", "")
        self.session_path = (
            _map_workspace_path(session_path_arg, self.container_workspace, self.host_workspace)
            if session_path_arg
            else self.root / "session.json"
        )
        if not self.session_path.is_file():
            raise FileNotFoundError(f"SAR session not found: {self.session_path}")
        with self.session_path.open("r", encoding="utf-8") as handle:
            self.session = json.load(handle)
        checkpoint_arg = getattr(args, "sar_stage1_checkpoint", "")
        checkpoint_value = checkpoint_arg or self.session.get("stage1_checkpoint")
        if not checkpoint_value:
            raise KeyError("session.json has no stage1_checkpoint")
        checkpoint_path = _map_workspace_path(
            checkpoint_value, self.container_workspace, self.host_workspace
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAR stage1 checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        active_dirs = list(self.session.get("active_dirs", []))
        self.dir_names = _select_dirs(
            active_dirs,
            str(getattr(args, "sar_dirs", "")),
            int(getattr(args, "sar_n_dirs", 0)),
        )
        self.dir_ctxs, self.transform = self.probe.build_dir_ctxs(
            self.dir_names, self.checkpoint, device
        )
        self.range_gt_mode = str(getattr(args, "sar_range_gt_mode", "B")).upper()
        if self.range_gt_mode not in {"A", "B"}:
            raise ValueError("sar_range_gt_mode must be A or B")
        range_path_arg = getattr(args, "sar_range_gt_path", "")
        self.range_gt_path = None
        if self.range_gt_mode == "A":
            if not range_path_arg:
                raise ValueError("sar_range_gt_path is required for range GT mode A")
            self.range_gt_path = _map_workspace_path(
                range_path_arg, self.container_workspace, self.host_workspace
            )
            if not self.range_gt_path.is_file():
                raise FileNotFoundError(f"independent range GT not found: {self.range_gt_path}")
        self.range_targets = {}
        self.range_valid_targets = {}
        for dctx in self.dir_ctxs:
            name = dctx["name"]
            shape = (int(dctx["H"]), int(dctx["W"]))
            if self.range_gt_mode == "A":
                target = _load_range_file(self.range_gt_path, name, shape)
                valid = torch.isfinite(target) & (target > 0)
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                rows = torch.arange(shape[0], dtype=torch.float32)
                target = float(dctx["r_near"]) + rows[:, None] / float(shape[0]) * (
                    float(dctx["r_far"]) - float(dctx["r_near"])
                )
                target = target.expand(shape[0], shape[1])
                valid = dctx["gt_target"] > 0
            self.range_targets[name] = target.to(device=device)
            self.range_valid_targets[name] = valid.to(device=device)
        self.forward_cfg = type("SarForwardConfig", (), {})()
        self.forward_cfg.win_r = int(getattr(args, "sar_win_r", 6))
        self.forward_cfg.sigma_min = float(getattr(args, "sar_sigma_min", 0.5))
        self.forward_cfg.sigma_max = float(getattr(args, "sar_sigma_max", 5.0))
        self.forward_cfg.tau_range = float(getattr(args, "sar_tau_range", 0.3))
        self.forward_cfg.alpha = float(getattr(args, "sar_alpha", 1.0))
        self.forward_cfg.window_margin_m = float(getattr(args, "sar_window_margin_m", 2.0))
        self.seed = int(getattr(args, "sar_seed", 42))
        self.gs_subsample = int(getattr(args, "sar_gs_subsample", 0))
        self.last_diags = []
        self.last_components = {"range": 0.0, "footprint": 0.0, "total": 0.0}

    def ramp_factor(self, iteration):
        warmup = int(getattr(self.args, "sar_warmup_iters", 7000))
        ramp_iters = int(getattr(self.args, "sar_ramp_iters", 7000))
        if iteration < warmup:
            return 0.0
        if ramp_iters <= 0:
            return 1.0
        return min(1.0, max(0.0, (iteration - warmup) / float(ramp_iters)))

    def _sample_indices(self, count, iteration, device):
        if self.gs_subsample <= 0 or self.gs_subsample >= count:
            return torch.arange(count, device=device)
        rng = np.random.default_rng(self.seed + int(iteration))
        values = np.sort(rng.choice(count, size=self.gs_subsample, replace=False))
        return torch.as_tensor(values, dtype=torch.long, device=device)

    def _geometry(self, gaussians, iteration):
        xyz_blender = gaussians._xyz
        indices = self._sample_indices(xyz_blender.shape[0], iteration, xyz_blender.device)
        xyz_blender = xyz_blender[indices]
        rotation = gaussians._rotation[indices]
        scaling = gaussians._scaling[indices]
        opacity = torch.sigmoid(gaussians._opacity[indices]).detach().reshape(-1)
        _xyz, scale, normal_blender, _tu, _tv = self.probe.decode_geometry(
            xyz_blender, rotation.detach(), scaling
        )
        transform = self.transform.to(dtype=_xyz.dtype, device=_xyz.device)
        xyz_pov = _xyz @ transform.T
        normal_pov = (normal_blender.detach() @ transform.T).detach()
        return xyz_pov, normal_pov, scale, opacity

    def _render(self, xyz, normal, scale, opacity, dctx):
        with _capture_probe_splat(self.probe, dctx) as capture:
            occupancy, diag = self.probe.render_direction_occupancy(
                xyz, normal, scale, opacity, dctx, self.forward_cfg
            )
        if capture.range_image is None or capture.valid is None:
            raise RuntimeError(f"probe renderer did not expose range for {dctx['name']}")
        return occupancy, capture.range_image, capture.valid, diag

    def compute(self, gaussians, iteration):
        xyz, normal, scale, opacity = self._geometry(gaussians, iteration)
        range_losses = []
        footprint_losses = []
        diagnostics = []
        for dctx in self.dir_ctxs:
            try:
                occupancy, predicted_range, predicted_valid, diag = self._render(
                    xyz, normal, scale, opacity, dctx
                )
            except RuntimeError as exc:
                if "no Gaussians fall inside" not in str(exc):
                    raise
                diagnostics.append({"dir": dctx["name"], "skipped": True, "reason": str(exc)})
                continue
            target_mask = dctx["gt_target"] > 0
            denom = max(1.0, float(target_mask.sum().item()) + float(diag["n_in_window"]))
            clipped = occupancy.clamp(1e-6, 1.0 - 1e-6)
            footprint = F.binary_cross_entropy(
                clipped, dctx["gt_target"].to(dtype=occupancy.dtype), reduction="sum"
            ) / denom
            footprint_losses.append(footprint)
            range_mask = target_mask & self.range_valid_targets[dctx["name"]]
            range_mask = range_mask & predicted_valid & torch.isfinite(predicted_range)
            if bool(range_mask.any()):
                diff = predicted_range[range_mask] - self.range_targets[dctx["name"]][range_mask]
                if str(getattr(self.args, "sar_range_loss", "huber")).lower() == "l1":
                    range_value = diff.abs().sum()
                else:
                    delta = float(getattr(self.args, "sar_huber_delta", 1.0))
                    range_value = F.smooth_l1_loss(
                        predicted_range[range_mask],
                        self.range_targets[dctx["name"]][range_mask],
                        beta=delta,
                        reduction="sum",
                    )
                range_losses.append(range_value / denom)
            else:
                range_losses.append(occupancy.sum() * 0.0)
            diag = dict(diag)
            diag.update(
                dir=dctx["name"],
                denom=denom,
                n_fg_px=float(target_mask.sum().item()),
                n_range_px=int(range_mask.sum().item()),
            )
            diagnostics.append(diag)
        zero = xyz.sum() * 0.0
        range_loss = torch.stack(range_losses).mean() if range_losses else zero
        footprint_loss = torch.stack(footprint_losses).mean() if footprint_losses else zero
        lambda_range = float(getattr(self.args, "lambda_sar_range", 0.0))
        lambda_footprint = float(getattr(self.args, "lambda_sar_fp", 0.0))
        ramp = self.ramp_factor(iteration)
        layover = zero
        total = ramp * (lambda_range * range_loss + lambda_footprint * footprint_loss)
        if bool(getattr(self.args, "sar_enable_layover_foreshortening", False)):
            total = total + float(getattr(self.args, "lambda_sar_layover", 0.0)) * layover
        self.last_diags = diagnostics
        self.last_components = {
            "range": float(range_loss.detach().item()),
            "footprint": float(footprint_loss.detach().item()),
            "ramp": float(ramp),
            "total": float(total.detach().item()),
        }
        return total, range_loss, footprint_loss, diagnostics

    def write_outputs(self, scene, iteration):
        model_path = Path(scene.model_path)
        output_ply = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not output_ply.is_file():
            scene.save(iteration)
        manifest_path = model_path / f"sar_geom_joint_{iteration}.manifest.json"
        pointer_path = model_path / "sar_geom_unfreeze_latest.json"
        output_ply_host = _host_display_path(
            output_ply, self.container_workspace, self.host_workspace
        )
        manifest_host = _host_display_path(
            manifest_path, self.container_workspace, self.host_workspace
        )
        pointer_host = _host_display_path(
            pointer_path, self.container_workspace, self.host_workspace
        )
        manifest = {
            "schema_version": "um_gsf_sar_geom_joint_manifest_v2",
            "mode": "hybrid_joint",
            "iteration": int(iteration),
            "output_ply": str(output_ply_host),
            "stage1_checkpoint": str(_host_display_path(self.checkpoint_path, self.container_workspace, self.host_workspace)),
            "range_gt_mode": self.range_gt_mode,
            "range_gt_path": str(_host_display_path(self.range_gt_path, self.container_workspace, self.host_workspace)) if self.range_gt_path else None,
            "dirs": self.dir_names,
            "n_dirs": len(self.dir_names),
            "gs_subsample": self.gs_subsample,
            "lambda_sar_range": float(getattr(self.args, "lambda_sar_range", 0.0)),
            "lambda_sar_fp": float(getattr(self.args, "lambda_sar_fp", 0.0)),
            "sar_warmup_iters": int(getattr(self.args, "sar_warmup_iters", 7000)),
            "sar_ramp_iters": int(getattr(self.args, "sar_ramp_iters", 7000)),
            "sar_every": int(getattr(self.args, "sar_every", 1)),
            "rotation_grad_blocked": True,
            "intensity_path_enabled": False,
            "last_components": self.last_components,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        with tmp_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp_manifest, manifest_path)
        pointer = {
            "schema_version": "um_gsf_sar_geom_unfreeze_latest_v2",
            "mode": "hybrid_joint",
            "exp": str(getattr(self.args, "sar_experiment_name", "hybrid")),
            "steps": int(iteration),
            "updated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
            "unfreeze_out_dir": str(_host_display_path(model_path, self.container_workspace, self.host_workspace)),
            "manifest_path": str(manifest_host),
            "output_ply": str(output_ply_host),
            "session_path": str(_host_display_path(model_path / "session.json",
                                                    self.container_workspace,
                                                    self.host_workspace)),
            "iteration": int(iteration),
        }
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_pointer = pointer_path.with_name(pointer_path.name + ".tmp")
        with tmp_pointer.open("w", encoding="utf-8") as handle:
            json.dump(pointer, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp_pointer, pointer_path)
        try:
            session_data = dict(self.session)
            session_data["sar_geom_unfreeze_manifest"] = str(manifest_host)
            session_out = model_path / "session.json"
            session_out.parent.mkdir(parents=True, exist_ok=True)
            tmp_session = session_out.with_name(session_out.name + ".tmp")
            with tmp_session.open("w", encoding="utf-8") as handle:
                json.dump(session_data, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
            os.replace(tmp_session, session_out)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[SAR] session pointer update skipped: {exc}")
        print(f"[SAR] output PLY: {output_ply_host}")
        print(f"[SAR] manifest: {manifest_host}")
        print(f"[SAR] latest pointer: {pointer_host}")
        return output_ply, manifest_path, pointer_path
