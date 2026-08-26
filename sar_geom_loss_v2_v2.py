"""Versioned t4-complete SAR geometry loss for the v0.3 carrier."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import sar_geom_loss_v2 as _base


def _import_v05(path: Path):
    module_name = "sar_silhouette_reconstruction_v0_5_t4_v2"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load t4 v0.5 loss module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_path(value, container_workspace: Path, host_workspace: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute() and path.exists():
        return path
    return _base._map_workspace_path(path, container_workspace, host_workspace)


class SarGeometryLoss(_base.SarGeometryLoss):
    """Use the t4 signal loss while preserving the native train_v2 contract."""

    def __init__(self, args, device):
        root_value = str(getattr(args, "sar_root", "")).strip()
        if root_value:
            runtime_root = Path(root_value).expanduser()
            for import_path in (
                runtime_root,
                runtime_root / "scripts",
                runtime_root.parent,
                runtime_root.parent.parent,
            ):
                if str(import_path) not in sys.path:
                    sys.path.insert(0, str(import_path))
        super().__init__(args, device)
        v05_value = str(getattr(args, "sar_v05_path", "")).strip()
        if not v05_value:
            raise ValueError("sar_v05_path is required for the t4 SAR signal loss")
        v05_path = _resolve_path(v05_value, self.container_workspace, self.host_workspace)
        if not v05_path.is_file():
            raise FileNotFoundError(f"t4 v0.5 loss module not found: {v05_path}")
        self.v05_path = v05_path
        self.v05 = _import_v05(v05_path)
        self.v05.probe = self.probe

        self.signal = str(getattr(args, "sar_signal", "combined"))
        if self.signal not in {"occupancy", "first_slant", "combined"}:
            raise ValueError(f"unsupported active SAR signal: {self.signal}")
        self.contrib_root = None
        if self.signal in {"first_slant", "combined"}:
            contrib_value = str(getattr(args, "sar_contrib_root", "")).strip()
            contrib_path = (
                _resolve_path(contrib_value, self.container_workspace, self.host_workspace)
                if contrib_value
                else self.root / "gateD"
            )
            if not contrib_path.is_dir():
                raise FileNotFoundError(f"gateD Contributions root not found: {contrib_path}")
            self.contrib_root = contrib_path

        threshold = float(getattr(args, "sar_first_gt_threshold_rel", 1.0e-6))
        self.dir_ctxs, self.transform = self.v05.prepare_contexts(
            self.dir_names,
            self.checkpoint,
            device,
            stride=1,
            first_slant_enabled=False,
            first_gt_threshold_rel=threshold,
        )
        if self.signal in {"first_slant", "combined"}:
            for context in self.dir_ctxs:
                self.v05.add_first_slant_support(
                    context, self.contrib_root, threshold
                )

        self.forward_cfg.soft_alpha = float(getattr(args, "sar_soft_alpha", 5.0))
        self.forward_cfg.first_range_margin_m = float(
            getattr(args, "sar_first_range_margin_m", 0.5)
        )
        self.forward_cfg.first_surface_thickness_m = float(
            getattr(args, "sar_first_surface_thickness_m", 1.0)
        )
        self.forward_cfg.first_range_temperature_m = float(
            getattr(args, "sar_first_range_temperature_m", 0.25)
        )
        self.forward_cfg.first_gt_threshold_rel = threshold
        self.forward_cfg.spec_selectivity_min = float(
            getattr(args, "sar_spec_selectivity_min", 0.0)
        )
        self.component_log_interval = max(
            1, int(getattr(args, "sar_component_log_interval", 10))
        )
        self.last_components = {
            "occupancy": 0.0,
            "first_front": 0.0,
            "first_shadow": 0.0,
            "specular_normal": 0.0,
            "first_slant": 0.0,
            "selected": 0.0,
            "ramp": 0.0,
            "total": 0.0,
        }

    def ramp_factor(self, iteration):
        value = super().ramp_factor(iteration)
        warmup = int(getattr(self.args, "sar_warmup_iters", 7000))
        if iteration % self.component_log_interval == 0 and iteration < warmup:
            print(
                "[SAR-COMP] iter={:05d} active=False occupancy=0.000000e+00 "
                "first_front=0.000000e+00 first_shadow=0.000000e+00 "
                "specular_normal=0.000000e+00 selected=0.000000e+00 finite=1".format(
                    iteration
                ),
                flush=True,
            )
        return value

    def _geometry(self, gaussians, iteration):
        xyz_blender = gaussians._xyz
        indices = self._sample_indices(
            xyz_blender.shape[0], iteration, xyz_blender.device
        )
        xyz_blender = xyz_blender[indices]
        rotation = gaussians._rotation[indices]
        scaling = gaussians._scaling[indices]
        opacity = torch.sigmoid(gaussians._opacity[indices]).reshape(-1)
        xyz, scale, normal_blender, _tu, _tv = self.probe.decode_geometry(
            xyz_blender, rotation, scaling
        )
        transform = self.transform.to(dtype=xyz.dtype, device=xyz.device)
        xyz_pov = xyz @ transform.T
        normal_pov = normal_blender @ transform.T
        return xyz_pov, normal_pov, scale, opacity

    def _render(self, xyz, normal, scale, opacity, dctx):
        return self.probe.render_direction_occupancy(
            xyz, normal, scale, opacity, dctx, self.forward_cfg
        )

    @staticmethod
    def _finite(value):
        return bool(torch.isfinite(value).all().item())

    def compute(self, gaussians, iteration):
        xyz, normal, scale, opacity = self._geometry(gaussians, iteration)
        occupancy_losses = []
        front_losses = []
        shadow_losses = []
        active_contexts = []
        diagnostics = []
        for dctx in self.dir_ctxs:
            try:
                coverage, diag = self._render(xyz, normal, scale, opacity, dctx)
            except RuntimeError as exc:
                if "no Gaussians fall inside" not in str(exc):
                    raise
                diagnostics.append(
                    {"dir": dctx["name"], "skipped": True, "reason": str(exc)}
                )
                continue
            prediction = self.v05.soft_occupancy(
                coverage, self.forward_cfg.soft_alpha
            )
            target = dctx["gt_target"].to(
                device=prediction.device, dtype=prediction.dtype
            )
            if self.signal in {"occupancy", "combined"}:
                occupancy_losses.append(
                    F.binary_cross_entropy(
                        prediction.clamp(1.0e-6, 1.0 - 1.0e-6), target
                    )
                )
            if self.signal in {"first_slant", "combined"}:
                front, shadow, stats = self.v05.first_slant_range_terms(
                    prediction, dctx, self.forward_cfg
                )
                front_losses.append(front)
                shadow_losses.append(shadow)
            else:
                stats = {}
            active_contexts.append(dctx)
            diag = dict(diag)
            diag.update(dir=dctx["name"], **stats)
            diagnostics.append(diag)

        zero = xyz.sum() * 0.0
        occupancy_component = (
            torch.stack(occupancy_losses).mean() if occupancy_losses else zero
        )
        first_front = torch.stack(front_losses).mean() if front_losses else zero
        first_shadow = torch.stack(shadow_losses).mean() if shadow_losses else zero
        specular = zero
        spec_active_fraction = 0.0
        spec_weight_sum = 0.0
        if self.signal in {"first_slant", "combined"} and active_contexts:
            specular, spec_active_fraction, spec_weight_sum = (
                self.v05.specular_normal_loss(
                    xyz,
                    normal,
                    active_contexts,
                    self.forward_cfg.spec_selectivity_min,
                )
            )
        first_slant_component = first_front + first_shadow + specular
        if self.signal == "occupancy":
            selected = occupancy_component
        elif self.signal == "first_slant":
            selected = first_slant_component
        else:
            selected = occupancy_component + first_slant_component

        values = (
            occupancy_component,
            first_front,
            first_shadow,
            specular,
            first_slant_component,
            selected,
        )
        labels = (
            "occupancy",
            "first_front",
            "first_shadow",
            "specular_normal",
            "first_slant",
            "selected",
        )
        for label, value in zip(labels, values):
            if not self._finite(value):
                raise FloatingPointError(f"{label} SAR component is non-finite")

        lambda_occupancy = float(getattr(self.args, "lambda_sar_fp", 0.0))
        lambda_first_slant = float(getattr(self.args, "lambda_sar_range", 0.0))
        ramp = self.ramp_factor(iteration)
        total = ramp * (
            lambda_occupancy * occupancy_component
            + lambda_first_slant * first_slant_component
        )
        if bool(getattr(self.args, "sar_enable_layover_foreshortening", False)):
            total = total + float(getattr(self.args, "lambda_sar_layover", 0.0)) * zero
        if not self._finite(total):
            raise FloatingPointError("total SAR geometry loss is non-finite")

        self.last_diags = diagnostics
        self.last_components = {
            "occupancy": float(occupancy_component.detach().item()),
            "first_front": float(first_front.detach().item()),
            "first_shadow": float(first_shadow.detach().item()),
            "specular_normal": float(specular.detach().item()),
            "first_slant": float(first_slant_component.detach().item()),
            "selected": float(selected.detach().item()),
            "ramp": float(ramp),
            "total": float(total.detach().item()),
            "spec_active_fraction": float(spec_active_fraction),
            "spec_weight_sum": float(spec_weight_sum),
        }
        if iteration % self.component_log_interval == 0:
            print(
                "[SAR-COMP] iter={:05d} active=True occupancy={:.6e} "
                "first_front={:.6e} first_shadow={:.6e} "
                "specular_normal={:.6e} selected={:.6e} total={:.6e} finite=1".format(
                    iteration,
                    self.last_components["occupancy"],
                    self.last_components["first_front"],
                    self.last_components["first_shadow"],
                    self.last_components["specular_normal"],
                    self.last_components["selected"],
                    self.last_components["total"],
                ),
                flush=True,
            )

        return total, first_slant_component, occupancy_component, diagnostics

    def write_outputs(self, scene, iteration):
        result = super().write_outputs(scene, iteration)
        manifest_path = result[1]
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.update(
            {
                "sar_signal": self.signal,
                "sar_v05_path": str(
                    _base._host_display_path(
                        self.v05_path, self.container_workspace, self.host_workspace
                    )
                ),
                "sar_contrib_root": (
                    str(
                        _base._host_display_path(
                            self.contrib_root,
                            self.container_workspace,
                            self.host_workspace,
                        )
                    )
                    if self.contrib_root is not None
                    else None
                ),
                "rotation_grad_blocked": False,
                "t4_components": self.last_components,
            }
        )
        temporary = manifest_path.with_name(manifest_path.name + ".t4.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, manifest_path)
        return result