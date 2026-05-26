from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelConv2d(nn.Module):
    AD_PERMUTATION = (3, 0, 1, 6, 4, 2, 7, 8, 5)
    RD_POSITIVE = (0, 2, 4, 10, 14, 20, 22, 24)
    RD_NEGATIVE = (6, 7, 8, 11, 13, 16, 17, 18)

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 groups=1, bias=False, theta=1.0):
        super(ParallelConv2d, self).__init__()
        if kernel_size != 3:
            raise ValueError("ParallelConv2d only supports kernel_size=3")
        if in_channels % 4 != 0:
            raise ValueError("in_channels must be divisible by 4")
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        if groups < 1:
            raise ValueError("groups must be a positive integer")
        if not 0.0 <= float(theta) <= 1.0:
            raise ValueError("theta must be in [0, 1]")

        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.theta = float(theta)
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = dilation
        self.padding = padding

        n_in = in_channels // 4
        n_out = out_channels // 4
        self.branch_in_channels = n_in
        self.branch_out_channels = n_out

        if n_in % groups != 0:
            raise ValueError("branch in_channels must be divisible by groups")
        if n_out % groups != 0:
            raise ValueError("branch out_channels must be divisible by groups")
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(n_in, n_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias)
            for _ in range(4)
        ])

    def cd_func(self, x, conv_layer):
        weights = conv_layer.weight
        bias = conv_layer.bias

        weights_c = weights.sum(dim=[2, 3], keepdim=True) * self.theta
        yc = F.conv2d(x, weights_c, bias=None, stride=self.stride, padding=0, groups=self.groups)
        y = F.conv2d(x, weights, bias=bias, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
        return y - yc

    def ad_func(self, x, conv_layer):
        weights = conv_layer.weight
        bias = conv_layer.bias

        shape = weights.shape
        weights = weights.view(shape[0], shape[1], -1)
        weights_conv = (weights - self.theta * weights[:, :, self.AD_PERMUTATION]).view(shape)
        y = F.conv2d(x, weights_conv, bias=bias, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
        return y

    def rd_func(self, x, conv_layer):
        weights = conv_layer.weight
        bias = conv_layer.bias

        padding = 2 * self.dilation

        shape = weights.shape
        buffer = weights.new_zeros(shape[0], shape[1], 5 * 5)
        weights = weights.view(shape[0], shape[1], -1)
        buffer[:, :, self.RD_POSITIVE] = weights[:, :, 1:]
        buffer[:, :, self.RD_NEGATIVE] = -weights[:, :, 1:] * self.theta
        buffer[:, :, 12] = weights[:, :, 0] * (1.0 - self.theta)
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        y = F.conv2d(x, buffer, bias=bias, stride=self.stride, padding=padding,
                     dilation=self.dilation, groups=self.groups)
        return y

    def forward(self, x):
        c_in = x.shape[1]
        if c_in != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c_in}")

        x_split = torch.split(x, self.branch_in_channels, dim=1)
        y_list = []

        for idx, conv_layer in enumerate(self.conv_layers):
            if idx == 0:
                y = conv_layer(x_split[idx])
            elif idx == 1:
                y = self.cd_func(x_split[idx], conv_layer)
            elif idx == 2:
                y = self.ad_func(x_split[idx], conv_layer)
            elif idx == 3:
                y = self.rd_func(x_split[idx], conv_layer)
            y_list.append(y)
        y = torch.cat(y_list, dim=1)
        return y

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_parallel_conv=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if use_parallel_conv:
            self.conv1 = ParallelConv2d(inplanes, planes, stride=stride,
                                        dilation=dilation, groups=groups, bias=False)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if use_parallel_conv:
            self.conv2 = ParallelConv2d(planes, planes, dilation=dilation,
                                        groups=groups, bias=False)
        else:
            self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetParallel(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, initial_channels=64):
        super(ResNetParallel, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = initial_channels
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], use_parallel_conv=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_parallel_conv=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_parallel_conv=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_parallel_conv=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, use_parallel_conv=False):
        norm_layer = nn.BatchNorm2d
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer=norm_layer,
                            use_parallel_conv=use_parallel_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                use_parallel_conv=use_parallel_conv))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))

import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 s: float = 64.0,
                 m: float = 0.5,
                 easy_margin: bool = True,
                 warmup_epochs: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('m_max', torch.tensor(m, dtype=torch.float))
        self.s = s
        self.easy_margin = easy_margin
        self.warmup_epochs = warmup_epochs
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.01)

    def _get_margin(self, current_epoch: int) -> torch.Tensor:
        if self.warmup_epochs == 0 or current_epoch >= self.warmup_epochs:
            return self.m_max
        ratio = current_epoch / float(self.warmup_epochs)
        return self.m_max * ratio

    @torch.no_grad()
    def _precompute_trig(self, m: torch.Tensor):
        cos_m = torch.cos(m)
        sin_m = torch.sin(m)
        th    = torch.cos(torch.pi - m)
        mm    = torch.sin(torch.pi - m) * m
        return cos_m, sin_m, th, mm

    def forward(self,
                input: torch.Tensor,
                label: torch.Tensor,
                current_epoch: int = 0,total_epoch=0):
        m = self._get_margin(current_epoch)
        cos_m, sin_m, th, mm = self._precompute_trig(m)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.square()).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th, phi, cosine - mm)
        one_hot = torch.zeros_like(cosine, device=input.device)
        one_hot.scatter_(1, label.reshape(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1. - one_hot) * cosine)
        output *= self.s
        return output

TrainStage = Literal["global", "local", "integrated", "full"]
_VALID_TRAIN_STAGES = {"global", "local", "integrated", "full"}
ClassifierMode = Literal["softmax", "arcface"]
_VALID_CLASSIFIERS = {"softmax", "arcface"}
LocalFeatureStage = Literal["stage2", "stage3"]
_VALID_LOCAL_FEATURE_STAGES = {"stage2", "stage3"}

DF_IRIS_MODEL_DEFAULTS: dict[str, int | float] = {
    "proj_dim": 128,
    "k_local": 4,
    "num_tokens": 4,
    "dropout_p": 0.1,
    "fusion_branch_dropout": 0.10,
    "diversity_weight": 0.03,
    "local_bg_loss_weight": 0.003,
    "local_patch_radius": 1,
    "arcface_scale": 16.0,
    "arcface_margin": 0.1,
    "arcface_warmup_epochs": 30,
    "cdm_negative_topq": 8,
    "nms_radius": 1,
    "saliency_intersection_weight": 0.0,
}

class TokenLearner(nn.Module):

    def __init__(self, in_channels: int, num_tokens: int = 6, hidden_dim: int = 64, dropout_p: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = nn.GroupNorm(1, in_channels)
        self.gating = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden_dim, num_tokens, 1, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        gates = self.gating(self.norm(x)).view(b, self.num_tokens, -1)
        if prior is not None:
            prior = prior.view(b, 1, h * w).to(dtype=gates.dtype)
            gates = gates + torch.log(prior.clamp_min(1e-6))
        gates = gates.softmax(dim=2)
        x_flat = x.view(b, c, h * w)
        tokens = torch.einsum("btn,bcn->btc", gates, x_flat)
        if return_attention:
            return tokens, gates.view(b, self.num_tokens, h, w)
        return tokens

class MultiBranchResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 120,
        *,
        k_local: int = int(DF_IRIS_MODEL_DEFAULTS["k_local"]),
        num_tokens: int = int(DF_IRIS_MODEL_DEFAULTS["num_tokens"]),
        proj_dim: int = int(DF_IRIS_MODEL_DEFAULTS["proj_dim"]),
        dropout_p: float = float(DF_IRIS_MODEL_DEFAULTS["dropout_p"]),
        fusion_branch_dropout: float = float(DF_IRIS_MODEL_DEFAULTS["fusion_branch_dropout"]),
        train_stage: TrainStage = "global",
        num_frozen_blocks: int = 0,
        cdm_negative_topq: int = int(DF_IRIS_MODEL_DEFAULTS["cdm_negative_topq"]),
        diversity_weight: float = float(DF_IRIS_MODEL_DEFAULTS["diversity_weight"]),
        local_bg_loss_weight: float = float(DF_IRIS_MODEL_DEFAULTS["local_bg_loss_weight"]),
        local_patch_radius: int = int(DF_IRIS_MODEL_DEFAULTS["local_patch_radius"]),
        local_feature_stage: LocalFeatureStage = "stage3",
        token_prior_radius: int | None = None,
        nms_radius: int = int(DF_IRIS_MODEL_DEFAULTS["nms_radius"]),
        saliency_intersection_weight: float = float(DF_IRIS_MODEL_DEFAULTS["saliency_intersection_weight"]),
        arcface_scale: float = float(DF_IRIS_MODEL_DEFAULTS["arcface_scale"]),
        arcface_margin: float = float(DF_IRIS_MODEL_DEFAULTS["arcface_margin"]),
        arcface_warmup_epochs: int = int(DF_IRIS_MODEL_DEFAULTS["arcface_warmup_epochs"]),
    ):
        super().__init__()
        if k_local < 1:
            raise ValueError("k_local must be positive")
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        if local_patch_radius < 0:
            raise ValueError("local_patch_radius must be non-negative")
        if token_prior_radius is not None and token_prior_radius < 0:
            raise ValueError("token_prior_radius must be non-negative")
        if nms_radius < 0:
            raise ValueError("nms_radius must be non-negative")
        if saliency_intersection_weight < 0:
            raise ValueError("saliency_intersection_weight must be non-negative")
        if not 0.0 <= float(fusion_branch_dropout) <= 0.5:
            raise ValueError("fusion_branch_dropout must be in [0, 0.5]")
        if local_bg_loss_weight < 0:
            raise ValueError("local_bg_loss_weight must be non-negative")
        if train_stage not in _VALID_TRAIN_STAGES:
            raise ValueError(f"train_stage must be one of {sorted(_VALID_TRAIN_STAGES)}, got {train_stage!r}")
        if local_feature_stage not in _VALID_LOCAL_FEATURE_STAGES:
            raise ValueError(
                f"local_feature_stage must be one of {sorted(_VALID_LOCAL_FEATURE_STAGES)}, "
                f"got {local_feature_stage!r}"
            )

        self.k_local = k_local
        self.num_tokens = num_tokens
        self.train_stage = train_stage
        self.cdm_negative_topq = cdm_negative_topq
        self.diversity_weight = diversity_weight
        self.local_bg_loss_weight = float(local_bg_loss_weight)
        self.local_patch_radius = local_patch_radius
        self.nms_radius = nms_radius
        self.saliency_intersection_weight = float(saliency_intersection_weight)
        self.fusion_branch_dropout = float(fusion_branch_dropout)
        self.local_feature_stage = local_feature_stage
        self.token_prior_radius = local_patch_radius if token_prior_radius is None else token_prior_radius

        rp = ResNetParallel(BasicBlock, [2, 2, 2, 2])
        self.stage0 = nn.Sequential(rp.conv1, rp.bn1, rp.relu, rp.maxpool)
        self.stage1 = rp.layer1
        self.stage2 = rp.layer2
        self.stage3 = rp.layer3
        self.stage4 = rp.layer4
        self._freeze_stages(num_frozen_blocks)

        self.global_pool = rp.avgpool
        self.global_fc = nn.Linear(512, proj_dim, bias=False)
        self.global_norm = nn.LayerNorm(proj_dim)

        local_in_channels = 128 if local_feature_stage == "stage2" else 256
        self.local_conv = nn.Sequential(
            nn.Conv2d(local_in_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.local_fc = nn.Linear(512, proj_dim, bias=False)
        self.local_norm = nn.LayerNorm(proj_dim)

        self.token_learner = TokenLearner(512, num_tokens=num_tokens, hidden_dim=64, dropout_p=dropout_p)
        self.token_fusion_norm = nn.LayerNorm(num_tokens * 512)
        self.integrated_fc = nn.Linear(proj_dim + num_tokens * 512, proj_dim, bias=False)
        self.integrated_norm = nn.LayerNorm(proj_dim)

        self.head_global = nn.Linear(proj_dim, num_classes, bias=False)
        self.head_local = nn.Linear(proj_dim, num_classes, bias=False)
        self.head_integrated = nn.Linear(proj_dim, num_classes, bias=False)

        self.arc_margin_global = ArcMarginProduct(
            proj_dim,
            num_classes,
            s=arcface_scale,
            m=arcface_margin,
            warmup_epochs=arcface_warmup_epochs,
        )
        self.arc_margin_global.weight = self.head_global.weight
        self.arc_margin_local = ArcMarginProduct(
            proj_dim,
            num_classes,
            s=arcface_scale,
            m=arcface_margin,
            warmup_epochs=arcface_warmup_epochs,
        )
        self.arc_margin_local.weight = self.head_local.weight
        self.arc_margin_integrated = ArcMarginProduct(
            proj_dim,
            num_classes,
            s=arcface_scale,
            m=arcface_margin,
            warmup_epochs=arcface_warmup_epochs,
        )
        self.arc_margin_integrated.weight = self.head_integrated.weight

        self.dropout = nn.Dropout(p=dropout_p)
        self.logit_lambda = nn.Parameter(torch.zeros(1))

    @property
    def backbone(self) -> nn.Sequential:
        return nn.Sequential(self.stage0, self.stage1, self.stage2, self.stage3, self.stage4)

    def _freeze_stages(self, num_stages: int) -> None:
        stages = [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]
        for stage in stages[: max(0, min(num_stages, len(stages)))]:
            for param in stage.parameters():
                param.requires_grad = False

    def _backbone_feats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage0(x)
        x = self.stage1(x)
        feat_stage2 = self.stage2(x)
        feat_stage3 = self.stage3(feat_stage2)
        feat_global = self.stage4(feat_stage3)
        return feat_stage2, feat_stage3, feat_global

    def _forward_global(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, feat_global_map = self._backbone_feats(x)
        pooled = self.global_pool(feat_global_map).flatten(1)
        g_feat = self.global_norm(self.global_fc(pooled))
        return g_feat, self.head_global(g_feat)

    def _classification_logits(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Linear,
        arc_margin: ArcMarginProduct,
        classifier: ClassifierMode,
        current_epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        if classifier == "softmax":
            return head(features)
        if classifier == "arcface":
            return arc_margin(features, labels, current_epoch, total_epochs)
        raise ValueError(f"classifier must be one of {sorted(_VALID_CLASSIFIERS)}, got {classifier!r}")

    @staticmethod
    def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        return (x - x_min) / (x_max - x_min + eps)

    def _global_spatial_prior(
        self,
        feat_global_map: torch.Tensor,
        spatial_size: Tuple[int, int],
        detach: bool = True,
    ) -> torch.Tensor:
        x = feat_global_map.detach() if detach else feat_global_map
        prior = x.abs().mean(dim=1, keepdim=True)
        prior = F.interpolate(
            prior,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        prior = prior.flatten(1)
        return self._normalize_map(prior)

    def _expand_support(
        self,
        support: torch.Tensor,
        spatial_size: Tuple[int, int],
        radius: int,
    ) -> torch.Tensor:
        if radius <= 0:
            return support
        b, n = support.shape
        h, w = spatial_size
        if h * w != n:
            raise ValueError("spatial_size must match flattened support length")
        support_map = support.view(b, 1, h, w)

        coord = torch.arange(-radius, radius + 1, device=support.device, dtype=support.dtype)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        sigma = max(float(radius) / 1.5, 0.75)
        kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(1e-6)
        kernel = kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)

        expanded = F.conv2d(support_map, kernel, padding=radius)
        return self._normalize_map(expanded.flatten(1))

    def _negative_reference(self, sim_all: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        b, n, c = sim_all.shape
        target_mask = torch.zeros(b, c, device=sim_all.device, dtype=torch.bool)
        target_mask.scatter_(1, target_idx.view(b, 1), True)
        num_neg = max(c - 1, 1)
        if self.cdm_negative_topq is None or self.cdm_negative_topq <= 0:
            neg_sum = sim_all.masked_fill(target_mask[:, None, :], 0.0).sum(dim=2)
            return neg_sum / num_neg

        masked_neg = sim_all.masked_fill(target_mask[:, None, :], float("-inf"))
        q = min(int(self.cdm_negative_topq), num_neg)
        return masked_neg.topk(q, dim=2).values.mean(dim=2)

    @staticmethod
    def _positions_to_indices(rows: torch.Tensor, cols: torch.Tensor, width: int) -> torch.Tensor:
        return rows * width + cols

    def _nms_topk(
        self,
        saliency: torch.Tensor,
        k: int,
        spatial_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        b, n = saliency.shape
        k = max(1, min(k, n))
        if spatial_size is None:
            return saliency.topk(k, dim=1).indices

        h, w = spatial_size
        if h * w != n:
            raise ValueError("spatial_size must match flattened saliency length")

        scores = saliency.clone()
        selected = []
        batch = torch.arange(b, device=saliency.device)
        radius = self.nms_radius

        for _ in range(k):
            idx = scores.argmax(dim=1)
            selected.append(idx)
            if radius == 0:
                scores[batch, idx] = -torch.inf
                continue

            row = idx // w
            col = idx % w
            for dr in range(-radius, radius + 1):
                rr = (row + dr).clamp(0, h - 1)
                for dc in range(-radius, radius + 1):
                    cc = (col + dc).clamp(0, w - 1)
                    suppress = self._positions_to_indices(rr, cc, w)
                    scores[batch, suppress] = -torch.inf

        return torch.stack(selected, dim=1)

    def _cam_cdm_mask(
        self,
        local_emb: torch.Tensor,
        target_idx: torch.Tensor,
        *,
        global_prior: torch.Tensor,
        local_energy: torch.Tensor,
        keep_ratio: float | None = None,
        spatial_size: Tuple[int, int] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, n, _ = local_emb.shape
        if keep_ratio is None:
            keep_ratio = self.k_local / n
        k = max(1, min(n, int(round(n * max(0.0, min(1.0, keep_ratio))))))

        local_norm = F.normalize(local_emb, dim=2)
        weight = F.normalize(self.head_local.weight, dim=1)
        sim_all = torch.einsum("bnd,cd->bnc", local_norm, weight)
        target = sim_all.gather(2, target_idx.view(b, 1, 1).expand(-1, n, 1)).squeeze(2)
        negative_ref = self._negative_reference(sim_all, target_idx)

        raw_cam = F.relu(target)
        cdm_tau = 0.07
        raw_cdm = F.softplus((target - negative_ref) / cdm_tau) * cdm_tau
        raw_cam_n = self._normalize_map(raw_cam)
        raw_cdm_n = self._normalize_map(raw_cdm)
        global_prior_n = self._normalize_map(global_prior)
        local_energy_n = self._normalize_map(local_energy)

        guided_cam = self._normalize_map(
            (raw_cam_n + 1e-6).pow(0.65)
            * (global_prior_n + 1e-6).pow(1.00)
            * (local_energy_n + 1e-6).pow(0.25)
        )
        guided_cdm = self._normalize_map(
            (raw_cdm_n + 1e-6).pow(1.15)
            * (global_prior_n + 1e-6).pow(0.70)
            * (local_energy_n + 1e-6).pow(0.45)
        )
        alpha = torch.sigmoid(self.logit_lambda)
        intersection = torch.sqrt((guided_cam + 1e-6) * (guided_cdm + 1e-6))
        saliency = self._normalize_map(
            alpha * guided_cam
            + (1.0 - alpha) * guided_cdm
            + self.saliency_intersection_weight * intersection
        )

        idx = self._nms_topk(saliency, k, spatial_size)
        mask = torch.zeros_like(saliency, dtype=torch.bool)
        batch = torch.arange(b, device=saliency.device).unsqueeze(1)
        mask[batch, idx] = True
        return mask, saliency, {
            "raw_cam": raw_cam,
            "raw_cdm": raw_cdm,
            "cam": guided_cam,
            "cdm": guided_cdm,
            "target_idx": target_idx,
            "selected_indices": idx,
        }

    def _select_local_descriptors(
        self,
        loc_maps: torch.Tensor,
        saliency: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = loc_maps.shape
        radius = self.local_patch_radius
        if radius == 0:
            loc_flat = loc_maps.flatten(2).transpose(1, 2)
            batch = torch.arange(b, device=loc_maps.device).unsqueeze(1)
            return loc_flat[batch, idx]

        kernel = 2 * radius + 1
        patches = F.unfold(loc_maps, kernel_size=kernel, padding=radius)
        patches = patches.view(b, c, kernel * kernel, h * w)

        sal_map = saliency.view(b, 1, h, w)
        sal_patches = F.unfold(sal_map, kernel_size=kernel, padding=radius)
        sal_patches = sal_patches.view(b, kernel * kernel, h * w)

        gather_idx_feat = idx[:, None, None, :].expand(-1, c, kernel * kernel, -1)
        selected_patches = patches.gather(3, gather_idx_feat)
        gather_idx_sal = idx[:, None, :].expand(-1, kernel * kernel, -1)
        selected_saliency = sal_patches.gather(2, gather_idx_sal).transpose(1, 2)
        patch_weights = selected_saliency.softmax(dim=2).to(dtype=loc_maps.dtype)
        selected_patches = selected_patches.permute(0, 3, 2, 1)
        return torch.einsum("bkr,bkrc->bkc", patch_weights, selected_patches)

    def _apply_fusion_branch_dropout(
        self,
        g_feat: torch.Tensor,
        token_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.fusion_branch_dropout
        if not self.training or self.train_stage not in {"integrated", "full"} or p <= 0.0:
            return g_feat, token_flat

        branch_choice = torch.rand(g_feat.size(0), 1, device=g_feat.device)
        drop_global = branch_choice < p
        drop_local = (branch_choice >= p) & (branch_choice < 2.0 * p)
        scale = 1.0 / (1.0 - p)
        g_keep = (~drop_global).to(dtype=g_feat.dtype)
        local_keep = (~drop_local).to(dtype=token_flat.dtype)
        return g_feat * g_keep * scale, token_flat * local_keep * scale

    def _forward_features(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feat_stage2, feat_stage3, feat_global_map = self._backbone_feats(x)
        b = x.size(0)

        pooled = self.global_pool(feat_global_map).flatten(1)
        g_feat = self.global_norm(self.global_fc(pooled))
        target_idx = labels if labels is not None else self.head_global(g_feat).detach().argmax(dim=1)

        feat_local = feat_stage2 if self.local_feature_stage == "stage2" else feat_stage3
        _, _, h, w = feat_local.shape
        global_prior = self._global_spatial_prior(feat_global_map, spatial_size=(h, w), detach=True)
        global_prior_map = global_prior.view(b, 1, h, w)
        local_input_gate = 0.10 + 0.90 * global_prior_map.to(dtype=feat_local.dtype)
        feat_local = feat_local * local_input_gate
        loc_maps = self.local_conv(feat_local)
        _, c, h, w = loc_maps.shape
        local_energy = loc_maps.detach().pow(2).mean(dim=1).flatten(1)
        local_energy = self._normalize_map(local_energy)
        loc_flat = loc_maps.flatten(2).transpose(1, 2)
        loc_emb = self.local_norm(self.local_fc(loc_flat.reshape(-1, c))).view(b, h * w, -1)

        keep_ratio = self.k_local / float(h * w)
        mask, selection_saliency, aux = self._cam_cdm_mask(
            loc_emb,
            target_idx,
            global_prior=global_prior,
            local_energy=local_energy,
            keep_ratio=keep_ratio,
            spatial_size=(h, w),
        )
        idx = aux["selected_indices"]
        local_saliency = selection_saliency
        if labels is not None and self.train_stage in {"local", "integrated"}:
            selection_saliency = selection_saliency.detach()
        local_vectors = self._select_local_descriptors(loc_maps, selection_saliency, idx)
        local_feats = self.local_norm(self.local_fc(local_vectors.reshape(-1, c))).view(b, -1, g_feat.size(1))
        local_feats = self.dropout(local_feats)
        local_flat = local_feats.reshape(-1, local_feats.size(-1))

        peak_support = torch.zeros_like(selection_saliency)
        peak_values = selection_saliency.gather(1, idx).clamp_min(1e-6)
        peak_support.scatter_(1, idx, peak_values)
        peak_support = self._expand_support(peak_support, spatial_size=(h, w), radius=self.token_prior_radius)
        token_support = self._normalize_map(
            (peak_support + 1e-6).pow(0.80)
            * (global_prior + 1e-6).pow(0.60)
            * (local_energy + 1e-6).pow(0.40)
        ).view(b, 1, h, w)
        token_gate = 0.05 + 0.95 * token_support.to(dtype=loc_maps.dtype)
        token_maps = loc_maps * token_gate
        tokens, token_attention = self.token_learner(token_maps, prior=token_support, return_attention=True)
        token_flat = self.token_fusion_norm(tokens.flatten(1))
        g_fused, token_fused = self._apply_fusion_branch_dropout(g_feat, token_flat)
        fused_input = torch.cat([g_fused, token_fused], dim=1)
        i_feat = self.dropout(self.integrated_norm(self.integrated_fc(fused_input)))

        aux.update(
            {
                "saliency": selection_saliency,
                "local_saliency": local_saliency,
                "global_prior": global_prior,
                "local_energy": local_energy,
                "mask": mask,
                "local_indices": idx,
                "token_attention": token_attention,
                "token_support": token_support,
                "token_gate": token_gate,
                "local_input_gate": local_input_gate,
                "local_bg_loss": self._local_background_loss(loc_maps, global_prior),
                "cam_tv_loss": self._tv_loss(aux["cam"], (h, w)),
                "cdm_tv_loss": self._tv_loss(aux["cdm"], (h, w)),
                "map_prior_loss": self._prior_consistency_loss(
                    selection_saliency,
                    self._normalize_map(global_prior * local_energy),
                ),
                "local_feature_stage": self.local_feature_stage,
            }
        )
        return g_feat, local_flat, i_feat, aux

    @staticmethod
    def _diversity_loss(local_flat: torch.Tensor, batch_size: int, k_local: int) -> torch.Tensor:
        if k_local <= 1:
            return local_flat.new_zeros(())
        local = F.normalize(local_flat.view(batch_size, k_local, -1), dim=2)
        gram = torch.bmm(local, local.transpose(1, 2))
        eye = torch.eye(k_local, device=local.device, dtype=torch.bool).unsqueeze(0)
        return gram.masked_select(~eye).pow(2).mean()

    @staticmethod
    def _local_background_loss(loc_maps: torch.Tensor, global_prior: torch.Tensor) -> torch.Tensor:
        local_energy = loc_maps.pow(2).mean(dim=1).flatten(1)
        local_energy = local_energy / (local_energy.mean(dim=1, keepdim=True) + 1e-6)
        bg_weight = (1.0 - global_prior).detach()
        loss = (local_energy * bg_weight).sum(dim=1) / (bg_weight.sum(dim=1) + 1e-6)
        return loss.mean()

    @staticmethod
    def _tv_loss(map_flat: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        b, n = map_flat.shape
        h, w = spatial_size
        if h * w != n:
            raise ValueError("spatial_size must match flattened map length")
        x = map_flat.view(b, 1, h, w)
        loss_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        loss_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return loss_h + loss_w

    @staticmethod
    def _prior_consistency_loss(saliency: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        s = saliency / (saliency.sum(dim=1, keepdim=True) + 1e-6)
        p = prior / (prior.sum(dim=1, keepdim=True) + 1e-6)
        return F.kl_div((s + 1e-6).log(), p.detach(), reduction="batchmean")

    def extract_features(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        g_feat, l_feat, i_feat, _ = self._forward_features(x, labels)
        return g_feat, l_feat, i_feat

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, labels: torch.Tensor | None = None) -> torch.Tensor:
        _, _, i_feat, _ = self._forward_features(x, labels)
        return F.normalize(i_feat, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.LongTensor | None = None,
        current_epoch: int = 0,
        total_epochs: int = 1,
        classifier: ClassifierMode = "arcface",
    ):
        if labels is not None and self.train_stage == "global":
            g_feat, _ = self._forward_global(x)
            logits_g = self._classification_logits(
                g_feat,
                labels,
                self.head_global,
                self.arc_margin_global,
                classifier,
                current_epoch,
                total_epochs,
            )
            loss_g = F.cross_entropy(logits_g, labels)
            return loss_g, logits_g, {"global": loss_g}

        g_feat, l_feat, i_feat, aux = self._forward_features(x, labels)
        b = x.size(0)

        if labels is None:
            return {
                "global_feat": F.normalize(g_feat, dim=1),
                "local_feats": F.normalize(l_feat, dim=1).view(b, min(self.k_local, aux["saliency"].size(1)), -1),
                "integrated_feat": F.normalize(i_feat, dim=1),
                "saliency": aux["saliency"],
                "local_saliency": aux["local_saliency"],
                "global_prior": aux["global_prior"],
                "local_energy": aux["local_energy"],
                "raw_cam": aux["raw_cam"],
                "raw_cdm": aux["raw_cdm"],
                "cam": aux["cam"],
                "cdm": aux["cdm"],
                "local_indices": aux["local_indices"],
                "token_attention": aux["token_attention"],
                "token_support": aux["token_support"],
            }

        logits_g = self._classification_logits(
            g_feat,
            labels,
            self.head_global,
            self.arc_margin_global,
            classifier,
            current_epoch,
            total_epochs,
        )
        labels_local = labels.unsqueeze(1).expand(-1, min(self.k_local, aux["saliency"].size(1))).reshape(-1)
        logits_l = self._classification_logits(
            l_feat,
            labels_local,
            self.head_local,
            self.arc_margin_local,
            classifier,
            current_epoch,
            total_epochs,
        )
        logits_i = self._classification_logits(
            i_feat,
            labels,
            self.head_integrated,
            self.arc_margin_integrated,
            classifier,
            current_epoch,
            total_epochs,
        )

        loss_g = F.cross_entropy(logits_g, labels)
        loss_l = F.cross_entropy(logits_l, labels_local)
        loss_i = F.cross_entropy(logits_i, labels)
        loss_div = self._diversity_loss(l_feat, b, min(self.k_local, aux["saliency"].size(1)))
        loss_local_bg = aux["local_bg_loss"]
        loss_cam_tv = aux["cam_tv_loss"]
        loss_cdm_tv = aux["cdm_tv_loss"]
        loss_map_prior = aux["map_prior_loss"]

        if self.train_stage == "local":
            return loss_l, logits_l, {"local": loss_l}
        if self.train_stage == "integrated":
            return loss_i, logits_i, {"integrated": loss_i}

        total = (
            loss_g
            + loss_l
            + loss_i
            + self.diversity_weight * loss_div
            + 1e-4 * (loss_cam_tv + loss_cdm_tv)
            + 0.001 * loss_map_prior
        )
        parts = {
            "global": loss_g,
            "local": loss_l,
            "integrated": loss_i,
            "diversity": loss_div,
            "cam_tv": loss_cam_tv,
            "cdm_tv": loss_cdm_tv,
            "map_prior": loss_map_prior,
        }
        if self.local_bg_loss_weight > 0:
            total = total + self.local_bg_loss_weight * loss_local_bg
            parts["local_bg"] = loss_local_bg
        return total, logits_i, parts

