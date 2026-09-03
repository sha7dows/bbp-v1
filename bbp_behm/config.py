"""用途：读取并校验 BBP-BeHM 的 INI 实验配置。"""

from __future__ import annotations

import configparser
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def _float_tuple(value: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _int_tuple(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    processed_path: Path
    processed_sha256: str
    expected_sessions: int
    expected_nodes: int
    expected_features: int


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float
    validation_fraction: float
    seeds: Tuple[int, ...]
    stratified: bool


@dataclass(frozen=True)
class HypergraphConfig:
    tau: float
    threshold: float
    semantic_gate: float
    time_weight: float
    time_window_seconds: float
    # 构建段冗余精简开关：none=原样构建；subset=剔除成员集合被其它同类型边严格包含的边；
    # jaccard=两两 Jaccard>=dedup_jaccard_tau 时保留 construction_strength 更高者。
    dedup_mode: str = "none"
    dedup_jaccard_tau: float = 0.90


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    dropout: float
    num_classes: int
    logit_adjustment: str
    louvain_seed: int
    positive_class_weight: float = 2.0
    confidence_weighting: str = "raw"
    hgnn_residual: bool = False
    edge_pooling: str = "sum"
    selector_dropout: Optional[float] = None
    ibg_weight_mode: str = "jaccard"
    ibg_hard_tau: Optional[float] = None


@dataclass(frozen=True)
class PruningConfig:
    enabled: bool
    retention_ratio: float
    candidate_ratios: Tuple[float, ...]
    gumbel_temperature: float
    type_embedding_dim: int
    use_ibg_centrality: bool = False
    budget_mode: str = "fixed"
    coverage_threshold: float = 0.90
    min_retention_ratio: float = 0.0
    max_retention_ratio: float = 1.0
    coverage_q: float = 0.20
    coverage_tail_gamma: float = 0.70
    coverage_eta: float = 0.25
    lambda_fidelity: float = 0.5
    lambda_comprehensiveness: float = 0.2
    comprehensiveness_margin: float = 0.1
    # 方向 A1'(09-03 提案):kept-set 互冗余可微正则强度;0=关闭(默认,行为与历史完全一致)。
    # >0 时训练损失 = 分类损失 + redundancy_lambda * R_keep(R_keep 定义见 model.py)。
    redundancy_lambda: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    gamma_curriculum_enabled: bool = False
    gamma_curriculum_start: float = 0.90
    gamma_curriculum_end_epoch: int = 8
    checkpoint_soup_enabled: bool = False
    checkpoint_soup_top_k: int = 3
    feature_noise_std: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    split: SplitConfig
    hypergraph: HypergraphConfig
    model: ModelConfig
    pruning: PruningConfig
    training: TrainingConfig

    def validate(self) -> None:
        if self.data.dataset.lower() not in {"vine", "ins"}:
            raise ValueError("The current implementation supports only dataset=vine or dataset=ins.")
        if not 0.0 < self.split.test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1).")
        if not 0.0 <= self.split.validation_fraction < 1.0 - self.split.test_fraction:
            raise ValueError("validation_fraction must leave non-empty training data.")
        if not 0.0 < self.pruning.retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1].")
        if self.pruning.budget_mode not in {"fixed", "adaptive", "robust_adaptive"}:
            raise ValueError("budget_mode must be fixed, adaptive, or robust_adaptive.")
        if not 0.0 < self.pruning.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in (0, 1].")
        if not 0.0 <= self.pruning.min_retention_ratio <= 1.0:
            raise ValueError("min_retention_ratio must be in [0, 1].")
        if not 0.0 < self.pruning.max_retention_ratio <= 1.0:
            raise ValueError("max_retention_ratio must be in (0, 1].")
        if self.pruning.min_retention_ratio > self.pruning.max_retention_ratio:
            raise ValueError("min_retention_ratio cannot exceed max_retention_ratio.")
        if not 0.0 < self.pruning.coverage_q <= 1.0:
            raise ValueError("coverage_q must be in (0, 1].")
        if not 0.0 < self.pruning.coverage_tail_gamma <= 1.0:
            raise ValueError("coverage_tail_gamma must be in (0, 1].")
        if not 0.0 <= self.pruning.coverage_eta <= 1.0:
            raise ValueError("coverage_eta must be in [0, 1].")
        if self.pruning.redundancy_lambda < 0.0:
            raise ValueError("redundancy_lambda must be non-negative.")
        if not 0.0 < self.training.gamma_curriculum_start <= 1.0:
            raise ValueError("gamma_curriculum_start must be in (0, 1].")
        if self.training.gamma_curriculum_end_epoch <= 0:
            raise ValueError("gamma_curriculum_end_epoch must be positive.")
        if self.training.gamma_curriculum_enabled:
            if self.pruning.budget_mode != "robust_adaptive":
                raise ValueError("gamma curriculum requires budget_mode=robust_adaptive.")
            if self.training.gamma_curriculum_start < self.pruning.coverage_tail_gamma:
                raise ValueError(
                    "gamma_curriculum_start cannot be below coverage_tail_gamma."
                )
            if self.training.gamma_curriculum_end_epoch > self.training.epochs:
                raise ValueError(
                    "gamma_curriculum_end_epoch cannot exceed training epochs."
                )
        if self.training.checkpoint_soup_top_k <= 0:
            raise ValueError("checkpoint_soup_top_k must be positive.")
        if self.training.checkpoint_soup_enabled:
            if self.training.checkpoint_soup_top_k < 2:
                raise ValueError("enabled checkpoint soup requires top_k >= 2.")
            if self.training.checkpoint_soup_top_k > self.training.epochs:
                raise ValueError("checkpoint_soup_top_k cannot exceed training epochs.")
        if not math.isfinite(self.training.feature_noise_std) or self.training.feature_noise_std < 0.0:
            raise ValueError("feature_noise_std must be finite and non-negative.")
        if self.hypergraph.tau <= 0.0 or self.hypergraph.time_window_seconds <= 0.0:
            raise ValueError("Hypergraph time parameters must be positive.")
        if self.hypergraph.dedup_mode not in {"none", "subset", "jaccard"}:
            raise ValueError("hypergraph.dedup_mode must be none, subset, or jaccard.")
        if not 0.0 < self.hypergraph.dedup_jaccard_tau <= 1.0:
            raise ValueError("hypergraph.dedup_jaccard_tau must be in (0, 1].")
        if self.model.logit_adjustment not in {"paper", "legacy", "none"}:
            raise ValueError("logit_adjustment must be paper, legacy, or none.")
        if self.model.edge_pooling not in {"sum", "mean"}:
            raise ValueError("edge_pooling must be sum or mean.")
        if self.model.ibg_weight_mode not in {"jaccard", "min_u", "prod_u", "softmin", "softprod", "hard_min"}:
            raise ValueError("ibg_weight_mode must be jaccard, min_u, prod_u, softmin, softprod, or hard_min.")
        if self.model.ibg_hard_tau is not None and not 0.0 < self.model.ibg_hard_tau <= 1.0:
            raise ValueError("ibg_hard_tau must be in (0, 1] when provided.")
        if self.model.ibg_weight_mode == "hard_min" and self.model.ibg_hard_tau is None:
            raise ValueError("hard_min ibg_weight_mode requires ibg_hard_tau.")
        if self.model.selector_dropout is not None and not 0.0 <= self.model.selector_dropout < 1.0:
            raise ValueError("selector_dropout must be in [0, 1).")
        if self.model.positive_class_weight <= 0.0:
            raise ValueError("positive_class_weight must be positive.")
        if self.model.confidence_weighting not in {"raw", "class_normalized", "none"}:
            raise ValueError("confidence_weighting must be raw, class_normalized, or none.")


def load_experiment_config(path: Path) -> ExperimentConfig:
    parser = configparser.ConfigParser()
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)

    data = DataConfig(
        dataset=parser.get("data", "dataset"),
        processed_path=Path(parser.get("data", "processed_path")),
        processed_sha256=parser.get("data", "processed_sha256").upper(),
        expected_sessions=parser.getint("data", "expected_sessions"),
        expected_nodes=parser.getint("data", "expected_nodes"),
        expected_features=parser.getint("data", "expected_features"),
    )
    split = SplitConfig(
        test_fraction=parser.getfloat("split", "test_fraction", fallback=0.20),
        validation_fraction=parser.getfloat("split", "validation_fraction", fallback=0.10),
        seeds=_int_tuple(parser.get("split", "seeds")),
        stratified=parser.getboolean("split", "stratified", fallback=True),
    )
    hypergraph = HypergraphConfig(
        tau=parser.getfloat("hypergraph", "tau"),
        threshold=parser.getfloat("hypergraph", "threshold"),
        semantic_gate=parser.getfloat("hypergraph", "semantic_gate"),
        time_weight=parser.getfloat("hypergraph", "time_weight"),
        time_window_seconds=parser.getfloat("hypergraph", "time_window_seconds"),
        dedup_mode=parser.get("hypergraph", "dedup_mode", fallback="none").strip().lower(),
        dedup_jaccard_tau=parser.getfloat("hypergraph", "dedup_jaccard_tau", fallback=0.90),
    )
    model_dropout = parser.getfloat("model", "dropout")
    model = ModelConfig(
        hidden_size=parser.getint("model", "hidden_size"),
        dropout=model_dropout,
        num_classes=parser.getint("model", "num_classes"),
        logit_adjustment=parser.get("model", "logit_adjustment"),
        louvain_seed=parser.getint("model", "louvain_seed"),
        positive_class_weight=parser.getfloat("model", "positive_class_weight", fallback=2.0),
        confidence_weighting=parser.get(
            "model", "confidence_weighting", fallback="raw"
        ).strip().lower(),
        hgnn_residual=parser.getboolean("model", "hgnn_residual", fallback=False),
        edge_pooling=parser.get("model", "edge_pooling", fallback="sum").strip().lower(),
        # 历史配置未声明时继续继承全局 dropout，保证训练行为向后兼容。
        selector_dropout=parser.getfloat(
            "model", "selector_dropout", fallback=model_dropout
        ),
        ibg_weight_mode=parser.get("model", "ibg_weight_mode", fallback="jaccard").strip().lower(),
        ibg_hard_tau=(
            parser.getfloat("model", "ibg_hard_tau")
            if parser.has_option("model", "ibg_hard_tau")
            else None
        ),
    )
    pruning = PruningConfig(
        enabled=parser.getboolean("pruning", "enabled"),
        retention_ratio=parser.getfloat("pruning", "retention_ratio"),
        candidate_ratios=_float_tuple(
            parser.get("pruning", "candidate_ratios", fallback=str(parser.getfloat("pruning", "retention_ratio")))
        ),
        gumbel_temperature=parser.getfloat("pruning", "gumbel_temperature"),
        type_embedding_dim=parser.getint("pruning", "type_embedding_dim"),
        use_ibg_centrality=parser.getboolean(
            "pruning", "use_ibg_centrality", fallback=False
        ),
        budget_mode=parser.get("pruning", "budget_mode", fallback="fixed").strip().lower(),
        coverage_threshold=parser.getfloat("pruning", "coverage_threshold", fallback=0.90),
        min_retention_ratio=parser.getfloat("pruning", "min_retention_ratio", fallback=0.0),
        max_retention_ratio=parser.getfloat("pruning", "max_retention_ratio", fallback=1.0),
        coverage_q=parser.getfloat("pruning", "coverage_q", fallback=0.20),
        coverage_tail_gamma=parser.getfloat(
            "pruning", "coverage_tail_gamma", fallback=0.70
        ),
        coverage_eta=parser.getfloat("pruning", "coverage_eta", fallback=0.25),
        lambda_fidelity=parser.getfloat("pruning", "lambda_fidelity", fallback=0.5),
        lambda_comprehensiveness=parser.getfloat("pruning", "lambda_comprehensiveness", fallback=0.2),
        comprehensiveness_margin=parser.getfloat("pruning", "comprehensiveness_margin", fallback=0.1),
        redundancy_lambda=parser.getfloat("pruning", "redundancy_lambda", fallback=0.0),
    )
    training = TrainingConfig(
        epochs=parser.getint("training", "epochs"),
        batch_size=parser.getint("training", "batch_size"),
        learning_rate=parser.getfloat("training", "learning_rate"),
        weight_decay=parser.getfloat("training", "weight_decay"),
        device=parser.get("training", "device"),
        gamma_curriculum_enabled=parser.getboolean(
            "training", "gamma_curriculum_enabled", fallback=False
        ),
        gamma_curriculum_start=parser.getfloat(
            "training", "gamma_curriculum_start", fallback=0.90
        ),
        gamma_curriculum_end_epoch=parser.getint(
            "training", "gamma_curriculum_end_epoch", fallback=8
        ),
        checkpoint_soup_enabled=parser.getboolean(
            "training", "checkpoint_soup_enabled", fallback=False
        ),
        checkpoint_soup_top_k=parser.getint(
            "training", "checkpoint_soup_top_k", fallback=3
        ),
        feature_noise_std=parser.getfloat("training", "feature_noise_std", fallback=0.0),
    )
    config = ExperimentConfig(data, split, hypergraph, model, pruning, training)
    config.validate()
    return config
