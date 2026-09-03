"""用途：只读加载 Vine 预处理会话，校验快照并生成无测试泄漏的分层划分。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SessionSample:
    session_index: int
    session_id: str
    x: Tensor
    global_node_indices: Tensor
    node_metadata: List[dict]
    label: int
    confidence: float


@dataclass
class SessionBatch:
    x: Tensor
    global_node_indices: Tensor
    graph_ids: Tensor
    ptr: Tensor
    node_metadata: List[List[dict]]
    labels: Tensor
    confidences: Tensor
    session_ids: List[str]
    session_indices: Tensor

    @property
    def num_graphs(self) -> int:
        return len(self.session_ids)

    def to(self, device: torch.device) -> "SessionBatch":
        return SessionBatch(
            x=self.x.to(device),
            global_node_indices=self.global_node_indices.to(device),
            graph_ids=self.graph_ids.to(device),
            ptr=self.ptr.to(device),
            node_metadata=self.node_metadata,
            labels=self.labels.to(device),
            confidences=self.confidences.to(device),
            session_ids=self.session_ids,
            session_indices=self.session_indices.to(device),
        )


@dataclass(frozen=True)
class SplitIndices:
    train: Tuple[int, ...]
    validation: Tuple[int, ...]
    test: Tuple[int, ...]
    seed: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


class PreprocessedSessionDataset(Dataset[SessionSample]):
    """直接切片 PyG 的 collated 张量，不加载 BERT/RoBERTa。"""

    def __init__(
        self,
        processed_path: Path,
        expected_sha256: Optional[str] = None,
        expected_sessions: Optional[int] = None,
        expected_nodes: Optional[int] = None,
        expected_features: Optional[int] = None,
        verify_hash: bool = True,
    ) -> None:
        self.processed_path = Path(processed_path)
        if not self.processed_path.is_file():
            raise FileNotFoundError(f"Processed dataset not found: {self.processed_path}")
        if verify_hash and expected_sha256:
            actual = sha256_file(self.processed_path)
            if actual != expected_sha256.upper():
                raise ValueError(f"Dataset SHA256 mismatch: expected={expected_sha256}, actual={actual}")

        # 该文件属于用户提供的可信研究资产；显式保持 CPU 加载。
        self.data, self.slices = torch.load(self.processed_path, map_location="cpu", weights_only=False)
        self._num_sessions = int(self.slices["x"].numel() - 1)
        self._has_confidence = hasattr(self.data, "confidence") and "confidence" in self.slices
        self._validate_snapshot(expected_sessions, expected_nodes, expected_features)
        self._labels = torch.tensor([self._session_label(i) for i in range(self._num_sessions)], dtype=torch.long)

    def _validate_snapshot(
        self,
        expected_sessions: Optional[int],
        expected_nodes: Optional[int],
        expected_features: Optional[int],
    ) -> None:
        if expected_sessions is not None and self._num_sessions != expected_sessions:
            raise ValueError(f"Session count mismatch: expected={expected_sessions}, actual={self._num_sessions}")
        if expected_nodes is not None and int(self.data.x.size(0)) != expected_nodes:
            raise ValueError(f"Node count mismatch: expected={expected_nodes}, actual={self.data.x.size(0)}")
        if expected_features is not None and int(self.data.x.size(1)) != expected_features:
            raise ValueError(f"Feature count mismatch: expected={expected_features}, actual={self.data.x.size(1)}")
        if len(self.data.node_metadata) != self._num_sessions:
            raise ValueError("node_metadata length does not match session count.")
        if self._has_confidence:
            confidence = self.data.confidence
            if not bool(torch.isfinite(confidence).all()):
                raise ValueError("confidence contains non-finite values.")
            if bool((confidence < 0.0).any()) or bool((confidence > 1.0).any()):
                raise ValueError("confidence must be in [0, 1].")

    def _bounds(self, key: str, index: int) -> Tuple[int, int]:
        values = self.slices[key]
        return int(values[index]), int(values[index + 1])

    def _session_label(self, index: int) -> int:
        start, _ = self._bounds("y", index)
        return int(self.data.y[start].item())

    @property
    def labels(self) -> Tensor:
        return self._labels.clone()

    @property
    def has_confidence(self) -> bool:
        """指示快照是否提供可用于加权分类损失的置信度。"""

        return self._has_confidence

    @property
    def session_confidences(self) -> Tensor:
        """返回每个会话的单一置信度；缺失时显式回退为 1。"""

        if not self._has_confidence:
            return torch.ones(self._num_sessions, dtype=torch.float32)
        values = []
        for index in range(self._num_sessions):
            start, _ = self._bounds("confidence", index)
            values.append(float(self.data.confidence[start].item()))
        return torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return self._num_sessions

    def __getitem__(self, index: int) -> SessionSample:
        if index < 0:
            index += self._num_sessions
        if index < 0 or index >= self._num_sessions:
            raise IndexError(index)

        x_start, x_end = self._bounds("x", index)
        g_start, g_end = self._bounds("global_node_indices", index)
        metadata = list(self.data.node_metadata[index])
        x = self.data.x[x_start:x_end]
        global_ids = self.data.global_node_indices[g_start:g_end].long()
        if x.size(0) != global_ids.numel() or x.size(0) != len(metadata):
            raise ValueError(f"Session {index} has inconsistent node-aligned fields.")

        confidence = 1.0
        if self._has_confidence:
            c_start, _ = self._bounds("confidence", index)
            confidence = float(self.data.confidence[c_start].item())

        session_id = str(metadata[0].get("session_id", index)) if metadata else str(index)
        return SessionSample(
            session_index=index,
            session_id=session_id,
            x=x,
            global_node_indices=global_ids,
            node_metadata=metadata,
            label=self._session_label(index),
            confidence=confidence,
        )

    def make_split(
        self,
        seed: int,
        test_fraction: float = 0.20,
        validation_fraction: float = 0.10,
    ) -> SplitIndices:
        indices = np.arange(self._num_sessions)
        labels = self._labels.numpy()
        outer = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_val_pos, test_pos = next(outer.split(indices, labels))
        train_val = indices[train_val_pos]
        test = indices[test_pos]

        if validation_fraction <= 0.0:
            train = train_val
            validation = np.array([], dtype=np.int64)
        else:
            inner_fraction = validation_fraction / (1.0 - test_fraction)
            inner = StratifiedShuffleSplit(n_splits=1, test_size=inner_fraction, random_state=seed + 100_003)
            train_pos, validation_pos = next(inner.split(train_val, labels[train_val]))
            train = train_val[train_pos]
            validation = train_val[validation_pos]

        return SplitIndices(
            train=tuple(int(i) for i in train),
            validation=tuple(int(i) for i in validation),
            test=tuple(int(i) for i in test),
            seed=seed,
        )


def collate_sessions(samples: Sequence[SessionSample]) -> SessionBatch:
    if not samples:
        raise ValueError("Cannot collate an empty session batch.")
    lengths = [int(sample.x.size(0)) for sample in samples]
    ptr = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.long)
    graph_ids = torch.repeat_interleave(torch.arange(len(samples), dtype=torch.long), torch.tensor(lengths))
    return SessionBatch(
        x=torch.cat([sample.x for sample in samples], dim=0),
        global_node_indices=torch.cat([sample.global_node_indices for sample in samples], dim=0),
        graph_ids=graph_ids,
        ptr=ptr,
        node_metadata=[sample.node_metadata for sample in samples],
        labels=torch.tensor([sample.label for sample in samples], dtype=torch.long),
        confidences=torch.tensor([sample.confidence for sample in samples], dtype=torch.float32),
        session_ids=[sample.session_id for sample in samples],
        session_indices=torch.tensor([sample.session_index for sample in samples], dtype=torch.long),
    )


class IndexedSessionDataset(Dataset[SessionSample]):
    """在不复制样本的前提下暴露给定会话索引子集。"""

    def __init__(self, dataset: PreprocessedSessionDataset, indices: Iterable[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> SessionSample:
        return self.dataset[self.indices[index]]
