"""用途：按 BeHM 论文参数构造带行为类型与来源的候选超边及批 incidence。"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import HypergraphConfig
from .data import SessionBatch


class BehaviorType(IntEnum):
    STC = 0
    USER = 1
    MENTION = 2

    @property
    def label(self) -> str:
        return {self.STC: "STC", self.USER: "User", self.MENTION: "Mention"}[self]


@dataclass(frozen=True)
class HyperedgeRecord:
    hyperedge_id: int
    session_position: int
    session_id: str
    behavior_type: BehaviorType
    members: Tuple[int, ...]
    source_ids: Tuple[str, ...]
    construction_strength: float
    time_span_days: float
    unique_user_count: int
    duplicate_count: int = 1

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class HyperedgeBatch:
    incidence: Tensor
    edge_to_session: Tensor
    type_ids: Tensor
    structural_stats: Tensor
    records: List[HyperedgeRecord]

    @property
    def num_edges(self) -> int:
        return int(self.incidence.size(0))

    @property
    def num_nodes(self) -> int:
        return int(self.incidence.size(1))


@dataclass
class _Candidate:
    members: Tuple[int, ...]
    source_ids: List[str]
    construction_strengths: List[float]
    time_span_days: float
    unique_user_count: int


class HyperedgeBuilder:
    """构造 STC、User 与 Mention 三类整边，并在类型内部去除完全重复项。"""

    _mention_pattern = re.compile(r"@(user\d+)")

    def __init__(self, config: HypergraphConfig) -> None:
        self.config = config

    def _metadata_maps(self, metadata: Sequence[dict]) -> Tuple[Dict[int, dict], Dict[int, str]]:
        by_global = {int(item["global_node_idx"]): item for item in metadata}
        users = {global_id: str(item.get("user_id", "")) for global_id, item in by_global.items()}
        return by_global, users

    @staticmethod
    def _edge_time_span_days(members: Iterable[int], by_global: Dict[int, dict]) -> float:
        timestamps = [by_global[member]["create_time"].timestamp() for member in members]
        if len(timestamps) < 2:
            return 0.0
        return float((max(timestamps) - min(timestamps)) / 86400.0)

    @staticmethod
    def _unique_user_count(members: Iterable[int], users: Dict[int, str]) -> int:
        return len({users[member] for member in members if member in users})

    def _append_candidate(
        self,
        bucket: Dict[Tuple[int, ...], _Candidate],
        members: Iterable[int],
        source_id: str,
        strength: float,
        by_global: Dict[int, dict],
        users: Dict[int, str],
    ) -> None:
        key = tuple(sorted(set(int(member) for member in members)))
        if len(key) < 2:
            return
        if key not in bucket:
            bucket[key] = _Candidate(
                members=key,
                source_ids=[],
                construction_strengths=[],
                time_span_days=self._edge_time_span_days(key, by_global),
                unique_user_count=self._unique_user_count(key, users),
            )
        bucket[key].source_ids.append(str(source_id))
        bucket[key].construction_strengths.append(float(strength))

    def _build_stc(
        self,
        features: Tensor,
        metadata: Sequence[dict],
        by_global: Dict[int, dict],
        users: Dict[int, str],
    ) -> Dict[Tuple[int, ...], _Candidate]:
        bucket: Dict[Tuple[int, ...], _Candidate] = {}
        if len(metadata) < 2:
            return bucket
        global_ids = [int(item["global_node_idx"]) for item in metadata]
        timestamps = torch.tensor(
            [item["create_time"].timestamp() for item in metadata],
            dtype=features.dtype,
            device=features.device,
        )
        time_diff = (timestamps[:, None] - timestamps[None, :]).abs() / self.config.time_window_seconds
        time_probability = torch.exp(-time_diff / self.config.tau)
        normalized = F.normalize(features.float(), p=2, dim=1)
        semantic_similarity = normalized @ normalized.t()
        semantic_gate = (semantic_similarity > self.config.semantic_gate).to(features.dtype)
        fused = semantic_similarity * (1.0 + self.config.time_weight * semantic_gate * time_probability)

        for center in range(len(metadata)):
            member_positions = torch.nonzero(fused[:, center] >= self.config.threshold, as_tuple=False).flatten()
            if member_positions.numel() < 2:
                continue
            member_list = [global_ids[int(position)] for position in member_positions.tolist()]
            strength = float(fused[member_positions, center].mean().detach().cpu().item())
            self._append_candidate(
                bucket,
                member_list,
                source_id=f"center:{global_ids[center]}",
                strength=strength,
                by_global=by_global,
                users=users,
            )
        return bucket

    def _build_user(self, metadata: Sequence[dict], by_global: Dict[int, dict], users: Dict[int, str]) -> Dict[Tuple[int, ...], _Candidate]:
        bucket: Dict[Tuple[int, ...], _Candidate] = {}
        user_to_nodes: DefaultDict[str, set] = defaultdict(set)
        for item in metadata:
            user_to_nodes[str(item.get("user_id", ""))].add(int(item["global_node_idx"]))
        for user_id, members in user_to_nodes.items():
            self._append_candidate(bucket, members, f"user:{user_id}", 1.0, by_global, users)
        return bucket

    def _build_mention(self, metadata: Sequence[dict], by_global: Dict[int, dict], users: Dict[int, str]) -> Dict[Tuple[int, ...], _Candidate]:
        bucket: Dict[Tuple[int, ...], _Candidate] = {}
        mentioned_to_nodes: DefaultDict[str, set] = defaultdict(set)

        # 保持原 BeHM 的顺序依赖语义：节点提及目标用户，同时被提及用户自己的后续发言加入该边。
        for item in metadata:
            global_id = int(item["global_node_idx"])
            user_id = str(item.get("user_id", ""))
            mentioned_users = set(self._mention_pattern.findall(str(item.get("content", ""))))
            for mentioned_user in mentioned_users:
                mentioned_to_nodes[mentioned_user].add(global_id)
            if user_id in mentioned_to_nodes:
                mentioned_to_nodes[user_id].add(global_id)

        for mentioned_user, members in mentioned_to_nodes.items():
            self._append_candidate(bucket, members, f"mention:{mentioned_user}", 1.0, by_global, users)
        return bucket

    @staticmethod
    def _records_from_bucket(
        bucket: Dict[Tuple[int, ...], _Candidate],
        behavior_type: BehaviorType,
        session_position: int,
        session_id: str,
    ) -> List[HyperedgeRecord]:
        records: List[HyperedgeRecord] = []
        for candidate in bucket.values():
            records.append(
                HyperedgeRecord(
                    hyperedge_id=-1,
                    session_position=session_position,
                    session_id=session_id,
                    behavior_type=behavior_type,
                    members=candidate.members,
                    source_ids=tuple(candidate.source_ids),
                    construction_strength=float(sum(candidate.construction_strengths) / len(candidate.construction_strengths)),
                    time_span_days=candidate.time_span_days,
                    unique_user_count=candidate.unique_user_count,
                    duplicate_count=len(candidate.source_ids),
                )
            )
        return records

    def _dedup_stc_bucket(self, bucket: Dict[Tuple[int, ...], _Candidate]) -> Dict[Tuple[int, ...], _Candidate]:
        """按配置在构建段精简 STC 候选边，其余类型边不参与。

        subset：成员集合被其它 STC 边严格包含的边视为严格冗余，直接剔除（其信息被超集边完全覆盖）。
        jaccard：两两 Jaccard>=dedup_jaccard_tau 时保留 construction_strength 更高者（并列取尺寸更大者），
        被剔除边仍由保留边高重叠代表。
        """
        if self.config.dedup_mode == "none" or len(bucket) < 2:
            return bucket
        keys = list(bucket.keys())
        if self.config.dedup_mode == "subset":
            keep = [True] * len(keys)
            for i in range(len(keys)):
                for j in range(len(keys)):
                    if i == j:
                        continue
                    if len(keys[j]) > len(keys[i]) and set(keys[i]).issubset(keys[j]):
                        keep[i] = False
                        break
            return {key: bucket[key] for index, key in enumerate(keys) if keep[index]}
        if self.config.dedup_mode == "jaccard":
            tau = self.config.dedup_jaccard_tau
            keep = [True] * len(keys)
            for i in range(len(keys)):
                if not keep[i]:
                    continue
                set_i = set(keys[i])
                for j in range(i + 1, len(keys)):
                    if not keep[j]:
                        continue
                    set_j = set(keys[j])
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    if union == 0 or intersection / union < tau:
                        continue
                    strength_i = sum(bucket[keys[i]].construction_strengths) / len(bucket[keys[i]].construction_strengths)
                    strength_j = sum(bucket[keys[j]].construction_strengths) / len(bucket[keys[j]].construction_strengths)
                    if strength_i > strength_j or (strength_i == strength_j and len(keys[i]) >= len(keys[j])):
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
            return {key: bucket[key] for index, key in enumerate(keys) if keep[index]}
        raise ValueError(f"Unsupported dedup_mode: {self.config.dedup_mode}")

    def build_session(
        self,
        features: Tensor,
        metadata: Sequence[dict],
        session_position: int,
        session_id: str,
    ) -> List[HyperedgeRecord]:
        if len(metadata) < 2:
            return []
        if features.size(0) != len(metadata):
            raise ValueError("Session feature count does not match metadata length.")
        by_global, users = self._metadata_maps(metadata)
        stc_bucket = self._build_stc(features, metadata, by_global, users)
        if self.config.dedup_mode != "none":
            # 仅在构建段精简 STC 候选边；User/Mention 无内部冗余且须保持不变。
            stc_bucket = self._dedup_stc_bucket(stc_bucket)
        typed_buckets = (
            (BehaviorType.STC, stc_bucket),
            (BehaviorType.MENTION, self._build_mention(metadata, by_global, users)),
            (BehaviorType.USER, self._build_user(metadata, by_global, users)),
        )
        records: List[HyperedgeRecord] = []
        for behavior_type, bucket in typed_buckets:
            records.extend(self._records_from_bucket(bucket, behavior_type, session_position, session_id))
        return records

    def build_batch(self, batch: SessionBatch) -> HyperedgeBatch:
        records: List[HyperedgeRecord] = []
        global_to_batch = {
            int(global_id): position for position, global_id in enumerate(batch.global_node_indices.detach().cpu().tolist())
        }
        for session_position in range(batch.num_graphs):
            start = int(batch.ptr[session_position].item())
            end = int(batch.ptr[session_position + 1].item())
            session_records = self.build_session(
                batch.x[start:end],
                batch.node_metadata[session_position],
                session_position,
                batch.session_ids[session_position],
            )
            for record in session_records:
                records.append(replace(record, hyperedge_id=len(records)))

        incidence = torch.zeros(
            (len(records), batch.x.size(0)), dtype=batch.x.dtype, device=batch.x.device
        )
        for record in records:
            for global_id in record.members:
                if global_id not in global_to_batch:
                    raise ValueError(f"Hyperedge member {global_id} is not in the current batch.")
                incidence[record.hyperedge_id, global_to_batch[global_id]] = 1.0

        edge_to_session = torch.tensor(
            [record.session_position for record in records], dtype=torch.long, device=batch.x.device
        )
        type_ids = torch.tensor(
            [int(record.behavior_type) for record in records], dtype=torch.long, device=batch.x.device
        )
        stats = torch.tensor(
            [
                [
                    math.log1p(record.size),
                    record.construction_strength,
                    math.log1p(record.time_span_days),
                    math.log1p(record.unique_user_count),
                ]
                for record in records
            ],
            dtype=batch.x.dtype,
            device=batch.x.device,
        )
        if not records:
            stats = torch.empty((0, 4), dtype=batch.x.dtype, device=batch.x.device)
        return HyperedgeBatch(incidence, edge_to_session, type_ids, stats, records)
