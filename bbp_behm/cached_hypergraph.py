"""用途：缓存无标签确定性会话超图并正确重组批记录，避免每轮重复构图。"""

from dataclasses import replace
import logging
import torch

from .data import collate_sessions
from .hypergraph import HyperedgeBatch, HyperedgeBuilder


class CachedHyperedgeBuilder:
    def __init__(self, cache):
        self.cache = cache

    @classmethod
    def prepare(cls, dataset, indices, config, device):
        builder = HyperedgeBuilder(config)
        cache = {}
        for offset,index in enumerate(indices):
            graph = builder.build_batch(collate_sessions([dataset[index]]).to(device))
            cache[int(index)] = replace(graph,incidence=graph.incidence.cpu(),edge_to_session=graph.edge_to_session.cpu(),
                                       type_ids=graph.type_ids.cpu(),structural_stats=graph.structural_stats.cpu())
            if offset % 100 == 0:
                logging.info("GRAPH_CACHE_PROGRESS sessions=%d",offset+1)
        return cls(cache)

    def build_batch(self,batch):
        graphs = [self.cache[int(i)] for i in batch.session_indices.cpu().tolist()]
        device = batch.x.device
        records = []
        for session,g in enumerate(graphs):
            for r in g.records:
                records.append(replace(r,session_position=session,hyperedge_id=len(records)))
        return HyperedgeBatch(torch.block_diag(*[g.incidence.to(device) for g in graphs]),
            torch.cat([torch.full((g.num_edges,),i,device=device,dtype=torch.long) for i,g in enumerate(graphs)]),
            torch.cat([g.type_ids.to(device) for g in graphs]),
            torch.cat([g.structural_stats.to(device) for g in graphs]),records)
