# -*- coding: utf-8 -*-
"""生成 docs 中的架构图（论文 BeHM 三段式 + 仓库新增 BBP 选择器）。

仅依赖 Pillow。运行：python docs/figs/make_figs.py
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 字体 / 颜色
FONT_FILES = {
    "zh": (r"C:\Windows\Fonts\msyh.ttc", 0),
    "zhb": (r"C:\Windows\Fonts\msyhbd.ttc", 0),
    "mono": (r"C:\Windows\Fonts\consola.ttf", 0),
}
_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _FONTS:
        path, index = FONT_FILES[kind]
        _FONTS[key] = ImageFont.truetype(path, size, index=index)
    return _FONTS[key]


INK = (32, 33, 36)
INK2 = (95, 99, 104)
INK3 = (128, 134, 139)
BLUE_F, BLUE_B, BLUE_T = (232, 240, 254), (26, 115, 232), (23, 78, 166)
ORANGE_F, ORANGE_B, ORANGE_T = (254, 243, 224), (232, 113, 10), (180, 83, 9)
PURPLE_F, PURPLE_B, PURPLE_T = (243, 232, 253), (123, 31, 162), (74, 20, 140)
GREEN_F, GREEN_B, GREEN_T = (230, 244, 234), (19, 115, 51), (12, 85, 34)
GRAY_F, GRAY_B, GRAY_T = (241, 243, 244), (154, 160, 166), (95, 99, 104)

KIND = {
    "paper": (BLUE_F, BLUE_B, BLUE_T),
    "new": (ORANGE_F, ORANGE_B, ORANGE_T),
    "loss": (PURPLE_F, PURPLE_B, PURPLE_T),
    "data": (GREEN_F, GREEN_B, GREEN_T),
    "proto": (GRAY_F, GRAY_B, GRAY_T),
}


class Fig:
    def __init__(self, width: int, height: int, title: str, subtitle: str = "") -> None:
        self.w, self.h = width, height
        self.img = Image.new("RGB", (width, height), "white")
        self.d = ImageDraw.Draw(self.img)
        self.text(56, 22, title, "zhb", 29, INK)
        if subtitle:
            self.text(56, 66, subtitle, "zh", 16, INK2)

    # -------------------------------------------------------------- 基本图元
    def text(self, x, y, s, kind="zh", size=18, fill=INK, anchor="lt"):
        self.d.text((x, y), s, font=font(kind, size), fill=fill, anchor=anchor)

    def wrap(self, s: str, kind: str, size: int, maxw: int) -> list[str]:
        f = font(kind, size)
        lines, cur = [], ""
        for ch in s:
            if self.d.textlength(cur + ch, font=f) <= maxw:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    def card(
        self,
        x0,
        y0,
        x1,
        y1,
        kind="paper",
        title="",
        lines=(),
        tag="",
        code="",
        title_size=20,
        radius=14,
    ):
        fill, border, title_color = KIND[kind]
        self.d.rounded_rectangle(
            [x0, y0, x1, y1], radius=radius, fill=fill, outline=border, width=2
        )
        inner_w = x1 - x0 - 32
        if title:
            self.text(x0 + 16, y0 + 11, title, "zhb", title_size, title_color)
        if tag:
            tw = self.d.textlength(tag, font=font("zh", 13))
            self.d.rounded_rectangle(
                [x1 - tw - 26, y0 + 11, x1 - 12, y0 + 33],
                radius=9,
                fill="white",
                outline=border,
                width=1,
            )
            self.text(x1 - tw - 19, y0 + 13, tag, "zh", 13, title_color)
        y = y0 + (40 if title else 14)
        for item in lines:
            if isinstance(item, tuple):
                style, s = item
            else:
                style, s = "txt", item
            if style == "eq":
                f, size, fillc, lh = "zh", 17, (28, 30, 34), 25
            elif style == "note":
                f, size, fillc, lh = "zh", 14, INK2, 22
            elif style == "code":
                if CJK.search(s):
                    f, size, fillc, lh = "zh", 14, (85, 90, 96), 22
                else:
                    f, size, fillc, lh = "mono", 13, (85, 90, 96), 21
            elif style == "hi":
                f, size, fillc, lh = "zhb", 16, title_color, 24
            else:
                f, size, fillc, lh = "zh", 16, INK, 25
            for piece in self.wrap(s, f, size, inner_w):
                self.text(x0 + 16, y, piece, f, size, fillc)
                y += lh
        if code:
            self.text(x0 + 16, y1 - 23, code, "zh" if CJK.search(code) else "mono", 13, INK3)
        limit = y1 - (26 if code else 6)
        if y > limit:
            print(f"  ! overflow: {title or 'card'} needs {y - limit}px more")

    def band_rail(self, y0, y1, number, label, color):
        cx = 36
        self.d.rounded_rectangle([18, y0, 54, y0 + 30], radius=8, fill=color)
        self.text(cx, y0 + 5, number, "zh", 17, "white", anchor="mt")
        yy = y0 + 42
        for ch in label:
            self.text(cx, yy, ch, "zhb", 19, color, anchor="mt")
            yy += 23

    # -------------------------------------------------------------- 线 / 箭头
    def _seg(self, p0, p1, color, width, dashed):
        if not dashed:
            self.d.line([p0, p1], fill=color, width=width)
            return
        (x0, y0), (x1, y1) = p0, p1
        length = max(abs(x1 - x0), abs(y1 - y0))
        if length == 0:
            return
        step, on = 9, 6
        pos = 0
        while pos < length:
            t0 = pos / length
            t1 = min((pos + on) / length, 1.0)
            self.d.line(
                [
                    (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                    (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1),
                ],
                fill=color,
                width=width,
            )
            pos += step

    def arrow(self, pts, color=INK2, width=3, dashed=False, head=12, joints=True):
        for a, b in zip(pts[:-1], pts[1:]):
            self._seg(a, b, color, width, dashed)
        if joints:
            for p in pts[1:-1]:
                self.d.ellipse(
                    [p[0] - width / 2, p[1] - width / 2, p[0] + width / 2, p[1] + width / 2],
                    fill=color,
                )
        if head <= 0:
            return
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        if x1 == x0:
            s = 1 if y1 > y0 else -1
            tri = [(x1, y1), (x1 - head * 0.62, y1 - s * head), (x1 + head * 0.62, y1 - s * head)]
        else:
            s = 1 if x1 > x0 else -1
            tri = [(x1, y1), (x1 - s * head, y1 - head * 0.62), (x1 - s * head, y1 + head * 0.62)]
        self.d.polygon(tri, fill=color)

    def note(self, x, y, s, color=INK2, size=15, bold=False, maxw=None):
        kind = "zhb" if bold else "zh"
        if maxw is None:
            self.text(x, y, s, kind, size, color)
            return y + size + 6
        for piece in self.wrap(s, kind, size, maxw):
            self.text(x, y, piece, kind, size, color)
            y += size + 7
        return y

    def save(self, name):
        path = HERE / name
        self.img.save(path, optimize=True)
        print("saved", path, self.img.size)


# ===================================================================== 图 1
def fig_overview():
    f = Fig(
        1800,
        1610,
        "图 1  BBP-BeHM 代码架构总览（对齐论文 Fig.2 的三段式，并标出仓库新增部分）",
        "蓝 = 论文 §IV 模块 ｜ 橙 = 仓库新增的 BBP 预算选择（论文未涉及）｜ 紫 = 损失 ｜ 绿 = 数据 / 产物",
    )
    rail = 78

    # ---------------- ① 输入与编码
    y0, y1 = 150, 300
    f.band_rail(y0, y1, "1", "输入与编码", BLUE_B)
    f.card(
        rail, y0, 700, y1, "data",
        "会话 S_i = {P_i, C_i, U_i, Y_i}",
        [
            ("txt", "原始 post + 按时间排序的评论 → 节点序列 N = {n_1..n_k}"),
            ("txt", "每节点带 user_id / create_time / content / global_node_idx（构图必需）"),
            ("note", "标签 y_i ∈ {0,1}；标注者一致度 c_i ∈ [0,1] 作为样本置信度"),
        ],
        code="bbp_behm/data.py · PreprocessedSessionDataset",
    )
    f.arrow([(700, 225), (738, 225)])
    f.card(
        740, y0, 1740, y1, "paper",
        "PLM 节点编码   Eq.1: X_i = M(n_i),  d = 768",
        [
            ("txt", "BERT 编码在数据预处理阶段离线完成，训练图里不出现 PLM —— 快照只存 768 维特征"),
            ("txt", "缓存：KbertConfidence_data_bert.pt（970 会话 / 68662 节点 / 768 维，带 SHA256 校验）"),
            ("note", "因此论文 Fig.2 的 “PLM Encoder” 在本仓库对应 data 层的一次性预处理，而非可训练模块"),
        ],
        tag="论文 §IV-A",
        code="data.py · PreprocessedSessionDataset.__init__  (torch.load + sha256)",
    )

    # ---------------- ② 超图构建
    y0 = 370
    f.band_rail(y0, 600, "2", "超图构建", BLUE_B)
    f.card(
        rail, y0, 610, 540, "paper",
        "STC 语义-时间协同行为",
        [
            ("eq", "F_ij = s_ij · (1 + β·G(s_ij)·exp(−T_ij/τ)) ≥ θ"),
            ("note", "s_ij = cos 相似度；G 为 s_ij > η 的 0/1 门；T_ij 以天为单位"),
            ("note", "θ=0.60  β=0.50  τ=0.10  η=0.55（Vine 归档值）"),
        ],
        tag="Eq.2",
        code="hypergraph.py · _build_stc",
        title_size=19,
    )
    f.card(
        630, y0, 950, 540, "paper",
        "User 用户一致性行为",
        [
            ("eq", "E_u = { n ∈ N | U(n) = u }"),
            ("note", "同一 user_id 的全部节点并成一条边"),
        ],
        tag="Eq.3",
        code="hypergraph.py · _build_user",
        title_size=19,
    )
    f.card(
        970, y0, 1290, 540, "paper",
        "Mention 提及交互行为",
        [
            ("eq", "E_u = {n | u ∈ Ment(n)}"),
            ("eq", "        ∪ {n | U(n) = u}"),
            ("note", "代码保留 BeHM 原始“顺序依赖”语义（见 §8 偏差 2）"),
        ],
        tag="Eq.4",
        code="hypergraph.py · _build_mention",
        title_size=19,
    )
    for cx in (350, 790, 1130):
        f.arrow([(cx, 540), (cx, 556)])
    f.d.rounded_rectangle([rail, 556, 1740, 600], radius=12, fill=GRAY_F, outline=GRAY_B, width=2)
    f.text(
        rail + 16, 566,
        "会话超图 H = {N, E_H}：incidence[|E_H|, |N|] ｜ edge_to_session ｜ type_ids ｜ 4 维结构统计 z_e ｜ dedup=subset 只精简 STC",
        "zh", 15, INK,
    )

    # ---------------- ③ BBP 选择
    y0, y1 = 670, 920
    f.band_rail(y0, y1, "3", "BBP 选择", ORANGE_B)
    f.card(
        rail, y0, 700, y1, "new",
        "① 打分器  EdgeSelector.score_edges",
        [
            ("txt", "边表征 ē = mean(成员节点的投影 p_i)，p = Linear(768→128)"),
            ("txt", "+ 类型嵌入 t_e(16) + LayerNorm(结构统计 4) [+ 可选 IBG 中心性]"),
            ("txt", "shared_scorer：MLP(148→128→1)，给全局显著性"),
            ("txt", "session_scorer：拼上会话上下文 c_s 再给会话内修正分"),
            ("eq", "s_e = s_shared(e) + s_session(e)"),
            ("note", "训练期仅 fixed / adaptive 加 Gumbel 噪声；robust_adaptive 不加"),
        ],
        tag="新增",
        code="selector.py · score_edges",
        title_size=19,
    )
    f.card(
        720, y0, 1330, y1, "new",
        "② 预算与覆盖约束（三选一）",
        [
            ("hi", "fixed：k_s = max(1, ceil(ρ·E_s))，按 s_e 精确 top-k"),
            ("hi", "adaptive：分数排序取前缀，首次达分类型覆盖阈值即停"),
            ("hi", "robust_adaptive：每步取 max [η·task + (1−η)·coverage_gain]"),
            ("txt", "停止条件：各类型“最低 q 比例尾部”覆盖均值的最小值 ≥ γ"),
            ("txt", "K 被夹在 [min_retention, max_retention]·E_s 之间"),
            ("note", "归档配置用 robust_adaptive：γ=0.85, q=0.20, η=0.25, T=1.0"),
        ],
        tag="新增",
        code="selector.py · select / select_adaptive / select_robust_adaptive",
        title_size=19,
    )
    f.card(
        1350, y0, 1740, y1, "new",
        "③ 掩码与梯度",
        [
            ("txt", "hard mask m ∈ {0,1}：决定前向拓扑"),
            ("txt", "soft σ((s−τ*)/T)，二分 τ* 使 Σsoft = K"),
            ("eq", "ST = m + (soft − m).detach()"),
            ("note", "前向值恒等于 m；梯度只经 soft 回传到打分器"),
            ("note", "m 同时决定 HGNN 边池化与 IBG 中哪些行为存在"),
        ],
        tag="新增",
        code="model.py · BBPBeHM.forward",
        title_size=19,
    )

    # ---------------- ④ 高阶交互建模
    y0, y1 = 990, 1200
    f.band_rail(y0, y1, "4", "高阶交互", BLUE_B)
    f.card(
        rail, y0, 500, y1, "paper",
        "HGNN 2 层",
        [
            ("txt", "HConv1 → ReLU → Dropout → HConv2"),
            ("eq", "Â = D_v^(-1/2) · H · D_e^(-1) · H^T · D_v^(-1/2)"),
            ("eq", "X' = Â · X · W"),
            ("note", "hgnn_residual 可选（归档关）；edge_pooling=sum"),
        ],
        tag="§IV-B",
        code="backbone.py · HGNN",
        title_size=19,
    )
    f.arrow([(500, 1095), (518, 1095)])
    f.card(
        520, y0, 900, y1, "paper",
        "IBG 交互行为图",
        [
            ("txt", "每条 kept 超边 = 一个“行为节点”"),
            ("eq", "W_ij = |V_i ∩ V_j| / |V_i ∪ V_j|"),
            ("note", "只对 kept 边建图；Jaccard>0 才连边；会话间独立"),
            ("note", "仓库扩展：min_u / prod_u / softmin / hard_min 变体"),
        ],
        tag="Eq.5",
        code="backbone.py · build_ibg",
        title_size=19,
    )
    f.arrow([(900, 1095), (918, 1095)])
    f.card(
        920, y0, 1260, y1, "paper",
        "行为关系传播",
        [
            ("txt", "在 IBG 上做一层图传播，边权 = W_ij"),
            ("txt", "代码：GCNConv(128→128)"),
            ("note", "论文写的是 GAT —— 实现层偏差（见 §8）"),
        ],
        tag="§IV-B",
        code="backbone.py · hyperedge_gnn",
        title_size=19,
    )
    f.arrow([(1260, 1095), (1278, 1095)])
    f.card(
        1280, y0, 1740, y1, "paper",
        "社区发现",
        [
            ("txt", "加权 Louvain（random_state = louvain_seed = 42）"),
            ("txt", "社区内对 X'_E 做 mean pooling"),
            ("eq", "C_comm = { mean(X'_E[e]), e ∈ community }"),
            ("note", "IBG 无边时每个行为自成一社区，信息不丢"),
        ],
        tag="§IV-B",
        code="backbone.py · CommunityFeatureExtractor",
        title_size=19,
    )

    # ---------------- ⑤ 多维融合与分类
    y0, y1 = 1270, 1520
    f.band_rail(y0, y1, "5", "融合与分类", BLUE_B)
    f.card(
        rail, y0, 700, y1, "paper",
        "多维注意力融合",
        [
            ("eq", "C_G = AttentionPooling(C_comm)   ← 全局协同模式"),
            ("eq", "X_L = [X_N ; X_E]"),
            ("eq", "C_L = softmax(C_G·X_L^T / √d_k) · X_L"),
            ("eq", "C_final = C_G + Linear(C_L)"),
            ("note", "代码加了三组 QKV 线性投影、LayerNorm 与 Dropout（Eq.6/7 的工程化版本）"),
        ],
        tag="§IV-C",
        code="backbone.py · ContextAwareHierarchicalFusion",
        title_size=19,
    )
    f.arrow([(700, 1395), (718, 1395)])
    f.card(
        720, y0, 1200, y1, "paper",
        "会话分类器 + logit 调整",
        [
            ("eq", "logits = f(C_final) − δ·log π"),
            ("note", "π = 该折训练集类频率；δ = softplus(a) 可学习，初值 0.5413"),
            ("txt", "f(·) = Linear(128→2)"),
            ("note", "Eq.8 的 Menon 长尾补偿，用于抵消类别不均衡"),
        ],
        tag="Eq.8",
        code="backbone.py · _adjust_logits",
        title_size=19,
    )
    f.arrow([(1200, 1395), (1218, 1395)])
    f.card(
        1220, y0, 1740, y1, "loss",
        "置信度加权交叉熵",
        [
            ("eq", "L = (1/B) Σ_i c_i · CE(logits_i, y_i)"),
            ("txt", "CE 内含类权 [1, 1.5]（仓库新增）；c_i 为标注者一致度"),
            ("txt", "训练：Adam，lr=1e-4，wd=5e-3，20 epoch，bs=64"),
            ("note", "无 scheduler / 无梯度裁剪 / 无早停；模型选择在协议层做（图 4）"),
        ],
        tag="Eq.9",
        code="backbone.py · forward_projected",
        title_size=19,
    )

    # 跨带连线
    f.arrow([(1220, 302), (1220, 332), (350, 332)], INK2, 3, head=0)
    for cx in (350, 790, 1130):
        f.arrow([(cx, 332), (cx, 368)], INK2, 3)
    f.text(760, 304, "节点特征 X_i + 元数据 → 三类超边", "zh", 15, INK2)
    f.arrow([(980, 602), (980, 640), (400, 640), (400, 668)], INK2, 3)
    f.text(700, 644, "H → 选择器", "zh", 15, INK2)
    f.arrow([(1545, 922), (1545, 958), (270, 958), (270, 988)], ORANGE_B, 3)
    f.text(1060, 930, "mask m（硬掩码决定前向拓扑）", "zhb", 15, ORANGE_T)
    f.arrow([(270, 1202), (270, 1232), (150, 1232), (150, 1266)], BLUE_B, 3, dashed=True)
    f.text(84, 1204, "X_N, X_E", "zh", 14, BLUE_T)
    f.arrow([(1480, 1202), (1480, 1232), (520, 1232), (520, 1266)], BLUE_B, 3)
    f.text(760, 1204, "C_comm", "zh", 14, BLUE_T)

    f.text(
        56, 1546,
        "图 1｜一个会话的完整数据流。②④⑤ 与论文 Fig.2(a) 的三段式一一对应；③ 是仓库相对论文新增的 BBP 预算选择层，"
        "它把“选哪些超边”变成可学习、可约束、可解释的显式决策。",
        "zh", 16, INK2,
    )
    f.text(
        56, 1574,
        "代码入口：scripts/train_vine_cv5.py（训练与选型）｜ bbp_behm/model.py::BBPBeHM.forward（③→④→⑤ 的组装处）",
        "zh", 14, INK3,
    )
    f.save("fig1_overview.png")


# ===================================================================== 图 2
def fig_selector():
    f = Fig(
        1800,
        1070,
        "图 2  BBP 预算选择器详解（仓库新增；论文中没有这一层）",
        "目的：把“哪些超边构成行为基”从隐式注意力变成显式的、带覆盖保证的预算决策；分类梯度只经 straight-through 通道回传。",
    )

    # ---------------- ① 输入与打分
    f.card(56, 110, 700, 520, "new", "① 输入与打分（可学习部分）", [], tag="score_edges", title_size=20)
    f.text(72, 150, "来源：H 的 incidence / type_ids / 4 维统计 + 投影节点 p = Linear(768→128)", "zh", 14, INK2)
    pills = [
        (176, "边表征   ē = (1/|e|) · Σ p_i"),
        (218, "类型嵌入 16 + 结构统计 4 → LayerNorm → 148 维"),
        (260, "会话上下文   c_s = mean(p_i),  i ∈ s"),
        (302, "可选：IBG 加权度中心性（use_ibg_centrality，默认关）"),
    ]
    for top, s in pills:
        f.d.rounded_rectangle([72, top, 684, top + 34], radius=9, fill="white", outline=ORANGE_B, width=1)
        f.text(84, top + 7, s, "zh", 15, INK)
    f.arrow([(378, 336), (378, 352)], INK2, 3)
    f.card(72, 352, 684, 458, "proto", "两路打分器", [
        ("eq", "shared_scorer：MLP(148→128→1)   全局显著性"),
        ("eq", "session_scorer：MLP(148+128→128→1)   会话内修正"),
    ], title_size=16)
    f.text(72, 470, "s_e = s_shared(e) + s_session(e)", "zhb", 18, ORANGE_T)
    f.text(72, 496, "→ 后续所有预算/覆盖决策只看这个标量", "zh", 14, INK2)

    # ---------------- ② 三种预算
    f.card(736, 110, 1284, 520, "new", "② 预算与覆盖约束（三选一）", [], tag="select*", title_size=20)
    f.card(750, 152, 1270, 248, "proto", "fixed：固定保留率 ρ", [
        ("eq", "k_s = max(1, ceil(ρ·E_s))，按 s_e 取 top-k"),
        ("note", "soft = σ(s_e / T)；训练期加 Gumbel 扰动"),
    ], title_size=16)
    f.card(750, 258, 1270, 356, "proto", "adaptive：按覆盖阈值停", [
        ("eq", "按 s_e 排序取前缀，首次达分类型覆盖 ≥ θ_cov 即停"),
        ("note", "K ∈ [min_r·E_s, max_r·E_s]；不含尾部鲁棒性"),
    ], title_size=16)
    f.card(750, 366, 1270, 500, "new", "robust_adaptive：联合贪心（归档配置）", [
        ("eq", "u(e) = η·norm(s_e) + (1−η)·norm(Δcover(e))"),
        ("eq", "stop: min_t tail_q(best_t) ≥ γ"),
        ("note", "γ=0.85, q=0.20, η=0.25, min_r=0, max_r=1, T=1.0"),
    ], title_size=16)

    # ---------------- ③ 伪代码
    f.card(1320, 110, 1744, 520, "new", "③ 每会话的确定性贪心", [], tag="伪代码", title_size=20)
    code_lines = [
        "for s in sessions:",
        "  best[t] = 0 for all types t",
        "  for step in 1..K_max:",
        "    g(e) = progress(best+e) - progress(best)",
        "    u(e) = eta*normalize(s_e) + (1-eta)*normalize(g)",
        "    pick = argmax u(e) over unselected e",
        "    best[type(pick)] = max(best, R[:, pick])",
        "    if step>=K_min and min_t tail_q(best)>=gamma:",
        "      break",
        "  m = picked;  K = len(picked)",
    ]
    yy = 158
    for line in code_lines:
        f.text(1336, yy, line, "mono", 12, (40, 44, 48))
        yy += 18
    f.text(1336, yy + 8, "progress = 各类型“尾部 q 均值”的平均", "zh", 14, INK2)
    f.text(1336, yy + 30, "K_min / K_max 由 min/max_retention 决定", "zh", 14, INK2)
    f.text(1336, yy + 58, "R = ⅓·Jaccard + ⅓·语义cos + ⅓·统计核", "zh", 15, (28, 30, 34))
    f.text(1336, yy + 84, "R 固定等权、不训练 → 覆盖口径训练/评估一致", "zh", 13, INK2)

    # 列间箭头
    f.arrow([(700, 300), (734, 300)])
    f.arrow([(1284, 300), (1318, 300)])

    # ---------------- ④⑤⑥ 掩码与梯度
    f.card(56, 550, 580, 740, "new", "④ hard mask：前向拓扑", [
        ("txt", "m_e ∈ {0,1}，前向只让 m_e=1 的边进入 HGNN 的边池化与 IBG。"),
        ("txt", "未被选中的边在 IBG 里不存在，其社区/行为关系随之消失。"),
        ("note", "因此“压缩率”与“证据集合”是同一个决策，可解释性直接来自 m。"),
    ], tag="硬决策", title_size=19)
    f.card(600, 550, 1124, 740, "new", "⑤ soft mask：只负责梯度", [
        ("txt", "在会话内二分阈值 τ*，使 Σ σ((s_e−τ*)/T) = K，"),
        ("txt", "得到一个“总预算恰好等于 K”的平滑竞争解。"),
        ("eq", "ST = m + (soft − m).detach()"),
        ("note", "前向数值 = m；∂L/∂s 只经 soft 这一支。"),
    ], tag="可微通道", title_size=19)
    f.card(1144, 550, 1744, 740, "loss", "⑥ 梯度实际流向", [
        ("eq", "fixed / adaptive：∂L/∂s_e 走完整 soft"),
        ("eq", "robust：∂L/∂s_e 仅来自 η·(task−task.detach()) 项"),
        ("txt", "覆盖、预算、停止、贪心排序全部 detached —— 它们是规则而非参数。"),
        ("note", "可选 A1′：R_keep = Σ_{e<f} m_e m_f J(e,f) / (Σ m_e)^2，λ·R_keep 只塑形打分器（默认 λ=0）。"),
    ], tag="autograd", title_size=19)
    f.arrow([(580, 635), (598, 635)])
    f.arrow([(1124, 635), (1142, 635)])

    # ---------------- 动机
    f.card(56, 760, 1744, 960, "proto", "为什么需要这一层（相对论文的动机）", [
        ("txt", "论文的 IBG 对所有超边一视同仁；但 Vine 上 STC 边占 91.7%（63389/69116），且近重复严重 → 行为基高度冗余，"),
        ("txt", "IBG 被大量“几乎同一条边”的连接淹没，Louvain 社区退化，融合时的注意力也被稀释。"),
        ("txt", "BBP 的做法：用可学习打分器 + 分类型尾部覆盖约束，先挑出“少量但覆盖完整”的行为基，再让 IBG 只在这批行为上建模关系。"),
        ("note", "这个“选边”目标是原论文没有的；它同时承担压缩（保留率）与证据选择（可解释）两个角色。"),
        ("code", "对应脚本：scripts/replay_vine_robust_masks.py（掩码重放）、scripts/probe_selector_stability.py（稳定性探针，需旧格式 checkpoint）"),
    ], title_size=19)

    f.text(56, 984, "图 2｜选择器的三段结构：打分（可学习）→ 预算与覆盖（规则，不可微）→ 掩码（straight-through，把规则接回梯度）。", "zh", 16, INK2)
    f.text(56, 1010, "关键取舍：覆盖与预算不可微，因此不作为损失项训练，而作为“约束”施加；模型只在打分器这一处保留自由度。", "zh", 16, INK2)
    f.save("fig2_selector.png")


# ===================================================================== 图 3
def fig_paper_modules():
    f = Fig(
        1800,
        1000,
        "图 3  论文 Fig.2(b)(c) 的两个子模块在代码中的落点",
        "左：Hyperedge Association Network（IBG + 图传播 + 社区发现）；右：Aggregation Module（注意力融合 + 分类 + 损失）。",
    )
    f.d.rounded_rectangle([48, 104, 900, 146], radius=10, fill=BLUE_B)
    f.text(60, 112, "(b) Hyperedge Association Network（行为关系建模）", "zhb", 21, "white")
    f.d.rounded_rectangle([916, 104, 1744, 146], radius=10, fill=BLUE_B)
    f.text(928, 112, "(c) Aggregation Module（多维信息融合）", "zhb", 21, "white")

    # 左列
    f.card(56, 166, 900, 306, "paper", "IBG 建图：Jaccard 作为行为间关联强度", [
        ("txt", "输入：HGNN 输出的 X_E（每条 kept 超边一个 128 维行为节点）"),
        ("eq", "W_ij = |V_i ∩ V_j| / |V_i ∪ V_j|"),
        ("note", "V_i 为超边 i 的成员节点集；只对 kept 边建图，Jaccard=0 不连边"),
        ("code", "backbone.py::BeHMBackbone.build_ibg"),
    ], tag="Eq.5", title_size=18)
    f.arrow([(478, 306), (478, 314)], BLUE_B, 3, head=9)
    f.card(56, 316, 900, 456, "paper", "图传播：聚合相邻行为的信息", [
        ("txt", "论文：Graph Attention Network（GAT）"),
        ("txt", "代码：GCNConv(128→128)，以 W_ij 作 edge_weight"),
        ("note", "实现层偏差：语义等价于“按重叠度加权聚集”，但没有注意力头"),
        ("code", "backbone.py::self.hyperedge_gnn"),
    ], tag="§IV-B", title_size=18)
    f.arrow([(478, 456), (478, 470)], BLUE_B, 3, head=9)
    f.card(56, 472, 900, 590, "paper", "社区发现：找出相互支撑的行为群", [
        ("txt", "加权 Louvain 划分 IBG（random_state = 42，保证可复现）"),
        ("txt", "社区内 mean pooling → 社区级协同模式 C_comm"),
        ("code", "backbone.py::CommunityFeatureExtractor"),
    ], tag="§IV-B", title_size=18)
    f.arrow([(478, 590), (478, 602)], BLUE_B, 3, head=9)
    f.card(56, 604, 900, 719, "paper", "输出：C_comm", [
        ("txt", "每个社区一个 128 维向量；社区数由 Louvain 决定（无超参）"),
        ("note", "IBG 无连接时每行为自成一社区，等价于不做社区聚合（代码显式兜底）"),
        ("note", "→ 作为“全局协同模式”的原料送入右侧融合模块"),
    ], title_size=18)

    # 右列
    f.card(916, 166, 1744, 259, "paper", "全局协同模式 C_G", [
        ("eq", "C_G = AttentionPooling(C_comm)"),
        ("note", "以社区为序列做掩码注意力池化（MaskedAttentionPooling）"),
    ], title_size=18)
    f.arrow([(1330, 259), (1330, 271)], BLUE_B, 3, head=9)
    f.card(916, 273, 1744, 366, "paper", "局部证据 X_L", [
        ("eq", "X_L = [ X_N ; X_E ]"),
        ("note", "节点序列 + 超边（行为）序列拼接，各自带 mask 做 padding 对齐"),
    ], title_size=18)
    f.arrow([(1330, 366), (1330, 378)], BLUE_B, 3, head=9)
    f.card(916, 380, 1744, 473, "paper", "条件注意力：让局部证据对齐全局模式", [
        ("eq", "C_L = softmax(C_G·X_L^T / √d_k) · X_L"),
        ("note", "代码：Q=C_G、K/V=X_L 的三组线性投影 + 缩放点积 + LayerNorm + Dropout"),
    ], tag="Eq.6", title_size=18)
    f.arrow([(1330, 473), (1330, 485)], BLUE_B, 3, head=9)
    f.card(916, 487, 1744, 580, "paper", "残差融合", [
        ("eq", "C_final = C_G + Linear(C_L)"),
        ("note", "Linear 处代码为 Linear→LayerNorm→Dropout（output_projection）"),
    ], tag="Eq.7", title_size=18)
    f.arrow([(1330, 580), (1330, 592)], BLUE_B, 3, head=9)
    f.card(916, 594, 1744, 734, "loss", "分类与损失", [
        ("eq", "logits = f(C_final) − δ·log π     (Eq.8)"),
        ("eq", "L = 1/B Σ c_i·CE(logits_i, y_i)  (Eq.9)"),
        ("note", "代码额外在 CE 里加类权 [1, 1.5]；π 与 δ 均来自该折训练集统计"),
        ("note", "f(·) = Linear(128→2)；δ = softplus(a) 可学习"),
    ], tag="Eq.8/9", title_size=18)

    # 底部要点
    f.card(56, 750, 1744, 930, "proto", "这一层的三个“论文写法 → 代码落点”要点", [
        ("hi", "1. 论文的 IBG 是“所有超边”；代码的 IBG 是“kept 超边”——因为前面插入了 BBP 选择层（图 2），两者不是同一张图。"),
        ("hi", "2. 论文写 GAT，代码用 GCNConv + Jaccard edge_weight；若审稿要求严格复现，这是唯一需要改的一行（backbone.py::self.hyperedge_gnn）。"),
        ("hi", "3. 论文 Eq.6 直接 inner-product；代码是 QKV 投影版，表达力更强但多 3 个 128×128 矩阵，参数量与论文描述不完全一致。"),
        ("note", "补充：Eq.7 的 Linear 在代码中被扩展为 Linear+LayerNorm+Dropout；Eq.9 额外乘了类权，属工程增强而非论文内容。"),
    ], title_size=19)

    f.text(
        56, 954,
        "图 3｜对应论文 Fig.2(b)(c) 与 Eq.5–9。左半部分决定“行为之间的关系”，右半部分决定“如何把多层证据压成一个会话向量”。",
        "zh", 15, INK2,
    )
    f.save("fig3_paper_modules.png")


# ===================================================================== 图 4
def fig_protocol():
    f = Fig(
        1800,
        740,
        "图 4  训练与选型协议（论文 §V-B 的 80/20 单次划分 → 本仓库的 CV5 协议）",
        "目的：最终测试集只花一次；所有超参 / epoch / 阈值的选择都发生在开发集内部。",
    )
    y0, y1 = 130, 300
    cards = [
        (56, 380, "data", "① 预处理快照", [
            ("txt", "970 会话 / 68662 节点 / 768 维"),
            ("note", "SHA256 校验，只读加载"),
        ]),
        (400, 720, "paper", "② 固定最终测试集", [
            ("txt", "StratifiedShuffleSplit(20%)"),
            ("txt", "→ final_test = 194 会话"),
            ("note", "此后不再参与任何选择"),
        ]),
        (740, 1060, "paper", "③ 开发集 5 折", [
            ("txt", "dev = 776 会话"),
            ("txt", "StratifiedKFold(5, seed=42)"),
            ("note", "每折 620 训练 / 156 验证"),
        ]),
        (1080, 1400, "paper", "④ 每折独立训练", [
            ("txt", "20 epoch，每 epoch 存 CPU 权重"),
            ("txt", "先验/置信度统计只用该折训练集"),
            ("note", "同 seed；折间只换数据"),
        ]),
        (1420, 1744, "new", "⑤ 全局 epoch 选择", [
            ("txt", "跨折同 epoch 聚合"),
            ("txt", "max(mean, −std, −epoch)"),
            ("note", "选完才允许碰测试集"),
        ]),
    ]
    for (x0, x1, kind, title, lines) in cards:
        f.card(x0, y0, x1, y1, kind, title, lines, title_size=18)
    for a, b in [(380, 400), (720, 740), (1060, 1080), (1400, 1420)]:
        f.arrow([(a, 215), (b, 215)])

    f.card(56, 340, 880, 540, "proto", "⑥ 产物（可复现性证据链）", [
        ("code", "split.json        : 折索引 + 类分布 + 数据 SHA256"),
        ("code", "cv_history.json   : 每折每 epoch 的完整验证指标"),
        ("code", "result.json       : 全局 epoch、聚合曲线、resolved_config"),
        ("code", "fold_checkpoints/ : fold_XX_epoch_YYY.pt（只存被选中的 epoch）"),
        ("code", "ensemble_model.pt : 5 折权重 + 配置 + 结果"),
    ], title_size=19)
    f.card(900, 340, 1744, 540, "proto", "⑦ 最终测试（可选、只跑一次）", [
        ("txt", "5 个折模型对同一 final_test 各自前向 → 正类概率取平均 → 阈值固定 0.5。"),
        ("txt", "默认关闭（--evaluate-final-test），避免在调参过程中反复消耗测试集。"),
        ("note", "另有 scripts/calibrate_vine_cv5_oof_threshold.py：用折外 OOF 概率校准统一阈值，不读测试集；"),
        ("note", "其结论目前未回灌主协议（主协议仍用 0.5），属并行实验通道。"),
    ], title_size=19)

    f.card(56, 560, 1744, 676, "loss", "协议级不变量", [
        ("txt", "① 任何统计量（类先验 π、置信度均值、δ 的初始化）只看该折训练会话；② 验证集只用于 epoch 选择与阈值；"),
        ("txt", "③ 最终测试集只在配置锁定后跑一次；④ 折间用同一训练 seed，差异只来自数据轮换 —— 因此“跨折均值”可比。"),
    ], title_size=18)

    f.text(
        56, 700,
        "图 4｜论文是“80/20 + 5 次随机重复取均值”；代码改成“80/20 + 开发集内 5 折 + 单一全局 epoch”，用折间一致性替代重复实验。",
        "zh", 15, INK2,
    )
    f.save("fig4_protocol.png")


if __name__ == "__main__":
    fig_overview()
    fig_selector()
    fig_paper_modules()
    fig_protocol()
