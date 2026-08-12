"""
chart 可视化模块：为期刊画像报告生成 4 类可视化图表（PNG）。

依赖：matplotlib（若未安装，各函数自动降级返回 None，不影响主流程）。
依赖声明已加入 requirements.txt（matplotlib>=3.5）。

- plot_method_distribution  : 研究方法范式饼图
- plot_sample_size_dist     : 定量样本量分布条形图
- plot_top_theories         : 高频理论框架横向条形图
- plot_open_science         : 开放科学实践占比条形图
- plot_reporting_styles     : 统计汇报风格占比条形图
- plot_radar               : 候选期刊多维对比雷达图（多期刊路由）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # 无界面后端，可安全在后台/服务器运行
    import matplotlib.pyplot as plt

    # 配置中文字体（Windows 优先微软雅黑，其次常见中文字体），避免图表中文变方块
    from matplotlib import font_manager

    _CJK_CANDIDATES = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
    ]
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    _chosen = next((f for f in _CJK_CANDIDATES if f in _installed), None)
    if _chosen:
        plt.rcParams["font.sans-serif"] = [_chosen] + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False

    HAS_MPL = True
except Exception as e:  # pragma: no cover - 环境缺依赖时的降级
    HAS_MPL = False
    logger.warning(f"matplotlib 不可用，可视化图表将被跳过（不影响报告生成）: {e}")

# 与报告风格一致的中文/学术配色
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#64B5CD", "#8C564B", "#E377C2"]


def _ensure_dir(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def _finalize(fig, out_path: str) -> Optional[str]:
    """保存图表并关闭 figure，返回路径（失败返回 None）。"""
    if not HAS_MPL:
        return None
    try:
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path
    except Exception as e:
        logger.warning(f"保存图表失败 {out_path}: {e}")
        plt.close(fig)
        return None


def plot_method_distribution(aggregated_stats: Dict[str, Any], out_path: str) -> Optional[str]:
    """研究方法范式分布饼图。"""
    if not HAS_MPL:
        return None
    dist = (aggregated_stats or {}).get("method_distribution", {})
    if not dist:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = list(dist.keys())
    sizes = [v.get("count", 0) for v in dist.values()]
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=_PALETTE[: len(labels)],
        textprops={"fontsize": 9},
    )
    ax.set_title("研究范式分布 (Method Distribution)", fontsize=11, weight="bold")
    return _finalize(fig, _ensure_dir(out_path))


def plot_sample_size_dist(aggregated_stats: Dict[str, Any], out_path: str) -> Optional[str]:
    """定量样本量 min/median/max 条形图。"""
    if not HAS_MPL:
        return None
    stats = (aggregated_stats or {}).get("sample_size_stats", {})
    vals = [stats.get(k) for k in ("min", "median", "max")]
    if not vals or all(v in (None, "N/A") for v in vals):
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    names = ["最低", "中位数", "最高"]
    colors = ["#C44E52", "#4C72B0", "#55A868"]
    data = [v if isinstance(v, (int, float)) else 0 for v in vals]
    bars = ax.bar(names, data, color=colors, alpha=0.85)
    for b, v in zip(bars, data):
        ax.text(b.get_x() + b.get_width() / 2, v + max(data) * 0.02, f"{int(v)}", ha="center", fontsize=10, weight="bold")
    ax.set_ylabel("样本量 (N)")
    ax.set_title("定量实证样本量分布 (Sample Size)", fontsize=11, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _finalize(fig, _ensure_dir(out_path))


def plot_top_theories(aggregated_stats: Dict[str, Any], out_path: str, top_n: int = 8) -> Optional[str]:
    """高频理论框架横向条形图。"""
    if not HAS_MPL:
        return None
    top = (aggregated_stats or {}).get("top_theories", [])
    if not top:
        return None
    top = top[:top_n]
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.45 * len(top))))
    names = [t.get("name", "")[:32] for t in top]
    counts = [t.get("count", 0) for t in top]
    ax.barh(names[::-1], counts[::-1], color=_PALETTE[0], alpha=0.85)
    ax.set_xlabel("出现频次")
    ax.set_title(f"高频理论框架 Top {len(top)} (Theories)", fontsize=11, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    return _finalize(fig, _ensure_dir(out_path))


def plot_open_science(aggregated_stats: Dict[str, Any], out_path: str) -> Optional[str]:
    """开放科学实践占比条形图。"""
    if not HAS_MPL:
        return None
    ostats = (aggregated_stats or {}).get("open_science_stats", {})
    if not ostats:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    names = list(ostats.keys())
    pcts = [v.get("percentage", 0) for v in ostats.values()]
    ax.bar(names, pcts, color=_PALETTE[1], alpha=0.85)
    ax.set_ylabel("占比 (%)")
    ax.set_title("开放科学实践占比 (Open Science)", fontsize=11, weight="bold")
    for i, p in enumerate(pcts):
        ax.text(i, p + 1, f"{p:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _finalize(fig, _ensure_dir(out_path))


def plot_reporting_styles(aggregated_stats: Dict[str, Any], out_path: str) -> Optional[str]:
    """统计汇报风格占比条形图。"""
    if not HAS_MPL:
        return None
    styles = (aggregated_stats or {}).get("top_reporting_styles", [])
    if not styles:
        return None
    styles = styles[:5]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = [s.get("style", "")[:30] for s in styles]
    counts = [s.get("count", 0) for s in styles]
    ax.barh(names[::-1], counts[::-1], color=_PALETTE[2], alpha=0.85)
    ax.set_xlabel("论文数")
    ax.set_title("统计汇报风格分布 (Reporting Styles)", fontsize=11, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    return _finalize(fig, _ensure_dir(out_path))


def plot_radar(journals: List[Dict[str, Any]], out_path: str) -> Optional[str]:
    """多期刊多维对比雷达图（多期刊路由/梯队对比用）。

    journals: [{"name": ..., "score": 0-100, ...}]，取前 6 个期刊，每个期刊最多 6 维。
    字典键用于维度名，取值限定 [0, 100]。
    """
    if not HAS_MPL or not journals:
        return None
    import math

    journals = journals[:6]
    # 维度 = 所有期刊键名的并集（排除 name 等非数值键）
    numeric_keys = ["录用难度", "范式契合", "样本门槛", "理论契合", "引用生态", "时效契合"]
    dim_keys = numeric_keys[: 6 if len(numeric_keys) <= 6 else len(numeric_keys)]
    # 若传入数据缺某些维度则补 0
    for j in journals:
        for k in dim_keys:
            j.setdefault(k, 0)

    n = len(dim_keys)
    angles = [i / float(n) * 2 * math.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for idx, j in enumerate(journals):
        values = [float(j.get(k, 0)) for k in dim_keys]
        values += values[:1]
        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(angles, values, "o-", linewidth=1.8, color=color, label=j.get("name", f"期刊{idx+1}"))
        ax.fill(angles, values, alpha=0.12, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_keys, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("候选期刊多维对比 (Radar)", fontsize=11, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)
    return _finalize(fig, _ensure_dir(out_path))


def generate_all_charts(
    aggregated_stats: Dict[str, Any],
    out_dir: str,
    journals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    """为聚合统计生成全套可视化图表，返回 {图表名: 路径}。"""
    paths: Dict[str, Optional[str]] = {}
    paths["method_distribution"] = plot_method_distribution(aggregated_stats, os.path.join(out_dir, "method_distribution.png"))
    paths["sample_size"] = plot_sample_size_dist(aggregated_stats, os.path.join(out_dir, "sample_size.png"))
    paths["top_theories"] = plot_top_theories(aggregated_stats, os.path.join(out_dir, "top_theories.png"))
    paths["open_science"] = plot_open_science(aggregated_stats, os.path.join(out_dir, "open_science.png"))
    paths["reporting_styles"] = plot_reporting_styles(aggregated_stats, os.path.join(out_dir, "reporting_styles.png"))
    if journals:
        paths["radar"] = plot_radar(journals, os.path.join(out_dir, "radar.png"))
    generated = {k: v for k, v in paths.items() if v}
    logger.info(f"可视化图表生成完成: {list(generated.keys())}")
    return generated
