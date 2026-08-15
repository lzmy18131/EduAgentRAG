# -*- coding: utf-8 -*-
"""检索参数网格搜索:进程内一次加载模型,遍历 dense/sparse 权重组合,输出对比表。

用法(项目根):
    python -m 2Milvus_RAG_Qa.RAG评测.grid_search
"""
import importlib
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from base.config import cfg

_eval_mod = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")

GRID = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]
TOP_K = 5
GOLDEN = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden.json"


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"金标 {len(golden)} 条, 网格 {len(GRID)} 组 × top_k={TOP_K}", flush=True)

    results = []
    for dw, sw in GRID:
        cfg.DENSE_WEIGHT = dw
        cfg.SPARSE_WEIGHT = sw
        t0 = time.time()
        r = _eval_mod.run_eval(golden, TOP_K)
        dt = time.time() - t0
        results.append((dw, sw, r["hit_rate"], r["mrr"]))
        print(f"dense={dw:.1f} sparse={sw:.1f} | HitRate@{TOP_K}={r['hit_rate']:.4f} MRR={r['mrr']:.4f} ({dt:.0f}s)", flush=True)

    best = max(results, key=lambda x: x[2])
    print("=" * 60)
    print(f"最优: dense={best[0]:.1f} sparse={best[1]:.1f} → HitRate@{TOP_K}={best[2]:.4f} MRR={best[3]:.4f}")
    print("=" * 60)

    out = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "grid_report.json"
    out.write_text(json.dumps(
        [{"dense_weight": dw, "sparse_weight": sw, "hit_rate": hr, "mrr": m}
         for dw, sw, hr, m in results],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"对比表已写入 {out}")


if __name__ == "__main__":
    main()
