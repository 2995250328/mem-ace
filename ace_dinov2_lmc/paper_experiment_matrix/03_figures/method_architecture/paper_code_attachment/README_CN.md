# Mem-ACE 绘图事实附件

这个附件只用于辅助 draw.io 绘图 Codex 核对方法事实，不是生产训练代码的替代实现。绘图时优先读图稿规格；需要代码事实时，按 `provenance/SOURCE_MAP.md` 回到生产代码核对。

推荐阅读顺序：

1. `../figure_spec.md`
2. `DRAWIO_HANDOFF_CN.md`
3. `configs/common_method.json`
4. `provenance/SOURCE_MAP.md`
绘图时只采用共同方法事实：pooled scene memory、Geo-token compressor、anchored tokens `(Z, P)`、`PE(P-c)`、默认 `value_only_raw` / value-geometry memory fusion、SCR coordinate head、single-view reprojection、cross-view track reprojection、Geometric Alignment、Cached Memory Adaptation。

不要把历史实验分支、下游全局 refinement 细节或代码阶段名画入正文主图。
