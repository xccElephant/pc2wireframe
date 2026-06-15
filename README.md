<div align="center">

# CAD Wireframe 神经压缩挑战赛

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

比赛主页: https://mathmagic-official.github.io/AICAD/

数据集以及 Baseline: https://pan.ustc.edu.cn/share/index/8902361d3b5745f78245

## 框架概览

点云 → wireframe 的**两阶段**流水线，每个阶段独立训练、独立 config，第一阶段的权重冻结后喂给第二阶段：

| 阶段 | 模块 | 作用 | config |
| --- | --- | --- | --- |
| Stage 1 | **Curve VAE** (`AutoencoderKL1D`) | 把单条**规范化曲线**编码成 12 维 token latent，并可在任意参数 `t` 处解码 | `configs/curve_vae.yaml` |
| Stage 2 | **PC2Wireframe** (PTv3 + Latent Compressor + Transformer Decoder) | 点云直接预测 wireframe：PTv3 提特征 → cross-attn 压缩成 `16×256` latent → Transformer 解码器并行预测节点集 + 边集（含曲线 latent）；**Curve VAE 全程冻结**，仅用来解码曲线 | `configs/pc2wireframe.yaml` |

`16 × 256 = 4096` floats，正好是比赛的 latent 预算上限。

## Pipeline (AI生成)

![pipeline](assets/pipeline.png)


## Curve VAE (50 Epoch)

![recon_spread](assets/recon_spread.png)

![recon_worst](assets/recon_worst.png)


## PC2Wireframe

端到端地把点云解码成 wireframe，**不再有独立的 wireframe VAE / teacher posterior**——直接对预测出的图做集合监督。参考官方基线，但把"候选对 + 贪心匹配"换成了 DETR 风格的**学习查询 + 全局匈牙利匹配**：

- **编码**：PTv3 backbone 提取逐体素点云特征；`16` 个可学 query 通过 cross-attention 把它们汇聚成 `(B, 16, 256)` 的高斯后验 latent（保留小 KL 正则，提交时用 `mu`）。
- **解码器**：隐变量 token 投影后作为 cross-attention 的 **memory**。
  - **节点**：`num_node_queries=768` 个学习查询并行 cross-attend 到 memory，两个头分别预测每个节点的**坐标**与**存在置信度**（免数数的节点集）。
  - **边**：`num_edge_queries=1024` 个边查询先 self-attn，再 cross-attend 到 latent memory，最后 cross-attend 到**节点特征**；预测每条边的**存在性**、两个**端点分布**（指针式 `edge_q · node_k` 在节点查询上的 softmax）以及一个**逐边曲线 latent**。
  - **曲线**：边查询输出的曲线 latent 喂给**冻结的 Stage-1 Curve VAE**，在端点线性插值基线上解码出残差折线（规范帧），推理时再 denorm 到预测端点上。
- **训练（集合预测）**：先用**匈牙利匹配**把节点查询对齐到 GT 顶点（坐标 L1 代价），再以该匹配为基准把边查询对齐到 GT 边（存在性 + 端点 log 概率代价）。在匹配空间监督：
  - 节点坐标 **L1** + 节点存在 **加权 BCE**；
  - 边存在性 **Focal BCE**（处理边的正负样本极端不平衡）；
  - 端点分布 **交叉熵**（在节点查询上的指针分类）；
  - 匹配边的曲线 **L1**（规范帧内，对齐冻结 Curve VAE 的解码）。
- **边方向**：沿用数据集的有向 `(start, end)` 约定，端点-A 目标与规范曲线朝向天然一致，推理时端点 A→B 顺序即解码顺序。

模型见 `src/models/{pc_encoder,wireframe_decoder,pc2wireframe}.py`，匹配 / 损失见 `src/models/criterion.py`，数据打包见 `src/models/packing.py`。

### 重建效果 (200 Epoch)

TODO......


## 训练

```bash
# Stage 1: Curve VAE
python -m src.main fit --config configs/data.yaml --config configs/curve_vae.yaml

# Stage 2: PC2Wireframe（Curve VAE 冻结，从 stage-1 ckpt 加载）
python -m src.main fit --config configs/data.yaml --config configs/pc2wireframe.yaml \
    --model.curve_vae_ckpt <stage1.ckpt>

# 8x A800 DDP 版本用 configs/pc2wireframe_ddp.yaml
# 也可以直接用 scripts/run.sh： CURVE_VAE_CKPT=<stage1.ckpt> bash scripts/run.sh stage2
```

## 推理 / 提交

```bash
python -m src.main predict --config configs/data.yaml --config configs/pc2wireframe.yaml \
    --ckpt_path <stage2.ckpt>
```
