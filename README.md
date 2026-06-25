<div align="center">

# CAD Wireframe 神经压缩挑战赛 — VQVAE 分支

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

比赛主页: https://mathmagic-official.github.io/AICAD/

数据集以及 Baseline: https://pan.ustc.edu.cn/share/index/8902361d3b5745f78245

## 框架概览

`点云 -> 冻结 Utonia PTv3(多尺度)-> 每尺度 compressor -> 每尺度 ResidualVQ -> 拼接索引(≤4096,提交)-> 边集合解码器 -> wireframe`。

整条流水线是一个**端到端单阶段离散自编码器(VQVAE)**:编码器把原始点云压成**多尺度**的连续 token,每个尺度用独立的
残差向量量化器(ResidualVQ)离散成码本索引;**拼接后的扁平索引**(`Σ_s N_s·n_q ≤ 4096`)即比赛提交内容。
解码器**仅凭这些索引**(索引 → 码本 → `z_q` → 解码器)以**边为中心**:`512` 个边 query 直接预测**每条边的存在性 +
32 个有序世界坐标采样点**(首/尾点即两端顶点),顶点由推理时**端点 union-find 聚合**得到。无曲线 VAE、无归一化、无 KL
(latent 是离散确定性的)。

```mermaid
flowchart LR
  PC["原始点云 (coord ΣN×3, offset B)"] --> ENC["冻结 Utonia PTv3 (enc_mode, no_grad)"]
  ENC --> MS["取若干 enc stage 的逐体素特征(不上采样)"]
  MS --> COMP["每尺度 compressor:N_s 个 query cross-attn"]
  COMP --> Z["多尺度连续 token z_s (B,N_s,256)"]
  Z --> RVQ["每尺度 ResidualVQ(n_q 级)"]
  RVQ --> IDX["拼接扁平索引 (B, ΣN_s·n_q ≤4096) = 提交"]
  IDX --> ZQ["decode_indices:索引 -> 码本 -> z_q"]
  ZQ --> DEC["边集合解码器:512 边 query(边间自注意 + 对 z_q cross-attn)"]
  DEC --> EX["逐边:exist logit"]
  DEC --> PT["逐边:32×3 有序采样点(pts[0]=v1, pts[-1]=v2)"]
  EX --> AGG["保留边 -> union-find 端点合并 -> 共享顶点"]
  PT --> AGG
  AGG --> WF["vertices, edge_index, edge_points"]
  WF --> MET["CCD / TA / VPE"]
```

| 模块 | 作用 |
| --- | --- |
| **UtoniaEncoder**(冻结 `Utonia PTv3` + 每尺度可训练 `LatentCompressor`) | **原始变长点云**(打包成 `coord (ΣN,3)` + `offset (B,)`)→ 体素去重 → 冻结的 [Utonia](https://huggingface.co/Pointcept/Utonia) 预训练 PTv3 编码器(`enc_mode`、`eval`+`no_grad`,确定性)→ 沿 `GridPooling` 的 `pooling_parent` 链取**若干 enc stage 的逐体素特征**(**不上采样**,每尺度保持其原生分辨率;通道 `enc_channels[stage]` 从 ckpt 配置读取)→ 每尺度一个 compressor 池化成 `z_s (B,N_s,256)`,输出多尺度 token 列表(细→粗)。默认用**全部 5 个 enc stage**(`scale_stages=[0..4]`),token 分配 `scale_tokens=[192,128,96,64,32]`(细尺度给更多 token 保细节);`compressor_heads=6` 须整除每个被用 stage 的通道(`54/108/216/432/576`)。backbone 冻结、只训 compressor。详见 `src/models/utonia_encoder.py`。 |
| **MultiScaleResidualVQ**(每尺度独立 `ResidualVQ`) | 每尺度用各自的残差 VQ(`n_q` 级)把连续 token 量化成索引;**拼接成扁平索引**(固定 layout:尺度→token→量化级),`总索引 = Σ_s N_s·n_q`,**构造时**校验 `≤4096`(默认 `5` 尺度 `512×8 = 4096` 顶满)。`codebook_size` 支持**逐尺度** list(粗尺度 token 少,缩小码本以匹配利用率、抗坍塌)。`forward` 返回直通 `z_q`、扁平索引与 commitment loss;`decode_indices` 仅凭扁平索引重建出 `z_q`(保证 索引→wireframe 的 round-trip)。`eval` 模式自动冻结码本(EMA 关闭)。依赖 `vector-quantize-pytorch`(自行安装)。详见 `src/models/quantizer.py`。 |
| **EdgeSetDecoder**(边 query 集合解码器) | 仅从多尺度 `z_q` 重建 wireframe。各尺度 `z_q` 投影到 `d_model` 并加上**尺度 embedding** 后 concat 成 memory;`512` 个**边 query** 经 `nn.TransformerDecoder`,每层 =**边间自注意**(边互相协调,利于共享顶点)+ 对 `z_q` **cross-attn** + FFN;每个边 query 直接输出 **exist logit** 与 **`sample_points_num` 个有序世界坐标采样点 `(P,3)`**(约定 `pts[0]=v1`、`pts[-1]=v2`,无曲线 VAE / 无归一化)。详见 `src/models/edge_set_decoder.py`。 |
| **端点聚合重建**(`aggregate_wireframe`) | 保留 `sigmoid(exist) ≥ edge_thresh` 的边(`topk_edges` 上限封顶);若清过阈值的边不足 `min_edges` 则**兜底**取概率最高的前 `min_edges` 条(`min_edges=1` 保证永不输出空 wireframe)。再收集 `2E` 个端点,**union-find** 合并距离 `< τ_merge` 的端点为**共享顶点**(按 exist 加权均值),重建 `edge_index` 并去**自环 / 重复边**;最后把每条边首尾点**钉合**到合并顶点、中间点按两端位移线性混合补偿。详见 `src/recon/wireframe.py`。 |

## 目标 / 监督 (target)

每个样本保留**原生 GT wireframe 图**:顶点 + `edge_index` + 每条边的有序采样点 `edge_points (E,P,3)`(其首/尾点即两端顶点)。
边集合解码器直接回归每条边的有序点列,**不再产出**参数化的 `edge_type`(line/arc/bezier)/ anchor 监督。

坐标**全程保持原始**(数据集已归一化到 `[-1, 1]`,无需额外归一化)。点云若少于 `min_pc_points=100` 个点,或顶点数 `> max_vertices=512` 该样本会被**跳过**。

## 损失(边集合预测 + VQ)

- **顺序无关匈牙利匹配**:用 `scipy.optimize.linear_sum_assignment` 把 `512` 个边 query 与 GT 边按**首尾端点** L1 代价匹配(取正序/反序中更优者,同时定下逐点监督的方向);
- `loss_exist`:边存在 **标定 BCE**(默认 `focal_gamma=0`):逐样本 `pos_weight = #neg/#pos`(上限 `exist_pos_weight_max`)平衡负例占多数 + 小幅标签平滑(`exist_label_smoothing`),让 `sigmoid(logit)` 在判决边界回到 `0.5` 附近(`edge_thresh=0.5` 即可用),不再被 focal 压扁。把 `focal_gamma>0` 即可切回旧的 focal loss;
- `loss_points`:匹配边的 **32 点有序逐点 L1**(按配对方向取 min,曲线"不乱飘"的根本,优于 chamfer),并按 GT **矢高(sagitta)**给曲边加权(`curv_l1_scale`),避免被占多数的直线边淹没;
- `loss_endpoint`:`pts[0]/pts[-1]` 端点的额外加权 L1(GT 中共享顶点的边端点坐标相同 → 把该聚的端点拉到一起,TA 的关键);
- `loss_sagitta`:把每条边的**弦残差**(各内部点相对端点弦的偏移)对齐到 GT 残差,让解码器学到**真实曲率**而非塌成直线(`chord_residual` 的零偏移默认值);
- `loss_smooth`:二阶差分平滑,但**相对 GT**(`(pred_2nd − gt_2nd)^2`,**小权重**)——只把 GT 本身是直线的边压直,曲边保留弯曲;
- `loss_seglen`:相邻段长方差(默认 `w_seglen=0` 关闭,曾误伤曲率);
- `loss_consistency`:把匹配端点按 GT 顶点 id 分组,同时惩罚**组内方差**与到该 GT 顶点**绝对坐标**的距离(抗约 2× 顶点过预测/悬空边,使 union-find 合并生效);
- `loss_commit`:VQ commitment loss,经 `quant_warmup_steps`(先用连续 `z` 预热)后,权重 `0 → w_commit` 线性 ramp。

并按尺度记录码本 `perplexity`(`vq/perplexity_s*`)监控码本利用率/塌缩;验证时额外记录可观测性指标(`recon/nonempty_frac`、`recon/pred_vertices`、`recon/pred_edges` vs `recon/gt_*`)。验证时从 `z_q` 解码并在 `(edge_thresh, tau_merge, topk_edges)` 网格上重算分数,checkpoint 按其中最优的 `val/score_best` 选优(越大越好),并记录最优解码参数 `val/best_{edge_thresh,tau_merge,topk_edges}`。

## 训练

依赖:点云栈(Utonia PTv3 需 `spconv` / `flash-attn` / `torch_scatter` / `timm`)、`pytorch_lightning` / `torchmetrics`、`pytorch3d`(KNN chamfer)、`scipy`(匈牙利匹配),以及 **`vector-quantize-pytorch`**(VQVAE 量化器,**自行安装**:`pip install vector-quantize-pytorch`)。Utonia 权重默认从本地 `logs/utonia/utonia.pth` 加载(在 `configs/vqvae*.yaml` 的 `pc_encoder.utonia` 配置)。

```bash
# 单 GPU
python -m src.main fit --config configs/data.yaml --config configs/vqvae.yaml
# 也可以： bash scripts/run.sh train

# 8x A800 DDP
python -m src.main fit --config configs/data.yaml --config configs/vqvae_ddp.yaml
# 也可以： bash scripts/run.sh train_ddp
```

显存/速度杠杆:`pc_encoder.scale_tokens`(多尺度 token 分配)、`quantizer.{n_q,codebook_size}`、`decoder.{num_edge_queries,sample_points_num,num_layers,d_model}`、`data.batch_size`。

## 推理 / 提交

提交导出是**单次前向**(encode → 多尺度 `z` → 每尺度 RVQ → 拼接索引),并在写出前用 `decode_indices` 从**提交的索引**还原 `z_q` 再解码(代码层面保证 round-trip 与 `budget ≤ 4096`)。每个样本的 `latent` 字段即扁平索引向量(float32)。

```bash
# 单 GPU
python scripts/export_submission.py --ckpt <vqvae.ckpt 目录或文件> --out-dir logs/submission
# 也可以： CKPT=<vqvae.ckpt> bash scripts/run.sh export_submission

# 8-GPU 数据并行(每 GPU 一个 worker,自动合并 + 打包 submission.zip)
python scripts/export_submission.py --spawn 8 --ckpt <vqvae.ckpt> --out-dir logs/submission

# 断点续跑
python scripts/export_submission.py --spawn 8 --ckpt <vqvae.ckpt> --out-dir logs/submission --resume
```

提交布局:

```text
submission/
    latent_pack.npz                 # stems (N,) + latents (N, K<=4096) 扁平索引
    sample_edge/<stem>.npz
        latent       : (K,) float32   # 拼接的 RVQ 索引(固定 layout)
        vertices     : (V, 3) float32
        edge_index   : (E, 2) int32
        edge_points  : (E, 32, 3) float32
        num_vertices : () int32
        num_edges    : () int32
```

## 可视化

```bash
# val：input point cloud | 预测 wireframe(+逐样本 score/ccd/ta/vpe) | GT wireframe
python scripts/vis_ae_val.py --ckpt <vqvae.ckpt> --split val --num 6

# test(无 GT)：额外画出 baseline submission 列对照
python scripts/vis_ae_val.py --ckpt <vqvae.ckpt> --split test --num 8 --out logs/ae_test.png
```

`scripts/make_split.py` 生成 `data/split.json`;`scripts/render_wireframe.py` 用于把单个 wireframe npz 渲染成图。
