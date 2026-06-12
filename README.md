# pc2wireframe

比赛主页: https://mathmagic-official.github.io/AICAD/

数据集以及 Baseline: https://pan.ustc.edu.cn/share/index/8902361d3b5745f78245

## 框架概览

点云 → wireframe 的**三阶段**流水线，每个阶段独立训练、独立 config，前一阶段的权重冻结后喂给下一阶段：

| 阶段 | 模块 | 作用 | config |
| --- | --- | --- | --- |
| Stage 1 | **Curve VAE** (`AutoencoderKL1D`) | 把单条**规范化曲线**编码成 12 维 token latent，并可在任意参数 `t` 处解码 | `configs/curve_vae.yaml` |
| Stage 2 | **Wireframe VAE** (`AutoencoderKLWireframe`) | 把"一组曲线（端点 + 差分邻接拓扑 + 曲线 latent）"编码成定长 `64×64` latent 再解码；曲线形状用**冻结的 Stage-1 Curve VAE** 编码 | `configs/wireframe_vae.yaml` |
| Stage 3 | **PC2Wireframe** (PTv3 + Latent Compressor) | 点云预测 `64×64` latent，对齐 Stage-2 teacher posterior 并 decode-through 监督；**两个 VAE 全程冻结** | `configs/pc2wireframe.yaml` |

## Pipeline (AI生成)

![pipeline](assets/pipeline.png)


## Curve VAE (50 Epoch)

![recon_spread](assets/recon_spread.png)

![recon_worst](assets/recon_worst.png)

## Wireframe VAE

TODO......


## PC2Wireframe

TODO......