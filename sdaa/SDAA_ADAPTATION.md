# SDAA 适配说明（Protenix）

本文件记录 Protenix 从 NVIDIA CUDA 迁移到 Teco SDAA 加速卡的适配方案，遵循「CUDA 与 SDAA 双兼容」原则。

> ## ⚠️ 自定义算子（未来发展项）
>
> Protenix 原生包含若干**自定义 CUDA / Triton 算子**，本阶段 SDAA 适配**尚未提供其原生实现**，
> 当前统一回退到 PyTorch 原生算子以保证功能正确性：
>
> | 自定义算子 | 类型 | 当前 SDAA 处理 | 未来计划 |
> |-----------|------|---------------|---------|
> | `FusedLayerNorm`（`layer_norm_cuda_kernel.cu`） | CUDA C++ 扩展 | 回退 `OpenFoldLayerNorm`（纯 torch） | 开发 SDAA 原生 fused kernel |
> | `tri_attention`（Triton） | Triton kernel | 回退 `scaled_dot_product_attention` | 开发 SDAA 原生 attention kernel |
> | `fused_ops`（dropout+residual，Triton） | Triton kernel | 走纯 PyTorch 路径（`FUSED_DROPOUT_RESIDUAL` 默认关闭） | 开发 SDAA 原生 fused kernel |
>
> **以上均为「未来会开发」的自定义算子项**，在原生实现落地前，SDAA 以 torch 算子兜底、精度对齐，
> 性能为后续优化点。

---

## 1. 适配环境

| 项目 | 版本/位置 |
|-----|----------|
| PyTorch | 2.12.0 (Torch-SDAA 3.2.1b0) |
| SDAA Runtime | 3.3.0 (`/opt/tecoai/lib64/libsdaart.so`) |
| SDAA Driver | 3.4.0 |
| TCCL（集合通信） | 3.2.0 (`/opt/tecoai/lib64/libtccl.so`) |
| 加速卡 | Teco 加速卡 × 4（`device_count()==4`，`get_device_name()=/dev/tcaicard0`） |

> 关键事实：Teco 的 PyTorch 构建里 `torch.cuda` **存在但 `is_available()==False`**，
> 真正的加速器由 `torch.sdaa` 暴露（`is_available()==True`）。因此原代码里所有
> `torch.cuda.is_available()` 判断在 SDAA 上都会错误地落到 CPU。

---

## 2. 适配策略（核心思路）

新增了一个统一设备后端模块 **`protenix/utils/torch_backend.py`**，把分散在各处的
`torch.cuda.*` / `"cuda"` 设备串 / `"nccl"` 后端统一抽象掉，让上游代码在 CUDA 与
SDAA 上都能直接跑，不硬编码任何一方。

抽象出的关键接口：

| 接口 | CUDA | SDAA | 说明 |
|-----|------|------|------|
| `device_type()` | `"cuda"` | `"sdaa"` | 设备类型串 |
| `accel_available()` | True | True | 是否有加速器 |
| `device_count()` | n_gpu | 4 | 设备数 |
| `distributed_backend()` | `"nccl"` | `"tccl"` | 集合通信后端 |
| `get_device_capability()` | (major, minor) | `None` | SDAA 上该接口不返回有效值 |
| `empty_cache()` / `manual_seed_all()` / `set_device()` | cuda 版本 | sdaa 版本 | 同名能力 |
| `supports_tf32()` | True | False | tf32 matmul 开关仅 CUDA 有 |
| `supports_sdp_kernel()` | True | False | SDPA kernel 上下文仅 CUDA 有 |

---

## 3. 改动清单

### 3.1 新增文件

- `protenix/utils/torch_backend.py` —— 统一设备后端模块（见上）。

### 3.2 修改文件

| 文件 | 改动 |
|-----|------|
| `protenix/utils/seed.py` | `manual_seed_all` 走抽象层；cudnn/CUBLAS 设置仅在 CUDA 下生效 |
| `protenix/utils/training.py` | optimizer 的 `device_type` 用 `device_type()`；nan 检查用 `accel_available()` |
| `protenix/utils/torch_utils.py` | `autocasting_disable_decorator` 中 autocast 设备串改为动态 |
| `protenix/data/esm/compute_esm.py` | `.cuda()` → `to_accel()`；`.to("cuda")` → `.to(device_type())` |
| `protenix/model/protenix.py` | `allow_tf32` 仅在 `supports_tf32()` 时设置 |
| `protenix/model/modules/confidence.py` | `empty_cache` + `autocast` 设备串动态化 |
| `protenix/model/modules/primitives.py` | `autocast` 设备串动态化 |
| `protenix/model/loss.py` | `empty_cache` + `autocast` 设备串动态化 |
| `protenix/model/utils.py` | `get_autocast_dtype("cuda")` → 动态设备串 |
| `protenix/model/triangular/layers.py` | autocast 动态化（LayerNorm 类型由 `LAYERNORM_TYPE` 环境变量显式控制，未改原逻辑） |
| `protenix/model/triangular/triangular.py` | autocast 动态化 |
| `protenix/model/tri_attention/__init__.py` | Triton 路径仅 CUDA 启用；SDAA 自动回退 SDPA，且 `sdp_kernel` 用 `supports_sdp_kernel()` 保护 |
| `protenix/model/tri_attention/autotune_helpers.py` | `get_device_capability`/`get_device_name` 走抽象层，capability 缺失时返回 `"unknown"` |
| `protenix/utils/permutation/*.py`（3 个文件） | autocast 设备串动态化 |
| `runner/train.py` | 设备初始化、`nccl`→`distributed_backend()`、autocast、GradScaler |
| `runner/inference.py` | 设备初始化、`nccl`→`distributed_backend()`、autocast、V100 能力检测用 `get_device_capability()` 判空 |

---

## 4. 运行前必须修改的配置

以下配置默认走 CUDA 专用实现（cuequivariance/CUTLASS），在 SDAA 上必须改成 torch 实现：

```yaml
# configs（train/inference 均需）
triangle_multiplicative: "torch"      # 原默认 "cuequivariance"
triangle_attention: "triattention"    # 原默认 "cuequivariance"；可选 "torch" / "triattention"
```

- `triattention` 会用到 `tri_attention` 模块，SDAA 上已自动回退到 `scaled_dot_product_attention`。
- 若走 `runner/batch_inference.py`，需传 `--trimul_kernel torch --triatt_kernel triattention`（或 `torch`）。

> 说明：`cuequivariance`（cuex）、`deepspeed`（DS4Sci/CUTLASS）均为 CUDA 专用，
> SDAA 不支持；代码里这些实现都是函数内懒加载，配置切到 `torch` 后不会被触发。

LayerNorm 需**显式设置环境变量** `LAYERNORM_TYPE=torch`（或任意非 `fast_layernorm` 的值），
使 `fastln_is_installed=False`，从而走纯 PyTorch 的 `OpenFoldLayerNorm`。SDAA 上若不设置该
变量，默认 `fast_layernorm` 会尝试编译 `layer_norm_cuda_kernel.cu`（调 nvcc）而报错。

---

## 5. SDAA 算子缺口修复（实测发现，非设备路由问题）

SDAA 的 torch 后端存在若干算子 dtype 覆盖缺口（CUDA/CPU 能自动处理的，SDAA 上会报错）。运行
两个 v2 推理案例时逐一暴露并修复如下（均为 CUDA/SDAA 双兼容、语义等价）：

| 文件 | 原代码 | 问题 | 修复 |
|-----|-------|------|------|
| `protenix/model/modules/transformer.py` | `torch.arcsinh(ref_charge)` | SDAA 的 `arcsinh` 不做 int→float 自动提升（`ref_charge` 是 int64） | `torch.arcsinh(ref_charge.float())` |
| `protenix/model/sample_confidence.py` | `vdw_clash[..., ...].reshape(...).max(dim=-1)[0]` | SDAA 不支持 bool 张量的 `max` 归约 | 改成 `.any(dim=-1)` |
| `protenix/model/sample_confidence.py` | `af3_clash.reshape(...).max(dim=-1)[0]` | 同上 | 改成 `.any(dim=-1)` |

> 说明：`.max(dim=-1)[0]` 作用在 bool 张量上语义等价于 `.any(dim=-1)`（是否有任一 True）。
> 这两处修复均为 int→float 显式转换 / bool 归约替换，在 CUDA 上结果与原文完全一致。

---

## 6. 已知限制与风险

1. **fp16 AMP 的 GradScaler**：`GradScaler` 的 `device` 已改为动态 `device_type()`（SDAA 上为
   `"sdaa"`）。实测 SDAA 的 torch 支持 `GradScaler(device="sdaa")` 且 `scale()` 能正确作用于
   sdaa 张量；而 `device="cuda"` 会触发 "CUDA not available, Disabling" 警告。注意 Protenix 默认
   bf16，`GradScaler` 处于 `enabled=False`（无副作用）；仅 `dtype=float16` 时才真正生效。
2. **`get_device_capability()` 在 SDAA 上返回 `None`**：依赖该值的逻辑（如 V100 降级）会
   安全跳过。
3. **精度未验证**：LayerNorm / tri_attention 在 SDAA 上使用 PyTorch 参考实现而非 CUDA
   fused kernel，数值结果理论上等价，但需实测精度对齐（见第 6 节）。
4. **tri_attention 的 SDPA 兜底**：bias 处理为简化实现，TriangleAttention 精度需重点对比。
5. **`torch.cuda` 仍存在于 SDAA 构建中**：任何新增代码若直接调 `torch.cuda.*`，仍会踩
   "伪可用" 的坑。约定：**新增代码一律走 `protenix.utils.torch_backend`**。

---

## 7. 精度与性能对比

> 测试配置：两个案例均带 MSA，200 步 / 5 样本 / bf16 / triatt=trimul=torch。
> CUDA 环境：4×A100-40GB（taichu04）；SDAA 环境：4×Teco 加速卡（ubuntu21）。
> 数值为 5 个样本的均值。

### 7.1 精度对比

| 案例 | 指标 | CUDA（A100） | SDAA（Teco） | 绝对误差 | 是否达标 |
|------|------|-------------|-------------|---------|---------|
| protein_dna | pLDDT | 89.13 | 89.18 | 0.05 | ✅ |
| | pTM | 0.823 | 0.826 | 0.003 | ✅ |
| | ipTM | 0.898 | 0.903 | 0.005 | ✅ |
| | ranking_score | 0.883 | 0.888 | 0.005 | ✅ |
| multimer_with_ligands | pLDDT | 89.24 | 89.24 | 0.003 | ✅ |
| | pTM | 0.864 | 0.871 | 0.007 | ✅ |
| | ipTM | 0.850 | 0.854 | 0.005 | ✅ |
| | ranking_score | 0.852 | 0.857 | 0.005 | ✅ |

**精度结论**：两个案例全部指标误差均在 1e-2 ~ 1e-3 量级，属 bf16 数值噪声正常范围，精度对齐通过。

### 7.2 性能对比

| 案例 | N_atom | CUDA 耗时 | SDAA 耗时 | 性能比（SDAA/CUDA） |
|------|-------|-------------|-------------|--------------------|
| protein_dna | 2529 | 18.27s | 543.27s | ~29.7× |
| multimer_with_ligands | 3596 | 36.84s | 1437.52s | ~39.0× |

---

## 附：验证记录

在 SDAA 环境下执行：

**静态验证**

- 全部改动文件 `py_compile` 通过；
- `torch_backend` 正确识别 `sdaa`（4 卡、`tccl`，`get_device_name()=/dev/tcaicard0`）；
- `layer_norm` 走纯 torch（设置 `LAYERNORM_TYPE=torch` 后 `fastln_is_installed=False`）；
- `tri_attention` 回退为 `TRITON_AVAILABLE=False`（SDPA 兜底）；
- 主模型 `from protenix.model.protenix import Protenix` 导入成功。

**适配命令**

```bash
LAYERNORM_TYPE=torch protenix pred \
    -i <input.json> \
    -o <output_dir> \
    -n protenix_base_20250630_v1.0.0 \
    --triatt_kernel torch --trimul_kernel torch \
    --use_msa False
```

> 如需 MSA：先执行 `protenix msa --input <input.json> --out_dir <out> --msa_server_mode protenix`，
> 再以 `--use_msa True` 预测。
