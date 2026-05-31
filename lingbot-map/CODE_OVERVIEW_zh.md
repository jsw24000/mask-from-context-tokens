# LingBot-Map 代码导览

这份导览帮助你快速看懂项目的主要文件和运行链路。项目整体是一个基于 PyTorch 的流式 3D 重建模型：输入一串图片或视频帧，模型预测相机位姿、深度图和点云，再通过浏览器可视化或 benchmark 评测。

## 最常用入口

- `demo.py`
  - 交互式 demo 主入口。
  - 负责读取图片/视频、加载 checkpoint、选择 `streaming` 或 `windowed` 推理模式、后处理位姿/深度/点云，并启动 `viser` 三维可视化。
  - 初次读代码建议从 `main()` 开始，再看 `load_images()`、`load_model()`、`postprocess()`。

- `gct_profile.py`
  - 性能 profiling 脚本。
  - 用来比较 FlashInfer、SDPA、`torch.compile` 等配置下的推理速度和显存占用。

- `pyproject.toml`
  - Python 包配置。
  - 声明包名、依赖、可选可视化依赖，以及 setuptools 如何发现 `lingbot_map*` 包。

- `README.md`
  - 项目说明、安装方式、模型下载和 demo 命令。
  - 当前文件在 Windows 控制台里可能显示乱码，但原始内容主要是英文 README 和命令示例。

## 核心模型代码

- `lingbot_map/models/gct_base.py`
  - 模型基类，定义通用 forward 流程。
  - 主要职责是把输入图像送入 aggregator 提取 token，再调用 camera/depth/point head 输出预测结果。
  - 子类需要实现 `_build_aggregator()` 和 `_build_camera_head()`。

- `lingbot_map/models/gct_stream.py`
  - 标准流式推理模型。
  - `inference_streaming()` 是关键函数：先用前若干帧估计全局尺度，再逐帧处理后续图像，并通过 KV cache 避免重复计算历史帧。

- `lingbot_map/models/gct_stream_window.py`
  - 长序列窗口化推理版本。
  - 适合几千帧甚至更长的视频，把序列切成带 overlap 的窗口，降低单次推理的显存压力和漂移风险。

- `lingbot_map/aggregator/base.py`
  - aggregator 基类。
  - 管理 patch embedding、frame blocks、global blocks、特殊 token 等通用结构。

- `lingbot_map/aggregator/stream.py`
  - 流式 aggregator。
  - 核心是因果时序注意力和 KV cache：当前帧只看过去帧，历史帧的 key/value 被缓存起来。

## 网络组件

- `lingbot_map/layers/`
  - Transformer/ViT 基础层。
  - `attention.py` 包含普通注意力、因果注意力、FlashInfer/SDPA 后端。
  - `flashinfer_cache.py` 管理 FlashInfer paged KV cache。
  - `block.py` 是 Transformer block 的组合。
  - `rope.py` 实现二维/三维 RoPE 位置编码。
  - `patch_embed.py`、`vision_transformer.py` 处理图像 patch 特征。

- `lingbot_map/heads/camera_head.py`
  - 相机位姿预测头。
  - 从 camera token 预测 9 维 pose encoding，内部有迭代 refinement。

- `lingbot_map/heads/dpt_head.py`
  - DPT 风格的 dense prediction head。
  - 用于输出深度图、世界点云或局部相机坐标点。

- `lingbot_map/heads/head_act.py`
  - 对 head 输出做激活，例如深度、置信度、位姿参数的数值范围处理。

## 工具与可视化

- `lingbot_map/utils/load_fn.py`
  - 图片读取、EXIF 方向修正、resize/crop/pad、转 tensor。

- `lingbot_map/utils/pose_enc.py`
  - pose encoding 和相机外参/内参之间的转换。

- `lingbot_map/utils/geometry.py`
  - SE(3) 变换、投影/反投影、点云和相机几何工具。

- `lingbot_map/vis/point_cloud_viewer.py`
  - 基于 `viser` 的点云/相机轨迹浏览器。

- `lingbot_map/vis/sky_segmentation.py`
  - 天空分割与天空 mask 缓存，主要用于室外场景过滤天空点。

- `lingbot_map/vis/glb_export.py`
  - 把预测结果导出成 GLB 场景。

## Benchmark 评测

- `benchmark/run.py`
  - 运行 benchmark 的入口。
  - 从配置文件读取数据集和方法，逐场景执行并保存预测结果。

- `benchmark/evaluate.py`
  - 对预测结果计算指标。
  - 包括轨迹、深度、点云等评测。

- `benchmark/prepare.py`
  - 数据准备入口。
  - 把原始数据集整理成 benchmark 内部统一的 BSS 存储格式。

- `benchmark/methods/lingbot_map.py`
  - 把 LingBot-Map 包装成 benchmark method。
  - 负责加载模型、运行推理，并把输出转换成 benchmark 需要的 RGB、depth、pose、intrinsics、confidence 列表。

- `benchmark/datasets/`
  - 各数据集适配器，例如 KITTI、Oxford Spires、Droid-W、VBR、7-Scenes。

- `benchmark/benchmark/core/`
  - benchmark 框架核心：配置、动态加载、存储、读取、保存、评估调度。

## 一条典型执行链

1. `demo.py main()` 解析命令行参数。
2. `load_images()` 读取图片或抽取视频帧，并统一预处理成 `[S, 3, H, W]`。
3. `load_model()` 创建 `GCTStream`，加载 checkpoint。
4. `GCTStream.inference_streaming()` 先处理 scale frames，再逐帧推理。
5. `GCTBase.forward()` 调用 aggregator 提取 token，再调用 camera/depth/point heads。
6. `postprocess()` 把 pose encoding 转成相机矩阵，并把结果搬到 CPU。
7. `PointCloudViewer` 在浏览器里显示点云、相机轨迹和图像。

