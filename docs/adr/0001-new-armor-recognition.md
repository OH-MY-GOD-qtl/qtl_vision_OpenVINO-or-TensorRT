# ADR-0001：新装甲识别方案（无灯条、中央图案发光）

- **状态**：Proposed（待装甲板实物与规则手册确认）
- **日期**：2026-10-27
- **决策者**：项目维护者
- **相关代码**：`include/armor.hpp`、`include/detector/`、`src/detector/`、`src/solver.cpp`、`src/tracker/`、`configs/`

---

## 1. 背景

新装甲板取消了两侧灯条，改为**中央图案发光**。当前没有实物、没有尺寸、没有图案/类别表，也没有标注数据。

现有链路的前提与新装甲不匹配：

- 传统链路 `Detector::detect` 依赖 `ChannelThresholder → extract_lightbars → Armor(left, right)` 的双灯条几何配对。没有灯条后该前提消失，调参无法恢复。
- 旧 YOLO 模型（`assets/0526.onnx`、`yolov8.xml`、`yolo11.xml`）学习的是旧装甲的灯条 + 板面特征，不能复用。
- 数字分类器 `assets/tiny_resnet.onnx` 基于旧图案，同样不能复用。
- 目标是不动下游主链路：`Armor → Solver(PnP) → Tracker → Aimer`，只替换检测前端。

---

## 2. 决策状态

### 2.1 已确认

| # | 决策 | 依据 |
|---|---|---|
| D1 | 新装甲走 YOLO-only，传统链路只服务旧装甲 | 灯条消失，传统几何配对结构失效 |
| D2 | 模型关键点使用**装甲板四角** | 维护者确认；待 §4 可标注性验证 |
| D3 | 最终类别语义沿用 `armor_properties` 的 38 类（颜色 × 名称 × 大小板） | 维护者确认“沿用” |
| D4 | 大小板判定沿用现有规则（`armor_properties` 的 `type`；`get_type_neural` 不动） | 维护者确认“沿用” |
| D5 | 传统链路保留，不删除；新旧装甲用配置隔离 | 维护者确认“兼容，先不删” |
| D6 | 当前没有标注数据 | 维护者确认 |
| D7 | 新配置固定 YOLO11-pose + OpenVINO 640（`yolo_name: yolo11`、`use_tensorrt: false`）；旧配置 `standard1.yaml` 保持 `use_tensorrt: true` | 维护者确认 A |
| D8 | 新旧配置隔离：复制 `standard1.yaml` → `new_armor.yaml`，只改模型相关键，传统键原样保留 | 维护者确认“是” |
| D9 | `Armor::priority` 默认初始化为 `ArmorPriority::fifth`；omni 的 `Decider::set_priority` 仍会覆盖 | 维护者确认 A |

### 2.2 待确认/暂缓

| # | 建议 | 状态 |
|---|---|---|
| P1 | 第一版先训 10 类（红/蓝 × 一至五号）打通链路，38 类作为第二阶段 | 待确认（依赖 U4/U6） |
| P4 | 先写离线预标注器：`ChannelThresholder` 找发光图案 → 轮廓 → `minAreaRect`/`approxPolyDP` → YOLO-pose txt，再人工修 | 暂缓（依赖 U3 采图） |

---

## 3. 影响与必须同步的契约

| # | 位置 | 现状 | 新方案要求 |
|---|---|---|---|
| C1 | `src/solver.cpp:11-13` | 物点为「板宽 × 56 mm 灯条长」：`LIGHTBAR_LENGTH = 56e-3`、`BIG_ARMOR_WIDTH = 230e-3`、`SMALL_ARMOR_WIDTH = 135e-3` | 板四角关键点对应的物点应改为「板宽 × 板高」。若板高仍为 126 mm 而不改 56 mm，PnP 距离会按 56/126 ≈ 0.44 倍被低估（约偏近 2.25 倍） |
| C2 | `include/detector/yolos/yolo11.hpp:27` | `class_num_ = 38` 硬编码 | 必须等于训练模型的实际类别数；它同时决定关键点在输出张量中的偏移 `(4 + class_num_) * anchor_num`，不一致会读错关键点 |
| C3 | `src/detector/yolos/yolo11.cpp:34` | 输入固定 640×640；解析按 feature-major `[1][features][anchors]`、4 个关键点 | 导出模型必须保持该布局与关键点数量，否则需改解析器 |
| C4 | `include/armor.hpp:44` | `armor_properties` 共 38 项，`class_id` 直接做下标 | 子集模型需要显式 `class_map`；例如 10 类步兵在 38 类表中的下标并不连续（3,4,6,7,9,10,12,13,15,16），不能假设 0..9 |
| C5 | `src/detector/yolos/yolo11.cpp:14` + `src/detector/detector.cpp:21` | `YOLO11` 内部构造 `Detector(config_path, false)`，默认 `use_classifier=true`，会加载 `classify_model` | 新配置缺 `classify_model` 会因 `yaml_read` 直接 `exit(1)`；YOLO 路径实际不用分类器，可改为不加载 |
| C6 | `include/armor.hpp:73` + `src/armor.cpp` | `ArmorPriority priority` 未初始化；`src/tracker/tracker.cpp:72-73,136-137` 却按它 `stable_sort` | 必须先初始化，否则新链路第一次选目标就是未定义行为 |
| C7 | `src/tracker/tracker.cpp:286,294` | 目标匹配条件为 `armor.name` + `armor.type` | 38 类直接分类时，远距离编号抖动可能导致丢目标；需用数据验证，必要时放宽为颜色 + 大小板匹配 |
| C8 | `src/detector/yolos/yolov5.cpp:242` | `use_traditional=true` 时用传统灯条在 ROI 内二次矫正关键点 | 新装甲配置必须关闭，否则可能把发光图案误当灯条覆盖关键点 |
| C9 | `assets/yolov5.engine` | TensorRT 引擎由旧 `0526.onnx` 导出 | 新模型必须重新导出，或改走 OpenVINO；旧引擎不可复用 |
| C10 | `include/detector/detector.cpp:284,330` | `get_pattern_neural` / `get_pattern` 的 `1.125 = 126 mm / (2 × 56 mm)` 是灯条延伸到板面的系数 | YOLO11/8 链路未调用；若复用裁剪图案，必须换成新板高/图案高比例 |

> 本表记录改造前基线。本次无硬件改造落地后，以代码为准。

---

## 4. 未知项（阻塞）

| # | 未知项 | 为什么阻塞 | 来源 | 验证方式 |
|---|---|---|---|---|
| U1 | 大/小板外框宽 × 高（mm） | 决定 PnP 物点，直接影响测距 | 规则手册 / 机械组 / CAD | 实物测量 + 卷尺/标定板交叉校验 |
| U2 | 发光图案区宽 × 高（mm） | 若板角不可标注，回退图案角时决定物点 | 同上 | 实物测量 |
| U3 | 板角在 3/5/10 m 是否可稳定标注 | 决定 D2 是否成立 | 现场采图 | 10 张远/近图，两人各标一次，比较角点方差与重投影误差 |
| U4 | 图案/编号/颜色/兵种对照表 | 决定 38 类表是否仍成立 | 规则手册 | 与 `armor_properties` 逐项对齐 |
| U5 | 最低识别距离、曝光/增益可用范围 | 决定 `min_confidence`、`score_threshold` 与光学方案 | 现场实测 | 不同距离/曝光下的检出率与角点误差 |
| U6 | 数据量与可投入的标注人力 | 决定第一版类别范围（P1） | 项目内部 | 预标注吞吐评估 |

> **禁止**：在 U1/U2 未确认前，把猜测尺寸写入 `configs/*.yaml` 并上机。占位值只能存在于本 ADR 或测试夹具中。

---

## 5. 迁移计划

### 5.1 无硬件即可做（本次范围）

1. 新增 `configs/new_armor.yaml` 骨架：复制 `standard1.yaml`，改模型键，传统键保留；模型路径为占位，文件头注明“暂不可运行”。
2. `YOLO11` 解析器去掉 `class_num = 38` 硬编码：优先按输出特征数自动推断（`features - 12`），可选 `class_num` 覆盖；新增可选 `class_map`（模型类别 → `armor_properties` 下标），缺省 identity。
3. `YOLO11` 不再构造带分类器的 `Detector`：改传 `use_classifier=false`，避免加载 `tiny_resnet.onnx`。
4. `Armor::priority` 默认初始化为 `ArmorPriority::fifth`，消除 `Tracker` 排序的未定义行为。

#### `class_map` 约定

- 一维数组，长度必须等于模型类别数；`class_map[model_class_id]` 是 `armor_properties` 的下标。
- 不写该键 = identity（模型类别 id 直接作为 38 类表下标）。
- 第一版 10 类步兵示例（待 P1 确认）：`[3, 4, 6, 7, 9, 10, 12, 13, 15, 16]`，对应蓝/红 1–5 号。

### 5.2 后续（依赖硬件/数据）

1. 预标注工具骨架（依赖 U3 采图方案）。
2. 单元测试：`class_map` 映射、PnP 物点、`priority` 排序、关键点顺序。
3. 第一版训练范围（P1）在 U4/U6 明确后确认。

### 5.3 硬件到货后

1. 测量 U1/U2，填 `new_armor.yaml`。
2. 采图，做 U3 可标注性验证；板角不稳则回退图案角并改物点。
3. 预标注 → 人工修 → 训练第一版模型。
4. 按 §6 验证；通过后再扩 38 类。
5. 旧装甲回归：`configs/standard1.yaml` 仍能跑传统链路。

---

## 6. 验收指标（已确认）

| 指标 | 目标 | 方法 |
|---|---|---|
| PnP 距离误差 | 3–8 m 内 < 5% | 已知距离靶标，与卷尺/激光测距对比 |
| 关键点重投影误差 | < 5 px | `solvePnP` 后 `projectPoints` 回投 |
| 跟踪连续性 | 连续 10 s 不误切换/不丢目标 | 实车或视频回放 |
| 编号抖动率 | 同一目标连续帧 `name` 一致率 > 95% | 录包统计 |
| 远距离检出 | 按队伍需求定义最小距离 | 不同距离检出率曲线 |

---

## 7. 给机械/规则组的索取清单

1. 大/小板外框宽 × 高，发光图案区宽 × 高（mm，含公差）。
2. 板角是否有可见边界或材质差异（决定四角可标注性）。
3. 图案/编号/颜色/兵种对照表。
4. 是否有 3D CAD / STEP 模型。
5. 新旧装甲是否同场使用（决定传统链路保留周期）。

---

## 8. 术语

- **灯条链路**：旧传统检测链路，依赖两条发光灯条 + 几何配对。
- **板角**：装甲板外框的四个角点。
- **图案角**：中央发光图案的四个角点。
- **物点**：传给 `cv::solvePnP` 的装甲板三维点，单位 m。
- **大小板**：`ArmorType::big` / `ArmorType::small`，决定 PnP 物点与跟踪参数。

---

## 9. 相关文件

- `src/solver.cpp`：PnP 物点与尺寸常量
- `src/detector/detector.cpp`：传统检测、类型判定、图案裁剪
- `src/detector/yolos/yolo11.cpp`、`yolov5.cpp`：YOLO 解析与传统二次矫正
- `src/tracker/tracker.cpp`：目标选择与 `priority` 排序
- `include/armor.hpp`：38 类表与 `Armor` 数据模型
- `configs/standard1.yaml`：旧装甲配置基线
