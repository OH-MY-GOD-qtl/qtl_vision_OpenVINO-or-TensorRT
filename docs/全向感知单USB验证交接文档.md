# 全向感知（单 USB + 海康）验证交接文档

> **日期**：2026-10-14 晚
> **状态**：验证已完成关键结论；`configs/omni.yaml` 主配置**尚未修改**；代码未改动
> **关联**：`README.md`「全向感知」章节；`docs/隐患排查交接文档.md`（其风险 #3、#5 与本文直接相关）
> **标记约定**：✅ 已实证 ｜ ⏸ 待办 ｜ 🔍 存疑待复核
> **注意**：`logs/` 下日志文件名的时间戳是 **UTC**（比北京时间慢 8 小时），例如 `14-30-13` 对应本地 `22:30:13`。

---

## 0. 一句话结论

**"海康有了、USB 只有一个，能识别到吗" —— 已实证：能。**
项目代码可正常打开这台唯一的 USB 相机（`/dev/video0`）并稳定出帧；「1 路 USB + 海康」的 omni 全链路（模拟模式）可跑通，也能干净退出。
**但**当前 `omni.yaml` 仍按 3 路 USB 配置，缺失的 `video2/video4` 会导致**程序退出时卡死**（已实证，机理见 §2.4）。明天只需按 §5 处理配置（或加个小代码加固）即可。

---

## 1. 硬件与设备现状（2026-10-14）

| 设备 | lsusb | 设备节点 | 备注 |
|---|---|---|---|
| 海康 MV-CS016-10UC | `2bdf:0001` | — | 已接入，出图正常 |
| USB 相机 Microdia HD720p | `0c45:6341` | `/dev/video0`（采集）、`/dev/video1`（metadata，不能用于采集） | WSL2 经 usbipd 透传 |

- 目前只有 1 路 USB 相机；全向方案目标为 3 路（设备名 `video0/video2/video4` 分别对应 usb0/1/2）。
- 权限已配好（用户已在 video 组）；海康 udev 规则按 README。
- 该 USB 相机在全向方案里只做传统灯条检测（不识别装甲板编号），1 路已够做"方位搜索"链路测试。

---

## 2. 已完成实证（逐条）

### 2.1 项目代码能打开这台 USB 相机 ✅

- **命令**：`timeout 6 ./build/usbcamera_test configs/omni.yaml video0`
- **结果**：`right USBCamera opened`（代码按 sharpness 判定命名为 "right"）、`USBCamera fps:30`、采集线程启动、持续出帧（读侧统计 10~51 fps）。
- **注意**：偶发 `Corrupt JPEG data: 80 extraneous bytes...` 是该相机 MJPEG 流的老毛病（当年 ffmpeg 抓帧也复现过），**不是故障**。
- **证据**：`logs/2026-10-14_14-20-57.960900915.log`

### 2.2 缺失设备（video2）时的行为 ✅

- **命令**：`timeout 5 ./build/usbcamera_test configs/omni.yaml video2`
- **结果**：每 ~100ms 报一次 `Failed to open USB camera`，约 2 秒后打印 `Give up to open USB camera`（重连上限 20 次，即隐患排查文档风险 #3）；**之后进程不再有任何输出——`read()` 永久阻塞在空队列上**。
- **证据**：`logs/2026-10-14_14-21-20.711584420.log`

### 2.3 omni 全链路运行情况 ✅

- **3 路配置 + 只接 1 路**：程序**运行正常**——usb0 在跑传统检测（日志可见 `[Detector] contours=...`）、海康 YOLO（TensorRT）在跑、主循环正常。启动约 2 秒时出现两条 `Give up to open USB camera`。
- **1 路配置**：启动正常，另有两条 `USB camera pointer is null (usb1)!` / `(usb2)!` —— error 级日志但**属预期**（代码对空指针直接跳过该路，见 §2.5）。
- **证据**：`logs/2026-10-14_14-23-32.405312105.log`（3 路）、`logs/2026-10-14_14-28-23.332219621.log`（1 路）

### 2.4 退出行为对照实验（本次关键产出）✅

控制变量实验：直接 `kill -INT <pid>`（等价于终端 Ctrl+C），观察退出过程：

| 实验 | 配置 | 结果 |
|---|---|---|
| **D** | 1 路：`omni_usb_devices: ["video0"]`、`omni_usb_yaw_offsets: [60]` | SIGINT 后 **~90ms 干净退出**，析构日志齐全：`[omni_standard] exit` → `Perceptron destructed.` → `USBCamera destructed.` → `HikRobot destructed.` ✅ |
| **E** | 3 路：原配置（缺 video2/video4） | `[omni_standard] exit` 后**卡死**：无任何析构日志、进程不自行退出，需强杀收尾 ❌ |

- **证据**：D → `logs/2026-10-14_14-30-09.823385695.log`（31 行）；E → `logs/2026-10-14_14-30-13.380115720.log`（86 行，最后一行即 `[omni_standard] exit`）
- 另一次 timeout 包装的复现：卡死 8 秒后被强杀（exit=137），特征同上 → `logs/2026-10-14_14-25-38.900990677.log`

**卡死机理（代码级，明天改代码前先读）**：

1. 配置了不存在的设备 → `USBCamera` 守护线程重试 20 次后放弃（`src/camera/usbcamera.cpp:31`），**采集线程从未启动**；
2. `Perceptron::search_thread`（`src/omni/perceptron.cpp`）对该路调用 `cam->read(img, ts)` → `queue_.pop()` **永久阻塞**（队列无人 push；`include/tools/thread_safe_queue.hpp` 的 `pop` 是无超时等待）；
3. 退出时 `~Perceptron` 依次 `join()` 3 个搜索线程 → 卡在第 2 路上 → 主线程析构流程死锁，程序无法退出。
4. **逃生手段**：再按一次 Ctrl+C 会强制终止（`src/tools/exiter.cpp` 的二次信号设计，现成可用）。
5. 隐患类比：WSL 里 usbipd **运行中掉线** 也会走同样的路径（采集线程退出→队列无人 push→卡死），所以这个坑不只是"配置多写了设备"。

### 2.5 顺带确认的代码事实（接手备考）

- `apps/omni_standard.cpp:44-49`：从 yaml 读 `omni_usb_devices`；只创建 yaml 中给出的路数，其余保持 `nullptr`。
- `Perceptron` 构造函数（`src/omni/perceptron.cpp:9`）：对 `nullptr` 相机，`search_thread` 打印 error 后直接 `return` → **1 路配置是安全的**（D 实验已验证）。
- `Decider`（`src/omni/decider.cpp`）：`omni_*` 键缺失有默认值；`delta_angle` 对安装角索引越界有保护 → 只配 1 个 `omni_usb_yaw_offsets` 没问题。
- 日志双落盘：stdout + `logs/YYYY-MM-DD_HH-MM-SS.log`（**文件名是 UTC 时间，慢 8 小时**，别被文件名搞混）。
- 代码里 USB 相机的 "left/right" 命名（按 sharpness 判定）与全向的 `usb0/1/2` 无关；**`omni_usb_devices` 数组顺序**才决定 usb0/1/2（与 `omni_usb_yaw_offsets` 索引一一对应）。

---

## 3. 当前配置状态（重要）

- `configs/omni.yaml` **尚未修改**，仍是 3 路：
  - `omni_usb_devices: ["video0", "video2", "video4"]`（第 102 行）
  - `omni_usb_yaw_offsets: [60, -60, -180]`（第 103 行）
- 已验证的临时配置已存档（模拟模式，可直接跑）：
  - `logs/omni_verify_20261014/omni_sim_1dev.yaml` —— 1 路版（D 实验同款，**可干净退出**）
  - `logs/omni_verify_20261014/omni_sim_3dev.yaml` —— 3 路版（用于复现卡死）

---

## 4. 明天的最短路径（TL;DR）

1. 想先跑起来 → 直接用 `logs/omni_verify_20261014/omni_sim_1dev.yaml` 或按下文把 `omni.yaml` 改成 1 路（§5 第 1 条）；
2. 想根治退出卡死 → 加一个小改动（§5 第 2 条，约 20 行代码）；
3. 实机标定安装角/FOV 不急，先把链路跑通。

---

## 5. 明天的 TODO（按优先级）

1. ⏸ **决定 `omni.yaml` 的处置**（二选一）：
   - **方案 A（最简，当天可用）**：改为 1 路
     ```yaml
     omni_usb_devices: ["video0"]
     omni_usb_yaw_offsets: [60]   # 按该 USB 相机实际安装方向改！60 是旧 3 路方案 usb0 的值
     ```
   - **方案 B（推荐，治本）**：保留 3 路不变，做第 2 条的代码加固。
2. ⏸（推荐，治本）**代码加固：消灭"缺设备/运行中拔线 → 退出卡死"**：
   - 给 `USBCamera` 增加带超时的读取接口，例如 `bool try_read(img, ts, timeout_ms)`（内部用现成的 `ThreadSafeQueue::try_pop`，不用新造轮子）；
   - `Perceptron::search_thread` 改用该接口：超时（如 100ms）就 `continue` 回到循环顶部检查 `stop_flag_`；
   - 效果：相机缺失、usbipd 掉线、拔插都不用强杀。
   - 改完重新编译，用 §6 复测 D/E 两场景（期望：两种配置都能干净退出；3 路版启动后应能正常退出、并为缺失路报"暂不可用"）。
   - 该问题与隐患排查文档**风险 #3**（USB 重连计数不重置）同源，建议顺手在其文档里补一句"放弃后 `read` 永久阻塞 → 退出卡死"。
3. ⏸ **实机项**：`omni_usb_fov_h/v`（当前 37.0/40.9）是旧相机参数，需按这台 Microdia 实测 FOV 替换；安装角按实际装配调整。
4. ⏸ **补齐 3 路 USB 相机后**：恢复 3 路配置整体验证（重点核跨相机引导 yaw 的方向正确性）。
5. 🔍 **低优先存疑（仅影响测试手段，不影响使用）**：用 `timeout` 命令包装运行时，观察到过不一致的退出行为（有时无日志快速被终止）。直接 `kill -INT`/终端 Ctrl+C 行为正常（D/E 即是这样测的）。怀疑与 `timeout` 对子进程组的信号投递方式有关；如再遇到，优先用"手工 `kill -INT` + 看日志"的方式测试。

---

## 6. 复现命令备忘（拷贝即用）

```bash
cd /home/qtl/qtl_vision_OpenVINO-or-TensorRT

# ① 验证 USB 相机出图（6 秒自动结束；加 -d 可看窗口，q 退出）
timeout 6 ./build/usbcamera_test configs/omni.yaml video0

# ② 一键生成 1 路模拟配置（与已验证版一致）
sed 's|\["video0", "video2", "video4"\]|["video0"]|; s|\[60, -60, -180\]|[60]|; s|^simulate: false|simulate: true|' configs/omni.yaml > /tmp/omni_sim1.yaml

# ③ 跑全向感知（模拟模式，不打开串口；退出按 Ctrl+C，若卡住再按一次）
./build/omni_standard /tmp/omni_sim1.yaml

# ④ 受控实验（观察退出是否干净）：
#    先启动程序，另开终端找到 pid，然后：
#    kill -INT <pid>     # 等价 Ctrl+C
#    观察：干净退出 = 出现 Perceptron/USBCamera/HikRobot destructed 日志
```

---

## 7. 证据文件清单

| 文件 | 内容 | 对应实验 |
|---|---|---|
| `logs/2026-10-14_14-20-57.960900915.log` | USB 相机打开成功、出帧 | §2.1 |
| `logs/2026-10-14_14-21-20.711584420.log` | video2 打开失败 20 次后放弃、之后阻塞 | §2.2 |
| `logs/2026-10-14_14-23-32.405312105.log` | 3 路配置全链路运行（timeout 场景） | §2.3 |
| `logs/2026-10-14_14-25-38.900990677.log` | 3 路配置卡死（8s 后强杀，exit=137） | §2.4 |
| `logs/2026-10-14_14-28-23.332219621.log` | 1 路配置启动（null 日志 ×2） | §2.3 |
| `logs/2026-10-14_14-30-09.823385695.log` | **D：1 路配置干净退出（关键）** | §2.4 |
| `logs/2026-10-14_14-30-13.380115720.log` | **E：3 路配置退出卡死（关键）** | §2.4 |
| `logs/omni_verify_20261014/omni_sim_1dev.yaml` | 已验证的 1 路模拟配置 | §3 |
| `logs/omni_verify_20261014/omni_sim_3dev.yaml` | 3 路模拟配置（复现卡死用） | §3 |

---

## 8. 遗留疑问（记录备查，不影响明天开工）

- 🔍 `timeout` 包装下退出行为的偶发差异（§5.5）——实际使用（终端 Ctrl+C）不受影响。
- 卡死期间海康守护线程曾自行重启了一次采集线程（`HikRobot's capture thread started/stopped`），属旁路现象，暂不深究。
