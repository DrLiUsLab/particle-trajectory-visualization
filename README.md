# Particle Trajectory Visualization｜颗粒轨迹识别与可视化

Automatic multi-particle tracking from image sequences, with trajectory overlays and batch statistics.

从连续图像帧自动识别暗色颗粒、连接多条轨迹，输出轨迹叠加图、逐点数据及起跳高度、水平位移、速度等统计量。支持单段分析和按压强批处理。核心代码取自原代码库的 `particle_tracker_v1` V3.1；这里清理了示例配置，并用可重复生成的合成帧验证运行流程。

![合成示例的自动追踪结果](examples/expected/trajectory_overlay_no_labels.png)

**示例中的两条轨迹均为程序合成，只说明软件能跑通，不代表实验观测或算法精度。**

## 快速开始

在项目目录运行：

```powershell
python -m pip install -r requirements.txt
python examples/generate_demo_frames.py
python main.py --config config.demo.yaml
```

结果保存在 `demo_results/single/`。`EDS_track_summary.csv` 每行是一条接受的轨迹；`EDS_motion_points.csv` 是逐帧位置；`trajectory_overlay_no_labels.png` 显示追踪结果。此示例提取出两条完整轨迹，记录在 `examples/expected/` 中。

批处理示例：

```powershell
python batch_main.py --batch_config batch_config.demo.yaml --discover_only
python batch_main.py --batch_config batch_config.demo.yaml
```

批处理按 `压强文件夹/segments/片段文件夹/frames_enhanced/图像帧` 查找数据，并在 `demo_results/batch/batch_summary/` 生成汇总表和压强对比图。演示配置只包含一个合成压强组；它不能用于压强效应推断。

## 处理自己的数据

将按时间排序、尺寸一致的 PNG/JPG 帧放到一个文件夹，复制 `config.demo.yaml` 为自己的配置文件，至少核对 `input_dir`、`output_dir`、`fps`、`pixel_size_mm`、`roi_points_px`、`wall_z_px`、识别阈值和追踪门限。运行 `python main.py --config 你的配置.yaml`。真实视频需要先抽帧；原压缩包中用于交互式选片段的脚本依赖旧电脑的固定路径，未放进这个自动追踪仓库。

`pixel_size_mm` 必须由真实比例尺标定，`fps` 必须是采集或抽帧后的实际帧率；二者错误会直接改变速度和距离。可在 `config.demo.yaml` 中填写颗粒粒径及密度，用于需要这些参数的后续分析。批处理还应填写实验压强和组别。

## 版本取舍

主流程保留原压缩包 `代码库/particle_tracker_v1/particle_tracker_v1` 的自动追踪 V3.1 及批处理入口。人工标注、群体最大起跳包络高度测量、旧版 `01/02/03` 脚本和历史输出没有纳入这个公开仓库；它们仍保存在原压缩包中。示例帧由 `examples/generate_demo_frames.py` 生成，不需要把整批演示帧上传。

## 正式展示还需要

一段允许公开的真实短视频或帧序列、对应帧率、像素到毫米的比例尺、观察区域及壁面位置。建议再给少量人工标注轨迹作误差核验。若要展示能量或不同压强的结果，还需颗粒粒径、密度、压强及实验重复信息。
