# Particle Trajectory Visualization｜颗粒轨迹识别与可视化

高速摄像可以记录颗粒从表面起跳、迁移到回落的过程，但单帧图像只能给出某一时刻的位置。要比较颗粒的起跳高度、水平位移和运动速度，需要先在连续帧中识别颗粒，再判断不同帧中的检测结果是否属于同一颗粒。画面里同时出现多颗粒、背景纹理或短时漏检时，逐帧手工追踪既耗时，也不容易在多个实验片段之间保持一致。

本项目面向电动力除尘（EDS）等颗粒运动视频的图像序列分析。它从指定的观察区域提取候选颗粒，支持自动选择或人工指定追踪起点，结合 Kalman 短期预测和局部抛物线约束连接轨迹，随后筛除不符合设定条件的轨迹。输出既包括叠加在原图上的轨迹，也包括逐帧坐标和每条轨迹的统计量。单段图像序列可以独立处理；按压强和片段组织的数据也可以批量运行。

![合成图像序列的轨迹叠加结果](examples/expected/trajectory_overlay_no_labels.png)

上图使用程序生成的两条模拟轨迹，用来演示输入、追踪和绘图流程，不用于评价真实实验中的识别精度。

## 运行示例

在项目目录执行：

```powershell
python -m pip install -r requirements.txt
python examples/generate_demo_frames.py
python main.py --config config.demo.yaml
```

生成器会建立一组按时间排序的图像帧。单段结果位于 `demo_results/single/`：`trajectory_overlay_no_labels.png` 是轨迹叠加图，`EDS_motion_points.csv` 记录逐帧位置与运动量，`EDS_track_summary.csv` 汇总每条有效轨迹的起跳高度、水平位移等指标。仓库中的 `examples/expected/` 保存了这组示例的预览图和汇总表。

批处理可先检查将要处理的片段，再正式运行：

```powershell
python batch_main.py --batch_config batch_config.demo.yaml --discover_only
python batch_main.py --batch_config batch_config.demo.yaml
```

批处理按 `压强文件夹/segments/片段文件夹/frames_enhanced/` 查找图像帧，并将跨片段汇总写入 `demo_results/batch/batch_summary/`。示例配置只有一个模拟压强组，只用于检查批处理流程。

## 使用实验图像

将同一片段的 PNG/JPG 图像帧放在一个文件夹中，保证文件名顺序与采集时间一致、所有帧尺寸相同。复制 `config.demo.yaml` 为实验配置，至少核对以下参数：

| 参数 | 含义 |
|---|---|
| `input_dir`、`output_dir` | 图像帧位置和结果目录 |
| `fps` | 实际采集帧率；若视频经过抽帧，应填写抽帧后的帧率 |
| `pixel_size_mm` | 由比例尺标定的每像素实际长度 |
| `roi_points_px`、`wall_z_px` | 观察区域和壁面在图像中的位置 |
| 识别阈值与追踪门限 | 依据颗粒亮度、尺寸和帧间位移调整 |

配置完成后运行 `python main.py --config 你的配置.yaml`。视频文件需先抽取为连续图像帧。若需要估算与颗粒质量有关的量，还应填写粒径和密度；按工况比较时，应同时记录压强、组别及实验重复信息。

帧率和像素比例尺会直接影响时间、速度与距离。反光、遮挡、颗粒交叉以及壁面位置误差也可能造成误连或错误筛选。建议从每组实验抽取少量片段进行人工核对，再使用批处理结果做统计。轨迹描述的是颗粒运动，不能单独证明除尘效率。
