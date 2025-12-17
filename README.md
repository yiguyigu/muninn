# MUNINN

# <Your Project Name>

[English](#overview-english) | [中文](#中文说明-chinese)

## Overview (English)

This repository showcases an AI deployment pipeline for UAV (drone) target detection, focusing on **model versioning and edge inference engineering** rather than only model accuracy.

Multiple YOLO-based models are managed via Git branches, providing clear isolation between implementations and making it easy to evolve or roll back model versions over time. The current deployment target is an NVIDIA Jetson Orin Nano–class edge GPU, but the structure is designed to be portable to other devices.

Current model branches:

- [`model/yolov5-7.0`](../../tree/model/yolov5-7.0) – YOLOv5 v7.0 integration for UAV target detection  
- [`model/ultralytics-8.1.0`](../../tree/model/ultralytics-8.1.0) – `ultralytics==8.1.0` integration (YOLOv8–YOLOv11 family) for UAV target detection  

## 中文说明 (Chinese)

本项目面向无人机（UAV）目标检测场景，重点展示 **AI 模型版本管理与边缘推理部署** 的工程实践，而不仅仅关注检测精度本身。

通过 Git 分支对不同 YOLO 实现与版本进行管理，在统一代码框架下实现：

- 不同模型实现与版本的隔离与可溯源管理  
- 在 UAV 任务中可方便切换 YOLOv5 与 Ultralytics YOLO  
- 面向资源受限 GPU（当前主要为 NVIDIA Jetson Orin Nano）的边缘推理配置与工程化结构，便于后续扩展与优化  

当前主要模型分支：

- `model/yolov5-7.0` – 基于 YOLOv5 v7.0 的 UAV 目标检测集成  
- `model/ultralytics-8.1.0` – 基于 `ultralytics==8.1.0`（覆盖 YOLOv8–YOLOv11 系列）的 UAV 目标检测集成  