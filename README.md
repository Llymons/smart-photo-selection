# 智能选片 Smart Photo Selection

> 机器视觉课程大作业 · 智能选片原型
>
> 同一场景拍了一组照片，自动选出最值得保留的一张，再给出一个更合适的取景框。

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

出去拍照时，同一场景常常会连拍好几张，回来再一张张挑，既花时间，也很看当时的心情。这个项目想做的就是把这件小事自动化，具体分两部分：

- **单张照片的取景**：在一张照片里找一个更顺眼的裁剪框；
- **一组照片的排序**：同一场景的 N 张照片中，选出最值得保留的那一张。

做法上分四步：先生成候选框，再给每个候选打分，然后在组内排序，最后按需让多模态大模型复核。候选框来自多种画幅、多个尺度和三分法锚定；打分综合考虑技术质量、美学、构图和语义；组内归一化之后，用加权融合或 Bradley–Terry 成对排序模型做选择。VLM 只用来复核少量高分候选，控制成本，也避免把整个流程拖慢。

## 方法概览

```
组照 ──► 候选框生成 ──► 多线索评价 ──► 组内排序 ──► 最佳照片 + 取景框
         多尺度/多比例      技术质量        组内归一化
         三分法/显著性      美学/构图       加权融合 / LTR
         IoU 去重           VLM 语义        VLM 仲裁
```

| 模块 | 做法 |
|---|---|
| 候选生成 | 4 种画幅比例 × 4 个尺度，配合网格和三分法锚定；贪心 NMS 去重，每张图最多 80 个候选 |
| 技术质量 | 清晰度、曝光和截断、对比度、色彩丰富度、噪声 |
| 美学 | 提供可插拔的 `AestheticScorer`，原型里是一个不依赖额外模型和权重的启发式评分 |
| 构图 | 显著性占比、三分法对齐、画面平衡、主体尺度、边界切割、人脸贴边惩罚 |
| 语义 | 提供可插拔的 `VLMRater`，默认给中性分，也可以接本地或云端 VLM |
| 排序 | 组内百分位归一化后加权融合，另外提供 Bradley–Terry 成对排序器 |
| 评测 | 候选召回、Top-1 一致率、成对准确率、NDCG@k、Kendall τ |

## 快速开始

需要 Python 3.9 以上，依赖只有 numpy 和 opencv-python，`pip install -e .` 会一并装好。

```bash
# 可选：创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装
pip install -e .

# 运行合成演示（不需要额外下载数据）
python -m sps.cli demo --out outputs/demo

# 对真实组照目录选片
python -m sps.cli select --input path/to/series --out outputs/run

# 单元测试
python -m unittest discover -s tests -v
```

不想安装也可以，直接用 `PYTHONPATH=src python -m sps.cli demo`。

### 输出文件

| 文件 | 说明 |
|---|---|
| `selected.jpg` | 最佳照片的最佳取景裁剪 |
| `contact_sheet.jpg` | 全组联系表，标注排名、得分与最佳取景框 |
| `candidates.jpg` | 最佳照片上的候选取景框可视化 |
| `scores.json` | 每张照片、每个候选的分项得分与详细指标 |

## 仓库结构

```
smart-photo-selection/
├── src/sps/
│   ├── __init__.py
│   ├── types.py           # 数据结构
│   ├── image_ops.py       # 图像读取与基础操作
│   ├── candidates.py      # 候选取景框生成、去重、可视化
│   ├── scoring.py         # 技术质量、美学、构图评分与融合
│   ├── ranking.py         # 组内归一化与 Bradley–Terry 排序器
│   ├── pipeline.py        # 单图 / 组照推理流程
│   ├── vlm.py             # VLM 评审接口、量表与提示词
│   ├── demo.py            # 合成演示组照生成
│   └── cli.py             # 命令行入口
├── tests/test_smoke.py    # 候选、排序、端到端演示测试
├── examples/demo_series/  # 合成演示组照
├── data/                  # 数据放置说明（数据不入库）
└── outputs/               # 运行输出（不入库）
```

## 当前状态

- [x] 问题定义与任务形式化
- [x] 相关方法调研与系统方案设计
- [x] 可运行的候选生成、评分、排序流水线
- [x] 组内归一化与 Bradley–Terry 排序器接口
- [x] VLM 评审接口与结构化提示词
- [x] 合成演示与单元测试
- [ ] 接入 NIMA / MUSIQ / CLIP-Aesthetic 预训练模型
- [ ] 在 Photo Triage / 自采组照上训练与评测
- [ ] VLM 成对复核与校准
- [ ] 用户研究与个性化反馈

## 数据说明

原型不附带受限数据。本地数据怎么放见 [data/README.md](data/README.md)。演示图像由 `src/sps/demo.py` 程序合成，可以随意使用。

## 团队

王颖超（202393005），个人独立完成。

## 致谢与参考

本项目的选题和数据设定参考：

> Huiwen Chang, Fisher Yu, Jue Wang, Douglas Ashley, Adam Finkelstein. *Automatic Triage for a Photo Series*. ACM Transactions on Graphics (Proc. SIGGRAPH), 2016.

## License

[MIT](LICENSE)。演示图像为程序合成；项目中引用的数据集和模型遵循其原始许可。
