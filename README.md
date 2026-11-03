# 智能选片 Smart Photo Selection

> 机器视觉课程大作业 · 期中初步设计报告与原型骨架
>
> 给定同一场景拍摄的一组照片，自动选出最值得保留的一张，并给出更优的取景框。

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

手机连拍和“多拍几张再挑”的习惯让相册里充满近冗余照片。人工选片需要比较清晰度、曝光、构图、表情与瞬间，耗时且主观。本项目把“智能选片”定义为：

- **单图取景优化**：在一张照片内寻找更符合审美的裁剪框；
- **组内相对排序**：在同一场景的 N 张照片中选出最值得保留的一张。

系统采用“候选框生成 → 多线索评价 → 组内排序 → VLM 复核”的分层方案：多尺度/多比例网格与三分法主体锚定生成候选，技术质量、美学、构图与语义四类线索打分，组内归一化后以加权融合或 Bradley–Terry 成对排序模型完成选择，多模态大模型只复核少量高分候选。

## 方法概览

```
组照 ──► 候选框生成 ──► 多线索评价 ──► 组内排序 ──► 最佳照片 + 取景框
         多尺度/多比例      技术质量        组内归一化
         三分法/显著性      美学/构图       加权融合 / LTR
         IoU 去重           VLM 语义        VLM 仲裁
```

| 模块 | 实现 |
|---|---|
| 候选生成 | 4 种画幅比例 × 4 个尺度 × 网格与三分法锚定，贪心 NMS，每图 ≤ 80 个候选 |
| 技术质量 | 清晰度、曝光/截断、对比度、色彩丰富度、噪声 |
| 美学 | 可插拔 `AestheticScorer`；原型提供无依赖的启发式评分 |
| 构图 | 显著性质量占比、三分法对齐、平衡、主体尺度、边界切割、人脸贴边惩罚 |
| 语义 | 可插拔 `VLMRater`；默认中性评分，支持接入本地/云端 VLM |
| 排序 | 组内百分位归一化 + 加权融合；提供 Bradley–Terry 成对排序器 |
| 评测 | 候选召回、Top-1 一致率、成对准确率、NDCG@k、Kendall τ |

## 快速开始

```bash
# 1. 创建虚拟环境（可选）
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# 2. 安装
pip install -e .

# 3. 运行合成演示（无需下载数据）
python -m sps.cli demo --out outputs/demo

# 4. 对真实组照目录选片
python -m sps.cli select --input path/to/series --out outputs/run

# 5. 运行单元测试
python -m unittest discover -s tests -v
```

也可以不安装，直接用 `PYTHONPATH=src python -m sps.cli demo`。

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
│   ├── candidates.py      # 候选取景框生成、去重、可视化
│   ├── scoring.py         # 技术质量、美学、构图评分与融合
│   ├── pipeline.py        # 单图 / 组照推理流程
│   ├── ranking.py         # 组内归一化与 Bradley–Terry 排序器
│   ├── vlm.py             # VLM 评审接口、量表与提示词
│   ├── demo.py            # 合成演示组照生成
│   └── cli.py             # 命令行入口
├── tests/test_smoke.py    # 候选、排序、端到端演示测试
├── scripts/
│   ├── make_figures.py    # 报告图表生成
│   └── build_report.py    # Markdown → Word 报告
├── docs/
│   ├── design-report.md   # 设计报告（源文件）
│   ├── 智能选片设计报告_期中.docx
│   ├── 智能选片设计报告_期中.pdf
│   └── figures/           # 报告插图
├── examples/demo_series/  # 合成演示组照
├── data/                  # 数据放置说明（数据不入库）
└── outputs/               # 运行输出（不入库）
```

## 设计报告

- [Markdown 源文件](docs/design-report.md)
- [Word 版](docs/智能选片设计报告_期中.docx)
- [PDF 版](docs/智能选片设计报告_期中.pdf)

报告包含：问题理解与任务定义、五条技术路线调研（图像美学评价、图像编辑/自动构图、图像偏好学习、排序学习、多模态大模型）、候选框生成/评价/选择的系统方案、数据采集与人工标注计划、实验设计与进度安排。

## 当前状态

- [x] 问题定义与任务形式化
- [x] 相关方法调研与系统方案设计
- [x] 可运行的候选生成、评分、排序流水线
- [x] 组内归一化与 Bradley–Terry 排序器接口
- [x] VLM 评审接口与结构化提示词
- [x] 合成演示、单元测试、设计报告
- [ ] 接入 NIMA / MUSIQ / CLIP-Aesthetic 预训练模型
- [ ] 在 Photo Triage / 自采组照上训练与评测
- [ ] VLM 成对复核与校准
- [ ] 用户研究与个性化反馈

## 数据说明

原型不附带受限数据。完整数据与实验计划见报告第 4 章；本地数据放置方式见 [data/README.md](data/README.md)。演示图像由 `src/sps/demo.py` 程序合成，可自由使用。

## 团队

王颖超（202393005），个人独立完成。工作包划分见报告第 5 章。

## 致谢与参考

本项目的选题与数据设定参考：

> Huiwen Chang, Fisher Yu, Jue Wang, Douglas Ashley, Adam Finkelstein. *Automatic Triage for a Photo Series*. ACM Transactions on Graphics (Proc. SIGGRAPH), 2016.

完整参考文献见 [设计报告](docs/design-report.md#参考文献)。

## License

[MIT](LICENSE)。演示图像为程序合成；报告中引用的数据集与模型遵循其原始许可。
