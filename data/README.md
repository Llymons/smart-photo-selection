# 数据目录说明

本目录用于放置本地数据集，**数据文件不纳入 Git 仓库**（见 `.gitignore`）。

## 目录约定

```
data/
├── raw/
│   └── <series_id>/
│       ├── IMG_0001.jpg
│       ├── IMG_0002.jpg
│       └── ...
├── annotations/
│   ├── pairs.jsonl        # 组内成对偏好
│   ├── best.jsonl         # 组内最佳照片
│   └── crops.jsonl        # 可选：人工裁剪框
└── metadata.csv           # 组号、场景、设备、隐私状态等
```

## 数据来源

| 数据集 | 用途 | 获取方式 |
|---|---|---|
| Photo Triage（Chang et al., 2016） | 主训练/评测集，5,953 组 / 15,545 张 | 按论文作者提供的 benchmark 页面申请或下载 |
| AVA（Murray et al., 2012） | 美学预训练 | 官方数据集页面 |
| GAICD（Zeng et al., 2020） | 候选框召回与裁剪评测 | 官方数据集页面 |
| 自采组照 | 真实手机分布补充 | 按报告 4.2 节方案采集 |

下载后请遵守各数据集原始许可，不要将受限数据提交到公开仓库。

## 标注格式示例

`annotations/pairs.jsonl`（每行一个偏好对）：

```json
{"series": "s0001", "preferred": "IMG_0003.jpg", "other": "IMG_0001.jpg", "annotator": "a01"}
```

`annotations/best.jsonl`：

```json
{"series": "s0001", "best": ["IMG_0003.jpg"], "annotator": "a01"}
```

`annotations/crops.jsonl`：

```json
{"image": "s0001/IMG_0003.jpg", "crop": [x, y, w, h], "annotator": "a01"}
```

## 隐私

自采数据在公开或共享前必须完成人脸、车牌、门牌等敏感信息脱敏，并取得被拍摄者授权。建议仅发布特征、标注与不可逆的匿名化图像。
