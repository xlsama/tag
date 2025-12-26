# 评论多标签分类模型

基于 Unsloth + LoRA 微调 Qwen3-4B 的评论多标签分类模型。

## 安装

```bash
uv sync
```

## 训练

```bash
uv run python -m src.train.main
```

## 目录结构

```
tag/
├── src/
│   ├── train/              # 训练模块
│   │   ├── main.py         # 训练脚本
│   │   ├── config.py       # 配置
│   │   ├── labels.json     # 标签定义
│   │   ├── data.jsonl      # 训练数据
│   │   └── output/         # 模型输出
│   └── api/                # 推理服务 (待实现)
│       └── main.py
├── pyproject.toml
└── README.md
```

## 数据格式

### labels.json

```json
{
  "labels": [
    { "name": "产品质量", "description": "关于产品本身质量的评价" }
  ]
}
```

### data.jsonl

```jsonl
{"text": "这个手机质量太差了", "labels": ["产品质量"]}
{"text": "快递送得很快，包装也很好", "labels": ["物流服务", "包装问题"]}
```

## vLLM 部署

训练完成后：

```bash
vllm serve src/train/output/merged_model --port 8000
```
