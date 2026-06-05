# Analysis

这是一个基于本地材料检索与大模型生成的学术报告生成项目。

当前版本固定输出三部分：
1. Introduction
2. Related Work
3. Multiple Viewpoints

## 以后换主题时改哪里
只改一个文件：

- `configs/topic.yaml`

示例：

```yaml
topic:
  subject: "你的研究主题"
  title: "你的报告标题"
  keywords: "关键词1, 关键词2"
  notes: "可选的补充提示"
```

`configs/config.yaml` 只放通用运行参数，不再放任何主题词。

## 运行

```bash
python -m src.main
```

或跳过已有步骤：

```bash
python -m src.main --skip-copy --skip-index
```
