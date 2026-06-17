---
name: analyze_image
description: 使用视觉大模型分析用户上传的医学图片，输出可见内容描述、图片类型、医学相关性和风险提示。适用于影像、皮肤照片、医学图表等图片输入。
script: analyze
function: analyze_image
parameters:
  - name: image_path
    type: string
    description: 本地图片路径
    required: false
  - name: image_url
    type: string
    description: 图片 URL 或 public_url
    required: false
  - name: question
    type: string
    description: 用户关于图片的真实问题
    required: false
---

# Analyze Image（医学图像分析）

## When to Use
- 用户上传医学影像、皮肤照片、检查图、医学图表
- 用户询问图片内容、可见异常、是否需要就医
- triage 判断到本轮输入包含图片

## 底层实现
- 调用当前项目的 `QwenVLClient.summarize_image`
- 只做可见内容描述和风险提示，不输出临床确诊
- VLM 不可用、缺少图片或图片无法读取时返回结构化降级结果

