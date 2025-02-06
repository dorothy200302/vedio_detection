# Safety Hat Detection with Claude Vision

基于Claude Vision API的安全帽检测系统，用于识别和分析图片中的安全帽佩戴情况。

## 功能特点

- 🎯 自动检测图片中的安全帽
- 📊 提供检测置信度
- 📍 定位安全帽在图片中的位置
- 📝 分析安全帽佩戴状态
- 🗃️ 批量处理图片并生成分析报告

## 环境要求

- Python 3.7+
- Anthropic API key

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository-url]
cd safety-hat-detection
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
- 复制 `.env.example` 为 `.env`
- 在 `.env` 文件中设置您的 Anthropic API key：
```
ANTHROPIC_API_KEY=your_api_key_here
```

## 使用方法

1. 准备数据：
- 将需要检测的图片放在 `images/train` 目录下

2. 运行检测：
```bash
python detect_safety_hat.py
```

3. 查看结果：
- 检测结果将保存在 `detection_results.json` 文件中
- 结果包含每张图片的以下信息：
  - 是否存在安全帽
  - 安全帽位置
  - 检测置信度
  - 佩戴状态分析

## 项目结构

```
safety-hat-detection/
├── detect_safety_hat.py   # 主检测脚本
├── requirements.txt       # 项目依赖
├── .env                  # 环境变量配置
├── images/               # 图片目录
│   └── train/           # 训练集图片
└── detection_results.json # 检测结果
```

## 注意事项

- 请确保您有足够的 Claude API 配额
- 图片格式支持：jpg, jpeg, png
- 建议使用清晰的图片以获得更准确的检测结果

## License

MIT License 