# GPT Image Generator

Azure OpenAI gpt-image-2 图片生成/编辑 Web 应用。

## 功能

- 文本生成图片（Text-to-Image）
- 图片编辑（上传参考图 + 提示词）
- 生成历史记录与管理
- 支持多种尺寸和质量选项
- 用户认证与隔离（Azure Easy Auth）

## 技术栈

- **后端**: FastAPI + Gunicorn/Uvicorn
- **前端**: Vue 3 (CDN, 无构建步骤)
- **AI**: Azure OpenAI gpt-image-2
- **部署**: Azure App Service (Linux, Python 3.11)
- **存储**: Azure Files 持久化图片

## 项目结构

```
GPT-Image/
├── src/                  # 源码（部署到 Azure 的内容）
│   ├── main.py           # FastAPI 后端
│   ├── index.html        # Vue 3 前端
│   ├── requirements.txt  # Python 依赖
│   ├── favicon.ico
│   └── favicon.png
├── .env                  # 本地环境变量（不部署）
├── deploy.cmd            # Azure 部署脚本
├── start.cmd             # 本地启动脚本
└── output/               # 本地生成图片存储
```

## 本地开发

1. 创建 `.env` 文件：

```env
AZURE_OPENAI_ENDPOINT=<your-endpoint>
OPENAI_API_VERSION=2025-04-01-preview
DEPLOYMENT_NAME=gpt-image-2
AZURE_OPENAI_API_KEY=<your-key>
OUTPUT_DIR=../output
```

2. 启动：

```cmd
start.cmd
```

3. 访问 http://localhost:8000

## Azure 部署

配置好 `deploy.cmd` 中的资源名后运行：

```cmd
deploy.cmd
```

需要的 Azure 资源：App Service (Linux Python 3.11)、Storage Account (Azure Files)、Azure OpenAI (gpt-image-2 部署)。

## 环境变量

| Key | 说明 |
|-----|------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI 终端节点 |
| `OPENAI_API_VERSION` | API 版本 |
| `DEPLOYMENT_NAME` | 模型部署名称 |
| `AZURE_OPENAI_API_KEY` | API 密钥 |
| `OUTPUT_DIR` | 图片输出目录（本地 `../output`，Azure `/mnt/output`） |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | 部署时构建（Azure 设为 `true`） |
| `AZURE_RG` | Azure 资源组名称（deploy.cmd 使用） |
| `AZURE_APP` | Azure Web App 名称（deploy.cmd 使用） |
| `AZURE_APP_URL` | Web App 完整 URL（deploy.cmd 使用） |
