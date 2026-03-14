# 使用 Miniconda 作为基础镜像，MFA 强烈依赖 conda 环境
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 替换 conda 源为清华源（加速下载），并安装 MFA 和相关依赖
RUN conda config --add channels conda-forge && \
    conda install -c conda-forge montreal-forced-aligner=2.2.17 -y

# 安装 Web 服务外壳 (FastAPI) 和 TextGrid 解析库
RUN pip install fastapi uvicorn python-multipart textgrid

# 【核心步骤】提前把普通话的声学模型和字典下载到镜像里，避免每次运行都下载
RUN mfa model download acoustic mandarin_mfa && \
    mfa model download dictionary mandarin_pinyin_mfa

# 把本地的微服务代码复制进容器
COPY server.py /app/server.py

# 暴露给宿主机（你的 Go 后端）调用的端口
EXPOSE 8000

# 启动微服务
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]