FROM continuumio/miniconda3:22.11.1

# 设置工作目录
WORKDIR /app

# 更新系统包
RUN apt-get update && apt-get install -y ffmpeg curl wget && apt-get clean

# 替换 conda 源为清华源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes

# 分别安装Python和基本依赖，避免一次性安装过多包
RUN conda install python=3.9 -y && \
    conda install pip -y

# 使用pip安装MFA，这通常比conda更稳定
RUN pip install Montreal-Forced-Aligner==2.2.17 textgrid fastapi uvicorn python-multipart

# 下载中文模型
RUN mfa model download acoustic mandarin_mfa || echo "Acoustic model download failed, continuing..." && \
    mfa model download dictionary mandarin_pinyin_mfa || echo "Dictionary model download failed, continuing..."

# 如果模型下载失败，提供备用方案
RUN mkdir -p /root/Documents/MFA/

# 把本地的微服务代码复制进容器
COPY server.py /app/server.py

# 暴露给宿主机调用的端口
EXPOSE 8000

# 启动微服务
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]