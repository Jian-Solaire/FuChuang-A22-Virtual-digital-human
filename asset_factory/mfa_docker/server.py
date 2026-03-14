import os
import uuid
import shutil
import subprocess
import textgrid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import logging

# SRE 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MFA-Service] - %(message)s')

app = FastAPI(title="MFA Alignment Microservice")

@app.post("/align")
async def align_audio(
    audio: UploadFile = File(...),
    text: str = Form(...)
):
    """
    接收音频和文本，返回对齐后的 JSON 音素时间轴
    """
    # 1. 创建隔离的工作区 (防止并发请求互相干扰)
    task_id = str(uuid.uuid4())
    corpus_dir = f"/tmp/mfa_workspace/{task_id}/corpus"
    aligned_dir = f"/tmp/mfa_workspace/{task_id}/aligned"
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)

    try:
        # 2. 将上传的音频和文本写入隔离目录
        wav_path = os.path.join(corpus_dir, f"{task_id}.wav")
        txt_path = os.path.join(corpus_dir, f"{task_id}.txt")
        
        with open(wav_path, "wb") as f:
            f.write(await audio.read())
            
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        logging.info(f"Task [{task_id}] - 开始执行 MFA 强制对齐...")

        # 3. 调用 MFA 命令行 (使用镜像内预装的 mandarin 模型)
        # mfa align [语料目录] [字典] [声学模型] [输出目录]
        command = [
            "mfa", "align", 
            corpus_dir, 
            "mandarin_pinyin_mfa", 
            "mandarin_mfa", 
            aligned_dir, 
            "--clean"
        ]
        
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode != 0:
            logging.error(f"Task [{task_id}] - MFA 运行失败: {process.stderr}")
            return JSONResponse(status_code=500, content={"error": "MFA 对齐失败", "details": process.stderr})

        # 4. 寻找并解析产出的 TextGrid 文件
        textgrid_path = os.path.join(aligned_dir, f"{task_id}.TextGrid")
        if not os.path.exists(textgrid_path):
            return JSONResponse(status_code=500, content={"error": "MFA 未生成 TextGrid 文件"})

        tg = textgrid.TextGrid.fromFile(textgrid_path)
        phones_tier = tg.getFirst('phones')

        # 5. 格式化为你的标准 JSON
        result = []
        for interval in phones_tier:
            p = interval.mark
            # 过滤掉静音和空标记
            if p and p not in ['sil', 'sp', 'spn', '']:
                result.append({
                    "p": p,
                    "s": round(interval.minTime, 4),
                    "e": round(interval.maxTime, 4)
                })

        logging.info(f"Task [{task_id}] - 对齐成功，提取到 {len(result)} 个音素。")
        return {"task_id": task_id, "alignment": result}

    except Exception as e:
        logging.error(f"Task [{task_id}] - 发生异常: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # 6. SRE 清理原则：销毁临时文件，防止 Docker 容器磁盘爆炸
        if os.path.exists(f"/tmp/mfa_workspace/{task_id}"):
            shutil.rmtree(f"/tmp/mfa_workspace/{task_id}")
            logging.info(f"Task [{task_id}] - 临时工作区已清理。")