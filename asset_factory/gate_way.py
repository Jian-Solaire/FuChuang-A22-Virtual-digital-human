from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os
import shutil
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, Any
import asyncio

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Digital Avatar Gateway Server", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# 确保头像资产目录存在
avatar_assets_path = Path("assets/avatars")
avatar_assets_path.mkdir(parents=True, exist_ok=True)

# 挂载头像资产目录
avatar_assets_path = Path("wife_assets")
avatar_assets_path.mkdir(exist_ok=True)
app.mount("/wife_assets", StaticFiles(directory="wife_assets"), name="wife_assets")

# 导入并注册头像管理API
try:
    from avatar_manager.api import router as avatar_router
    app.include_router(avatar_router)
except ImportError:
    print("Warning: avatar_manager module not found, skipping integration")

# 音素到动作文件的映射表
# 优化后的：音素(Phoneme) -> 视素(Viseme) 映射表
# 你的数字人只需要准备这 8 个动作资产！

# 用于连续对话的缓存和状态管理
SESSION_CACHE = {}

PHONEME_ACTION_MAP = {
    # 1. 大张嘴 (A音主导)
    "a": "vocal_A.json", "ia": "vocal_A.json", "ua": "vocal_A.json",
    "ai": "vocal_A.json", "uai": "vocal_A.json", "an": "vocal_A.json",
    "ang": "vocal_A.json", "ian": "vocal_A.json", "iang": "vocal_A.json",
    "iao": "vocal_A.json", "uan": "vocal_A.json", "uang": "vocal_A.json",

    # 2. 圆圆嘴 (O音主导)
    "o": "vocal_O.json", "ou": "vocal_O.json", "uo": "vocal_O.json",
    "ong": "vocal_O.json", "iong": "vocal_O.json", "io": "vocal_O.json",
    "iou": "vocal_O.json",

    # 3. 半开咧嘴 (E音主导)
    "e": "vocal_E.json", "ei": "vocal_E.json", "en": "vocal_E.json",
    "eng": "vocal_E.json", "ie": "vocal_E.json",

    # 4. 闭齿咧嘴 (I音主导)
    "i": "vocal_I.json", "ii": "vocal_I.json", "iii": "vocal_I.json",
    "in": "vocal_I.json", "ing": "vocal_I.json", "er": "vocal_I.json", # er卷舌音嘴型偏向I/E

    # 5. 小圆嘟嘴 (U音主导)
    "u": "vocal_U.json", "uei": "vocal_U.json", "uen": "vocal_U.json",
    "ueng": "vocal_U.json",

    # 6. 撮口呼 (V/ü音)
    "v": "vocal_V.json", "ve": "vocal_V.json", "van": "vocal_V.json",
    "vn": "vocal_V.json",

    # 7. 完全闭嘴 (双唇音 & 静音)
    "b": "mouth_closed.json", "p": "mouth_closed.json", "m": "mouth_closed.json",
    "sil": "mouth_closed.json", "sp": "mouth_closed.json",

    # 8. 微张/默认齿音 (唇齿音 & 舌根音 & 舌尖音)
    "f": "mouth_neutral.json", "d": "mouth_neutral.json", "t": "mouth_neutral.json",
    "n": "mouth_neutral.json", "l": "mouth_neutral.json", "g": "mouth_neutral.json",
    "k": "mouth_neutral.json", "h": "mouth_neutral.json", "j": "mouth_neutral.json",
    "q": "mouth_neutral.json", "x": "mouth_neutral.json", "zh": "mouth_neutral.json",
    "ch": "mouth_neutral.json", "sh": "mouth_neutral.json", "r": "mouth_neutral.json",
    "z": "mouth_neutral.json", "c": "mouth_neutral.json", "s": "mouth_neutral.json"
}


async def mock_llm(question: str) -> str:
    """
    Step 1: LLM 思考 (Mock)
    无论问什么，直接返回固定文本
    """
    logger.info(f"Step 1: LLM 思考 - 处理问题: {question}")
    start_time = time.time()
    
    # 模拟LLM思考
    await asyncio.sleep(0.1)  # 模拟处理时间
    
    response = "今天天气真不错，我们出去玩吧。"
    end_time = time.time()
    logger.info(f"Step 1 完成，耗时: {end_time - start_time:.2f}s")
    
    return response


async def mock_tts(text: str, task_id: str) -> str:
    """
    Step 2: TTS 语音合成 (Mock) -> 写入文件
    在 assets/audio/ 目录下生成一个名为 {task_id}.wav 的实体文件
    """
    logger.info(f"Step 2: TTS 语音合成 - 生成音频: {task_id}.wav")
    start_time = time.time()
    
    # 确保目录存在
    audio_dir = Path("assets/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否存在 test.wav 用于复制
    test_wav_path = Path("test.wav")
    if not test_wav_path.exists():
        # 如果没有 test.wav，创建一个空文件作为占位符（实际部署时需要替换为真实音频）
        logger.warning("test.wav 不存在，将创建一个占位符文件用于测试")
        test_wav_path.touch()
    
    # 复制 test.wav 为 task_id.wav
    target_path = audio_dir / f"{task_id}.wav"
    shutil.copy2(test_wav_path, target_path)
    
    result_path = f"assets/audio/{task_id}.wav"
    end_time = time.time()
    logger.info(f"Step 2 完成，耗时: {end_time - start_time:.2f}s，生成文件: {result_path}")
    
    return result_path


def validate_avatar_exists(avatar_id: str) -> bool:
    """
    验证头像是否存在
    检查 wife_assets/avatars/{avatar_id}/ 目录是否存在
    """
    avatar_dir = Path(f"wife_assets/avatars/{avatar_id}")
    return avatar_dir.exists()


@app.post("/api/chat")
async def chat_endpoint(request_data: Dict[str, Any]):
    """
    核心接口: POST /api/chat
    处理前端发送的用户问题，执行完整的4步流水线
    """
    start_total = time.time()
    
    try:
        # 提取请求数据
        avatar_id = request_data.get("avatar_id")
        session_id = request_data.get("session_id")
        user_question = request_data.get("user_question")
        
        if not all([avatar_id, session_id, user_question]):
            raise HTTPException(status_code=400, detail="Missing required fields: avatar_id, session_id, user_question")
        
        # 验证头像是否存在
        if not validate_avatar_exists(avatar_id):
            raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")
        
        logger.info(f"收到聊天请求 - Avatar: {avatar_id}, Session: {session_id}, Question: {user_question}")
        
        # 生成唯一任务ID
        task_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"生成任务ID: {task_id}")
        
        # Step 1: LLM 思考
        answer_text = await mock_llm(user_question)
        
        # Step 2: TTS 语音合成
        audio_path = await mock_tts(answer_text, task_id)
        
        # Step 3: 调用 AI 对齐微服务
        logger.info(f"Step 3: 调用 AI 对齐微服务 - 任务ID: {task_id}")
        step3_start = time.time()
        
        # 读取音频文件并发送到 Docker 服务 - 修复了文件流组装方式
        audio_file_path = Path(audio_path)
        if not audio_file_path.exists():
            raise HTTPException(status_code=500, detail=f"Audio file does not exist: {audio_file_path}")
        
        # 正确的文件和表单数据发送方式
        with open(audio_file_path, 'rb') as audio_file:
            files = {
                'audio': (f"{task_id}.wav", audio_file, 'audio/wav')  # 仅音频文件
            }
            data = {
                'text': answer_text  # 文本通过 form data 发送
            }
            
            try:
                # 调用MFA对齐服务，使用正确的端点URL
                # 如果在Docker环境中，应该使用服务名；如果是本地测试，使用localhost:8080
                mfa_service_url = os.getenv("MFA_SERVICE_URL", "http://localhost:8080/align")
                response = requests.post(mfa_service_url, files=files, data=data)
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"MFA service error: {response.text}")
                
                mfa_result = response.json()
                logger.info(f"MFA 服务返回数据: {len(mfa_result.get('alignment', []))} 个音素")
                
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Failed to connect to MFA service: {str(e)}")
        
        step3_end = time.time()
        logger.info(f"Step 3 完成，耗时: {step3_end - step3_start:.2f}s")
        
        # Step 4: Mapper 映射 -> 写入文件
        logger.info(f"Step 4: Mapper 映射 - 任务ID: {task_id}")
        step4_start = time.time()
        
        # 确保脚本目录存在
        scripts_dir = Path("assets/scripts")
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换 MFA 返回的数据
        converted_script = []
        if "alignment" in mfa_result:
            for item in mfa_result["alignment"]:
                phoneme = item["p"]
                start_time = item["s"]
                end_time = item["e"]
                
                # 查找映射的动作文件，如果没有则使用默认
                action_file = PHONEME_ACTION_MAP.get(phoneme, "mouth_closed.json")
                
                converted_item = {
                    "action": action_file,
                    "s": start_time,
                    "e": end_time,
                    "duration": round(end_time - start_time, 4)
                }
                converted_script.append(converted_item)
        
        # 保存剧本文件
        script_path = scripts_dir / f"{task_id}.json"
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(converted_script, f, ensure_ascii=False, indent=2)
        
        step4_end = time.time()
        logger.info(f"Step 4 完成，耗时: {step4_end - step4_start:.2f}s")
        
        # 构建返回数据
        # 使用环境变量或默认值确定基础URL
        base_url = os.getenv("BASE_URL", "http://localhost:8081")
        avatar_base_url = f"{base_url}/assets/avatars/{avatar_id}/"
        audio_url = f"{base_url}/assets/audio/{task_id}.wav"
        script_url = f"{base_url}/assets/scripts/{task_id}.json"
        
        total_time = time.time() - start_total
        logger.info(f"完整流水线完成，总耗时: {total_time:.2f}s")
        
        return {
            "avatar_id": avatar_id,
            "answer_text": answer_text,
            "preset_base_url": avatar_base_url,
            "audio_url": audio_url,
            "script_url": script_url
        }
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Digital Avatar Gateway Server is running", "status": "ok"}


@app.get("/api/avatar/{avatar_name}")
async def get_avatar_presets(avatar_name: str):
    """
    根据数字人名称获取其预设内容
    返回数字人的配置信息和可用的动作预设
    """
    import json
    from pathlib import Path
    
    avatar_dir = Path(f"assets/avatars/{avatar_name}")
    
    # 检查数字人是否存在
    if not avatar_dir.exists():
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_name}' not found")
    
    # 获取数字人配置信息
    manifest_path = avatar_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        # 如果没有 manifest.json，创建一个基本的配置
        manifest = {
            "name": avatar_name,
            "created_at": "unknown",
            "motions": []
        }
    
    # 获取动作预设信息
    motions_dir = avatar_dir / "motions"
    if motions_dir.exists():
        motion_files = []
        for motion_file in motions_dir.glob("*.npy"):
            motion_files.append({
                "name": motion_file.stem,
                "path": f"/assets/avatars/{avatar_name}/motions/{motion_file.name}",
                "size": motion_file.stat().st_size,
                "modified": motion_file.stat().st_mtime
            })
        manifest["motions"] = motion_files
    
    # 添加基础URL信息，方便前端使用
    base_url = os.getenv("BASE_URL", "http://localhost:8081")
    manifest["base_url"] = base_url
    manifest["avatar_url"] = f"{base_url}/assets/avatars/{avatar_name}/"
    
    return manifest


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)