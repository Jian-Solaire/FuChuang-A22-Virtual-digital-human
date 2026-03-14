# 创建位置: e:\ALtool\project_chuang\2d_wife_make\asset_factory\avatar_manager\api.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import asyncio
from .manager import AvatarManager
from .models import AvatarInfo

router = APIRouter(prefix="/api/avatars", tags=["avatars"])
avatar_manager = AvatarManager(base_path="assets/avatars")

@router.get("/", response_model=list)
async def get_avatars():
    """获取所有数字人预设，只返回name和最早上传的图像路径"""
    try:
        avatars = avatar_manager.get_avatars()
        return avatars
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取头像列表失败: {str(e)}")

@router.post("/create")
async def create_avatar(
    name: str = Form(...),                 # 头像名称（必填）
    avatar_image: UploadFile = File(...), # 用户上传的头像图像（必填）
    ref_audio: UploadFile = File(...),    # 参考声音文件（必填）
    persona: Optional[str] = Form(None)   # 人设（可选）
):
    """创建数字人头像"""
    try:
        # 调用管理器创建头像
        success = await avatar_manager.create_avatar(
            avatar_name=name,
            image_file=avatar_image,
            audio_file=ref_audio,
            persona=persona
        )
        
        if success:
            return {"status": "success", "message": f"数字人 {name} 创建成功"}
        else:
            raise HTTPException(status_code=500, detail="创建数字人失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建数字人时出错: {str(e)}")