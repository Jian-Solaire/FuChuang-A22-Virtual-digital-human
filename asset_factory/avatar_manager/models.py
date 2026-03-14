# 创建位置: e:\ALtool\project_chuang\2d_wife_make\asset_factory\avatar_manager\models.py
from pydantic import BaseModel
from typing import Optional

class AvatarCreateRequest(BaseModel):
    name: str              # 头像名称（必填）
    persona: Optional[str] = None  # 人设（可选）

class AvatarInfo(BaseModel):
    name: str              # 头像名称
    image_path: str        # 图像路径