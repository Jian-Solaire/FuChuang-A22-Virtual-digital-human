# 创建位置: e:\ALtool\project_chuang\2d_wife_make\asset_factory\avatar_manager\utils.py
import os
import shutil
from pathlib import Path
from typing import Optional

def create_avatar_directory(avatar_name: str, base_path: str = "wife_assets/avatars") -> Path:
    """创建头像专属目录及其子目录"""
    avatar_dir = Path(base_path) / avatar_name
    avatar_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录用于分类存储不同类型的文件
    (avatar_dir / "images").mkdir(exist_ok=True)    # 存储头像图像
    (avatar_dir / "audios").mkdir(exist_ok=True)    # 存储参考音频
    (avatar_dir / "motions").mkdir(exist_ok=True)   # 存储动作数据
    
    return avatar_dir

def save_avatar_media(avatar_dir: Path, image_file, audio_file):
    """保存头像媒体文件到指定目录"""
    # 保存图像文件
    if image_file:
        # 保存用户上传的图像
        image_ext = image_file.filename.split('.')[-1].lower()
        image_path = avatar_dir / "images" / f"{avatar_dir.name}.{image_ext}"
        with open(image_path, "wb") as f:
            f.write(image_file.file.read())
    
    # 保存音频文件
    if audio_file:
        # 保存用户上传的音频
        audio_ext = audio_file.filename.split('.')[-1].lower()
        audio_path = avatar_dir / "audios" / f"{avatar_dir.name}.{audio_ext}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.file.read())

def scan_avatar_folders(base_path: str = "wife_assets/avatars") -> list:
    """扫描头像文件夹，获取最早的用户上传图像"""
    base_dir = Path(base_path)
    if not base_dir.exists():
        return []
    
    avatars = []
    for item in base_dir.iterdir():
        if item.is_dir():
            # 获取头像信息
            images_dir = item / "images"
            
            # 找到最早的用户上传图像
            image_path = ""
            if images_dir.exists():
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    image_files.extend(list(images_dir.glob(ext)))
                
                if image_files:
                    # 按创建时间排序，取最早的
                    earliest_image = min(image_files, key=lambda x: x.stat().st_ctime)
                    image_path = str(earliest_image.relative_to(Path.cwd()))
            
            avatars.append({
                "name": item.name,
                "image_path": str(image_path)
            })
    
    return avatars