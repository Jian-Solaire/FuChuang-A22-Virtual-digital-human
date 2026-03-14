# 创建位置: e:\ALtool\project_chuang\2d_wife_make\asset_factory\avatar_manager\manager.py
import asyncio
import subprocess
from pathlib import Path
from typing import Optional
from .utils import create_avatar_directory, save_avatar_media, scan_avatar_folders
from .models import AvatarInfo

class AvatarManager:
    def __init__(self, base_path: str = "assets/avatars"):
        self.base_path = base_path
        Path(base_path).mkdir(parents=True, exist_ok=True)

    async def create_avatar(
        self, 
        avatar_name: str, 
        image_file, 
        audio_file, 
        persona: Optional[str] = None
    ) -> bool:
        """创建数字人头像的完整流程"""
        try:
            # 1. 创建头像目录结构
            avatar_dir = create_avatar_directory(avatar_name, self.base_path)
            
            # 2. 保存媒体文件（图像和音频）
            save_avatar_media(avatar_dir, image_file, audio_file)
            
            # 3. 创建头像清单文件，记录头像元数据
            manifest_path = avatar_dir / "manifest.json"
            import json
            manifest = {
                "name": avatar_name,
                "persona": persona or "",
                "created_at": str(Path(avatar_dir).stat().st_ctime),
                "assets": {
                    "image": str(list((avatar_dir / "images").glob("*.*"))[0]) if list((avatar_dir / "images").glob("*.*")) else "",
                    "audio": str(list((avatar_dir / "audios").glob("*.*"))[0]) if list((avatar_dir / "audios").glob("*.*")) else ""
                }
            }
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            
            # 4. 同步调用LivePortrait生成动作预设（使用与run.py相同的方式）
            user_image_path = str(list((avatar_dir / "images").glob("*.*"))[0])
            self.run_liveportrait_pipeline(user_image_path, avatar_name)
            
            return True
        except Exception as e:
            print(f"创建头像失败: {str(e)}")
            return False

    def run_liveportrait_pipeline(self, source_image_path: str, character_name: str):
        """同步调用LivePortrait运行完整管道，与直接运行run.py相同"""
        try:
            import subprocess
            import os
            
            print(f"开始执行LivePortrait处理...")
            print(f"源图片路径: {source_image_path}")
            print(f"角色名称: {character_name}")
            
            # 获取当前工作目录
            original_cwd = os.getcwd()
            
            # 确保源图片路径是绝对路径，因为run.py将在LivePortrait目录下执行
            abs_source_path = os.path.abspath(source_image_path)
            
            # 构建命令，与直接运行run.py相同
            cmd = [
                "python", 
                "run.py",
                "--source", abs_source_path,
                "--character", character_name
            ]
            
            try:
                # 切换到LivePortrait目录执行
                lp_dir = os.path.join(original_cwd, "LivePortrait")
                print(f"在目录 {lp_dir} 中执行命令: {' '.join(cmd)}")
                
                # 使用实时输出而不是捕获输出，这样可以看到处理过程
                result = subprocess.run(cmd, cwd=lp_dir, text=True, timeout=600)  # 10分钟超时
                
                if result.returncode != 0:
                    print(f"LivePortrait处理失败，返回码: {result.returncode}")
                else:
                    print(f"LivePortrait处理成功完成")
                    
            except subprocess.TimeoutExpired:
                print("LivePortrait处理超时（超过10分钟）")
            except Exception as e:
                print(f"执行LivePortrait时出错: {str(e)}")
            
        except Exception as e:
            print(f"运行LivePortrait管道时出错: {str(e)}")

    async def generate_motion_presets(self, avatar_name: str, image_path: str):
        """保留原始方法以兼容"""
        try:
            # 使用与run.py相同的方式来调用LivePortrait
            # 在项目根目录执行命令
            cmd = [
                "python", 
                "LivePortrait/run.py", 
                "--character", 
                avatar_name,
                "--source", 
                image_path
            ]
            
            # 在子进程中异步执行LivePortrait
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".."  # 调整到项目根目录，因为命令是从asset_factory目录执行的
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"LivePortrait处理失败: {stderr.decode()}")
            else:
                print(f"LivePortrait处理成功: {stdout.decode()}")
                
        except Exception as e:
            print(f"生成动作预设时出错: {str(e)}")

    def get_avatars(self) -> list:
        """获取所有头像信息，只返回name和最早上传的图像路径"""
        return scan_avatar_folders(self.base_path)