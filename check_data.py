import os
import argparse
from PIL import Image
from tqdm import tqdm

def safe_check_images(folder_path):
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 创建进度条
    pbar = tqdm(total=len(image_files), desc="Checking images", unit="image")
    
    # 遍历并检查每个图像文件
    for filename in image_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                # 检查图像文件是否完整
                if img.fp.tell() != os.path.getsize(file_path):
                    print(f"Image file is truncated: {file_path}")
                    # 询问用户是否尝试修复
                   # choice = input("Do you want to attempt a repair? (y/n): ")
                    if 1==1:
                        # 尝试修复截断的图像（这可能不总是有效）
                        img.load()  # 尝试加载图像数据以修复截断问题
                else:
                    try:
                        # 尝试转换为 RGB 格式
                        img = img.convert('RGB')
                    except (IOError, OSError) as e:
                       print(f"Failed to convert image to RGB: {file_path}. Error: {e}")
                        # 询问用户是否尝试修复
                        #choice = input("Do you want to attempt a repair? (y/n): ")
                       if 1 == 1:
                            # 尝试修复转换错误（这可能不总是有效）
                            # 例如，对于某些格式，可能需要先转换为其他格式再转换为 RGB
                            try:
                                img = img.convert('P').convert('RGB')
                            except Exception as e:
                                print(f"Failed to repair image: {file_path}. Error: {e}")
                    else:
                        # 验证图像文件是否损坏
                        img.verify()
        except (IOError, SyntaxError, OSError) as e:
            print(f"Error processing image: {file_path}. Error: {e}")
            # 询问用户是否尝试修复
            #choice = input("Do you want to attempt a repair? (y/n): ")
            if 1==1:
                # 尝试修复处理错误（这可能不总是有效）
                # 例如，对于某些错误，可能需要尝试重新打开或转换图像
                try:
                    with Image.open(file_path) as img:
                        img.load()
                except Exception as e:
                    print(f"Failed to repair image: {file_path}. Error: {e}")
        finally:
            pbar.update(1)  # 更新进度条
    
    pbar.close()  # 关闭进度条

# 命令行参数解析
parser = argparse.ArgumentParser(description='Safe check images in a folder.')
parser.add_argument('folder_path', type=str, help='The path to the folder containing images to check.')

args = parser.parse_args()

# 使用脚本
safe_check_images(args.folder_path)
