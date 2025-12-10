import os
import sys
import requests
import tarfile
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

DATA_PATH = os.path.join(PROJECT_ROOT, "data")

IMAGENETTE_ARCHIVE_PATH = os.path.join(DATA_PATH, "imagenette2.tgz")

IMAGENETTE_EXTRACTED_PATH = os.path.join(DATA_PATH, "imagenette2")

SPEECH_COMMANDS_PATH = os.path.join(DATA_PATH, "speech_commands")

def download_imagenette():
    
    if os.path.exists(IMAGENETTE_EXTRACTED_PATH):
        print(f"ImageNette数据集已存在于: {IMAGENETTE_EXTRACTED_PATH}")
        return

    print(f"将在 '{DATA_PATH}' 文件夹中准备ImageNette数据集...")
    os.makedirs(DATA_PATH, exist_ok=True)

    print(f"正在从 {IMAGENETTE_URL} 下载ImageNette数据集...")
    response = requests.get(IMAGENETTE_URL, stream=True)
    response.raise_for_status() 

    total_size = int(response.headers.get('content-length', 0))

    with open(IMAGENETTE_ARCHIVE_PATH, 'wb') as f, tqdm(
        desc="下载中",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    print("下载完成。")

    print(f"正在解压文件到: {DATA_PATH}...")
    with tarfile.open(IMAGENETTE_ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_PATH)
    print("解压完成。")

    os.remove(IMAGENETTE_ARCHIVE_PATH)
    print(f"已删除压缩包: {IMAGENETTE_ARCHIVE_PATH}")

    print("\nImageNette数据集已准备就绪！")


def download_speech_commands():
    """下载Google Speech Commands数据集。"""
    
    if os.path.exists(SPEECH_COMMANDS_PATH):
        files = os.listdir(SPEECH_COMMANDS_PATH)
        if len(files) > 0:
            print(f"Speech Commands数据集已存在于: {SPEECH_COMMANDS_PATH}")
            return
    
    print(f"正在下载Google Speech Commands数据集到: {SPEECH_COMMANDS_PATH}...")
    
    try:
        import torchaudio
        from torchaudio.datasets import SPEECHCOMMANDS
        
        os.makedirs(SPEECH_COMMANDS_PATH, exist_ok=True)
        
        print("开始下载...")
        dataset = SPEECHCOMMANDS(root=SPEECH_COMMANDS_PATH, download=True, subset='training')
        print(f"Speech Commands数据集下载完成！共 {len(dataset)} 个样本。")
        
    except ImportError:
        print("[ERROR] 需要安装torchaudio: pip install torchaudio")
        raise
    except Exception as e:
        print(f"[ERROR] 下载Speech Commands数据集时出错: {e}")
        raise


def download_all_datasets():
    """下载所有需要的数据集。"""
    print("="*70)
    print("下载数据集")
    print("="*70 + "\n")
    
    print("[1/2] 下载ImageNette数据集...")
    download_imagenette()
    
    print("\n[2/2] 下载Speech Commands数据集...")
    download_speech_commands()
    
    print("\n" + "="*70)
    print("所有数据集已准备就绪！")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载数据集')
    parser.add_argument('--dataset', type=str, choices=['imagenette', 'speech_commands', 'all'], 
                       default='all', help='要下载的数据集')
    
    args = parser.parse_args()
    
    if args.dataset == 'imagenette':
        download_imagenette()
    elif args.dataset == 'speech_commands':
        download_speech_commands()
    else:
        download_all_datasets()