import os
import requests
import tarfile
from tqdm import tqdm

# --- 配置信息 ---
# ImageNette数据集的官方下载链接
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

# 我们希望将数据存放在项目根目录下的 'data' 文件夹中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# 下载的压缩包保存路径
ARCHIVE_PATH = os.path.join(DATA_PATH, "imagenette2.tgz")

# 解压后的文件夹最终路径
EXTRACTED_FOLDER_PATH = os.path.join(DATA_PATH, "imagenette2")

# --- 主逻辑 ---
def download_and_prepare_dataset():
    """检查、下载并解压数据集到项目文件夹内。"""

    # 1. 检查数据集是否已经存在，如果存在则无需任何操作
    if os.path.exists(EXTRACTED_FOLDER_PATH):
        print(f"数据集已存在于: {EXTRACTED_FOLDER_PATH}")
        return

    # 2. 如果不存在，则创建 'data' 文件夹
    print(f"将在 '{DATA_PATH}' 文件夹中准备数据集...")
    os.makedirs(DATA_PATH, exist_ok=True)

    # 3. 下载数据集压缩包，并显示进度条
    print(f"正在从 {DATA_URL} 下载数据集...")
    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status() # 确保请求成功

    total_size = int(response.headers.get('content-length', 0))

    with open(ARCHIVE_PATH, 'wb') as f, tqdm(
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

    # 4. 解压下载好的 .tgz 文件
    print(f"正在解压文件到: {DATA_PATH}...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_PATH)
    print("解压完成。")

    # 5. (可选) 清理下载的压缩包以节省空间
    os.remove(ARCHIVE_PATH)
    print(f"已删除压缩包: {ARCHIVE_PATH}")

    print("\n数据集已准备就绪！")

if __name__ == "__main__":
    download_and_prepare_dataset()