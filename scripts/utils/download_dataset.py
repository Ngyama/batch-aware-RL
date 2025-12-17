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
    """Download and extract ImageNette dataset."""
    
    if os.path.exists(IMAGENETTE_EXTRACTED_PATH):
        print(f"ImageNette dataset already exists at: {IMAGENETTE_EXTRACTED_PATH}")
        return

    print(f"Preparing ImageNette dataset in '{DATA_PATH}' folder...")
    os.makedirs(DATA_PATH, exist_ok=True)

    print(f"Downloading ImageNette dataset from {IMAGENETTE_URL}...")
    response = requests.get(IMAGENETTE_URL, stream=True)
    response.raise_for_status() 

    total_size = int(response.headers.get('content-length', 0))

    with open(IMAGENETTE_ARCHIVE_PATH, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    print("Download completed.")

    print(f"Extracting files to: {DATA_PATH}...")
    with tarfile.open(IMAGENETTE_ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_PATH)
    print("Extraction completed.")

    os.remove(IMAGENETTE_ARCHIVE_PATH)
    print(f"Removed archive: {IMAGENETTE_ARCHIVE_PATH}")

    print("\nImageNette dataset is ready!")


def download_speech_commands():
    """Download Google Speech Commands dataset."""
    
    if os.path.exists(SPEECH_COMMANDS_PATH):
        files = os.listdir(SPEECH_COMMANDS_PATH)
        if len(files) > 0:
            print(f"Speech Commands dataset already exists at: {SPEECH_COMMANDS_PATH}")
            return
    
    print(f"Downloading Google Speech Commands dataset to: {SPEECH_COMMANDS_PATH}...")
    
    try:
        import torchaudio
        from torchaudio.datasets import SPEECHCOMMANDS
        
        os.makedirs(SPEECH_COMMANDS_PATH, exist_ok=True)
        
        print("Starting download...")
        dataset = SPEECHCOMMANDS(root=SPEECH_COMMANDS_PATH, download=True, subset='training')
        print(f"Speech Commands dataset download completed! Total {len(dataset)} samples.")
        
    except ImportError:
        print("[ERROR] torchaudio is required: pip install torchaudio")
        raise
    except Exception as e:
        print(f"[ERROR] Error downloading Speech Commands dataset: {e}")
        raise


def download_all_datasets():
    """Download all required datasets."""
    print("="*70)
    print("DOWNLOADING DATASETS")
    print("="*70 + "\n")
    
    print("[1/2] Downloading ImageNette dataset...")
    download_imagenette()
    
    print("\n[2/2] Downloading Speech Commands dataset...")
    download_speech_commands()
    
    print("\n" + "="*70)
    print("All datasets are ready!")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--dataset', type=str, choices=['imagenette', 'speech_commands', 'all'], 
                       default='all', help='Dataset to download')
    
    args = parser.parse_args()
    
    if args.dataset == 'imagenette':
        download_imagenette()
    elif args.dataset == 'speech_commands':
        download_speech_commands()
    else:
        download_all_datasets()