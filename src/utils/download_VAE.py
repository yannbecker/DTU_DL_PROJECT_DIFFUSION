import requests
from pathlib import Path

# === CONFIGURATION ===
url = "https://zenodo.org/record/8286452/files/annotation_model_v1.tar.gz?download=1"
output_path = Path("annotation_model_v1.tar.gz")  # fichier local

chunk_size = 1024 * 1024  # 1 MB par chunk

# === Reprendre si fichier existant ===
resume_byte_pos = 0
if output_path.exists():
    resume_byte_pos = output_path.stat().st_size

headers = {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos > 0 else {}

# Obtenir la taille totale pour afficher progression
response_head = requests.head(url, allow_redirects=True)
total_size = int(response_head.headers.get('Content-Length', 0))
if resume_byte_pos > 0:
    total_size += resume_byte_pos

with requests.get(url, stream=True, headers=headers) as r:
    r.raise_for_status()
    mode = "ab" if resume_byte_pos > 0 else "wb"
    with open(output_path, mode) as f:
        downloaded = resume_byte_pos
        print(f"Downloading to {output_path} ({total_size / 1e9:.2f} GB total)")
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = downloaded / total_size * 100
                print(f"\rDownloaded {downloaded / 1e9:.2f} GB ({percent:.2f}%)", end="")
print("\nDownload complete!")
