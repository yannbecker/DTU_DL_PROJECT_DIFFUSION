import requests
from pathlib import Path

# === CONFIGURATION ===
url = "https://zenodo.org/record/8286452/files/annotation_model_v1.tar.gz?download=1"
output_path = Path("annotation_model_v1.tar.gz")  # fichier local

# === TELECHARGEMENT EN STREAM ===
chunk_size = 1024 * 1024  # 1 MB par chunk

resume_byte_pos = 0
if output_path.exists():
    resume_byte_pos = output_path.stat().st_size

headers = {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos > 0 else {}

with requests.get(url, stream=True, headers=headers) as r:
    r.raise_for_status()
    mode = "ab" if resume_byte_pos > 0 else "wb"
    with open(output_path, mode) as f:
        print(f"Downloading to {output_path}...")
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

print(f"Download complete: {output_path}")
