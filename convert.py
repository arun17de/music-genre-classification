import os
import subprocess
from pathlib import Path

# === CONFIGURATION ===
DATASET_DIRS = [
    Path("C:/Users/arunb/Desktop/mgc/backend/mgc_split/train"),  # change path if needed
    Path("C:/Users/arunb/Desktop/mgc/backend/mgc_split/test")
]

OUTPUT_BITRATE = "128k"
SAMPLE_RATE = 22050
CHANNELS = 1


def reencode_audio(input_path: Path):
    """
    Re-encode an MP3 file to clean, standard format using FFmpeg.
    Output overwrites the same file (creates temp file first).
    """
    try:
        temp_output = input_path.with_suffix(".temp.mp3")

        cmd = [
            "ffmpeg",
            "-y",                     # overwrite without asking
            "-i", str(input_path),    # input file
            "-ac", str(CHANNELS),     # mono
            "-ar", str(SAMPLE_RATE),  # 22.05 kHz
            "-b:a", OUTPUT_BITRATE,   # 128 kbps CBR
            str(temp_output)
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Replace original file with re-encoded one
        os.replace(temp_output, input_path)
        print(f"✅ Re-encoded: {input_path}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error re-encoding {input_path}: {e}")


def process_dataset_folder(folder: Path):
    """
    Walk through all subfolders and re-encode all .mp3 files.
    """
    print(f"\n🔍 Processing folder: {folder}")
    if not folder.exists():
        print(f"⚠️ Folder not found: {folder}")
        return

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".mp3"):
                file_path = Path(root) / file
                reencode_audio(file_path)


if __name__ == "__main__":
    for dataset in DATASET_DIRS:
        process_dataset_folder(dataset)

    print("\n🎉 All MP3 files re-encoded successfully!")
