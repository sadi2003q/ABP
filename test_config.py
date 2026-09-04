from pathlib import Path

from src.utils.downloader import GoogleDriveDownloader

url = "https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV"

output = Path("~/HDD/EventDatasets/mvsec/raw").expanduser()

GoogleDriveDownloader().download(
    url=url,
    output=output,
)