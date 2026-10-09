import subprocess
from pathlib import Path


# Could add ability for diff url here, would need to handle year & month parsing
def fetch_raw(
    url: str = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst",
):
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / url.split("/")[-1]

    try:
        print(f"Downloading {url}")
        subprocess.run(
            [
                "curl",
                "-f",  # Fail on HTTP errors (404, 500, etc.)
                "-L",  # Follow redirects
                "-C",  # Resume downloads
                "-",  # Automatically determine where to resume
                "--retry",
                "10",  # Retry up to 10 times
                "--progress-bar",  # Display progress bar
                "-o",
                str(output),  # Specify output file
                url,
            ],
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("curl is not installed") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {url}") from e

    print(f"Downloaded to {output}")
    return output
