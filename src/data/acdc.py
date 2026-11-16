"""Conservative grouping of GoPro chapter directories for internal development."""
from pathlib import Path
import re


def recording_group(image):
    name = Path(image).parent.name
    match = re.fullmatch(r"(?:GOPR|GP\d{2})(\d{4})", name)
    return f"gopro-{match[1]}" if match else name
