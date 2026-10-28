import bz2
import contextlib
import io
import logging
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

LOG = logging.getLogger(__name__)

ATTEMPTS = 5
TIMEOUT_SECONDS = 60
LOCAL_HEADER_SIZE = 30
NAME_LENGTHS = slice(26, 30)
BUFFER_SIZE = 1 << 20
GIGABYTE = 1e9


def read_range(request: Request) -> bytes:
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        return response.read()


def get_range(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    request = Request(url, headers=headers)  # noqa: S310
    for _ in range(ATTEMPTS - 1):
        with contextlib.suppress(OSError):
            return read_range(request)
    return read_range(request)


class RangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = Request(url, method="HEAD")  # noqa: S310
        with urlopen(head) as response:  # noqa: S310
            self.size = int(response.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(buffer)) - 1
        data = get_range(self.url, self.pos, end)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_archive(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=BUFFER_SIZE))


def fetch_member(url: str, info: zipfile.ZipInfo, out: Path) -> int:
    header_end = info.header_offset + LOCAL_HEADER_SIZE - 1
    header = get_range(url, info.header_offset, header_end)
    name_length, extra_length = struct.unpack("<HH", header[NAME_LENGTHS])
    start = info.header_offset + LOCAL_HEADER_SIZE + name_length + extra_length
    data = get_range(url, start, start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_BZIP2:
        data = bz2.decompress(data)
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -zlib.MAX_WBITS)
    if zlib.crc32(data) != info.CRC:
        message = f"{info.filename}: CRC mismatch"
        raise ValueError(message)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return len(data)


def fetch_members(
    url: str, jobs: list[tuple[zipfile.ZipInfo, Path]], workers: int, every: int
) -> int:
    done, size = 0, 0
    with ThreadPoolExecutor(workers) as pool:
        for count in pool.map(lambda job: fetch_member(url, *job), jobs):
            done, size = done + 1, size + count
            if done % every == 0:
                LOG.info(f"{done}/{len(jobs)}, {size / GIGABYTE:.1f} GB")
    return size
