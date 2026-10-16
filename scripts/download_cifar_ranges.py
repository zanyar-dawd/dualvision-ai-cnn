"""Fetch the official Toronto CIFAR archive in bounded parallel HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import requests
import time
import threading

URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
SIZE = 170498071
MD5 = 'c58f30108f718f92721af3b95e74349a'


def main():
    root = Path('.downloads/cifar-parts')
    root.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024 * 1024
    pieces = [(i, start, min(start + chunk_size, SIZE) - 1)
              for i, start in enumerate(range(0, SIZE, chunk_size))]
    # The failed torchvision transfer may contain a useful prefix. The final
    # published MD5 still gates extraction/use of every byte, including this prefix.
    partial = Path('data/cifar-10-python.tar.gz')
    if partial.exists() and (root / '000.part').exists():
        with partial.open('rb') as source:
            prefix_matches = source.read(chunk_size) == (root / '000.part').read_bytes()
            if prefix_matches:
                for index, start, end in pieces:
                    if end >= partial.stat().st_size:
                        break
                    target = root / f'{index:03}.part'
                    if not target.exists() or target.stat().st_size != end - start + 1:
                        source.seek(start)
                        target.write_bytes(source.read(end - start + 1))
                print('Reused the incomplete official transfer prefix; full archive checksum remains mandatory.', flush=True)
    local = threading.local()
    def fetch(piece):
        index, start, end = piece
        target = root / f'{index:03}.part'
        if target.exists() and target.stat().st_size == end - start + 1:
            return index
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        for attempt in range(5):
            try:
                with local.session.get(URL, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(30, 45)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                        raise RuntimeError('Server did not honor the exact requested byte range.')
                    with target.open('wb') as handle:
                        for block in response.iter_content(32 * 1024):
                            handle.write(block)
                if target.stat().st_size != end - start + 1:
                    raise IOError('Incomplete range response.')
                return index
            except (requests.RequestException, IOError) as error:
                if attempt == 4:
                    raise
                print(f'Range {index}: retry {attempt + 1} after {type(error).__name__}', flush=True)
                time.sleep(attempt + 1)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, piece) for piece in pieces]
        for future in as_completed(futures):
            try:
                index = future.result()
            except Exception as error:
                print(f'Range download failed: {type(error).__name__}: {error}', flush=True)
                for pending in futures:
                    pending.cancel()
                raise
            print(f'Official CIFAR archive range {index + 1}/{len(pieces)} verified ({time.perf_counter() - started:.1f}s).', flush=True)
    assembled = root / 'cifar-10-python.tar.gz'
    digest = hashlib.md5()
    with assembled.open('wb') as output:
        for index, _, _ in pieces:
            with (root / f'{index:03}.part').open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
                    digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError('Official archive MD5 mismatch; refusing to install the download.')
    print(f'PASS: official CIFAR-10 archive MD5 {MD5}; staged at {assembled}', flush=True)


if __name__ == '__main__':
    main()
