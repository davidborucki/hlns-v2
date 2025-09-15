#!/usr/bin/env python3
"""Build a dataset from YouTube podcast channels and their Shorts."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import parse_qs, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
import yt_dlp
import whisper

try:
    import torch
except ImportError:  # torch ships with whisper but guard just in case
    torch = None  # type: ignore[assignment]

LOGGER = logging.getLogger("shorts-scraper")
WATCH_FULL_PATTERN = re.compile(
    r"\"watchFullVideoEndpoint\"\s*:\s*\{[^}]*?\"videoId\"\s*:\s*\"(?P<video_id>[^\"]+)\"",
    re.S,
)
YT_INITIAL_DATA_PATTERN = re.compile(r"ytInitialData\s*=\s*(\{.+?\})\s*;\s*</script>", re.S)


def read_channels(file_path: Path) -> List[str]:
    channels: List[str] = []
    if not file_path.exists():
        raise FileNotFoundError(f"Channels file not found: {file_path}")
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        channels.append(line)
    return channels


def fetch_shorts_urls(channel_url: str, limit: int) -> List[str]:
    shorts_url = channel_url.rstrip("/") + "/shorts"
    LOGGER.info("Fetching up to %d shorts from %s", limit, shorts_url)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistend": limit,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(shorts_url, download=False)
    except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
        LOGGER.error("Failed to fetch shorts from %s: %s", shorts_url, exc)
        return []
    entries = info.get("entries") if isinstance(info, dict) else None
    if not entries:
        LOGGER.warning("No shorts found for %s", channel_url)
        return []
    urls: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if isinstance(url, str) and url:
            if not url.startswith("http"):
                url = urljoin("https://www.youtube.com", url)
            urls.append(normalize_short_url(url))
            continue
        video_id = entry.get("id")
        if isinstance(video_id, str) and video_id:
            urls.append(f"https://www.youtube.com/shorts/{video_id}")
    return list(OrderedDict.fromkeys(urls))


def normalize_short_url(url: str) -> str:
    parsed = urlsplit(url)
    match = re.search(r"/shorts/([^/?]+)", parsed.path)
    if match:
        return f"https://www.youtube.com/shorts/{match.group(1)}"
    return url


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def extract_full_episode_url(short_url: str) -> Optional[str]:
    short_id = extract_video_id(short_url)
    try:
        html = fetch_html(short_url)
    except requests.RequestException as exc:
        LOGGER.error("Failed to fetch short page %s: %s", short_url, exc)
        return None
    soup = BeautifulSoup(html, "html.parser")

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not isinstance(href, str):
            continue
        full_href = urljoin("https://www.youtube.com", href)
        if "/watch" not in urlsplit(full_href).path:
            continue
        if "shorts" in full_href:
            continue
        video_id = extract_video_id(full_href)
        label = link.get_text(strip=True).lower()
        if video_id and video_id != short_id and ("watch" in label or "full" in label or label):
            return normalize_watch_url(full_href)

    match = WATCH_FULL_PATTERN.search(html)
    if match:
        video_id = match.group("video_id")
        if video_id and video_id != short_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    data = extract_yt_initial_data(html)
    if data is not None:
        for url in collect_watch_urls(data):
            absolute = urljoin("https://www.youtube.com", url)
            video_id = extract_video_id(absolute)
            if video_id and video_id != short_id:
                return normalize_watch_url(absolute)
    return None


def extract_yt_initial_data(html: str) -> Optional[Any]:
    match = YT_INITIAL_DATA_PATTERN.search(html)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def collect_watch_urls(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        command = node.get("commandMetadata")
        if isinstance(command, dict):
            metadata = command.get("webCommandMetadata")
            if isinstance(metadata, dict):
                url = metadata.get("url")
                if isinstance(url, str) and url.startswith("/watch"):
                    yield url
        endpoint = node.get("watchEndpoint")
        if isinstance(endpoint, dict):
            video_id = endpoint.get("videoId")
            if isinstance(video_id, str):
                yield f"/watch?v={video_id}"
        for value in node.values():
            yield from collect_watch_urls(value)
    elif isinstance(node, list):
        for item in node:
            yield from collect_watch_urls(item)


def extract_video_id(url: str) -> Optional[str]:
    parsed = urlsplit(url)
    if parsed.netloc.endswith("youtu.be"):
        video_id = parsed.path.strip("/")
        return video_id or None
    if parsed.path.startswith("/shorts/"):
        segment = parsed.path.split("/")[-1]
        return segment or None
    if parsed.path == "/watch":
        query = parse_qs(parsed.query)
        values = query.get("v")
        if values:
            return values[0]
    return None


def normalize_watch_url(url: str) -> str:
    video_id = extract_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def download_audio(url: str, dest: Path) -> Optional[Path]:
    outtmpl = str(dest) + ".%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
        LOGGER.error("yt-dlp failed for %s: %s", url, exc)
        return None
    file_path = _resolve_download_path(info)
    if file_path is None:
        LOGGER.error("Could not determine downloaded file path for %s", url)
        return None
    return Path(file_path)


def _resolve_download_path(info: Any) -> Optional[str]:
    if isinstance(info, dict):
        requested = info.get("requested_downloads")
        if isinstance(requested, list):
            for item in requested:
                if isinstance(item, dict):
                    path = item.get("filepath")
                    if isinstance(path, str):
                        return path
        path = info.get("filepath")
        if isinstance(path, str):
            return path
    return None


def transcribe_audio(model: whisper.Whisper, audio_path: Path, fp16: bool) -> str:
    result = model.transcribe(str(audio_path), fp16=fp16)
    text = result.get("text", "") if isinstance(result, dict) else ""
    return text.strip()


def build_dataset(channels: Sequence[str], output_dir: Path, shorts_limit: int, model_name: str) -> None:
    full_map: Dict[str, Dict[str, Any]] = OrderedDict()

    for channel in channels:
        LOGGER.info("Processing channel %s", channel)
        short_urls = fetch_shorts_urls(channel, shorts_limit)
        LOGGER.info("Found %d shorts for %s", len(short_urls), channel)
        for index, short_url in enumerate(short_urls, start=1):
            LOGGER.info("  [%d/%d] %s", index, len(short_urls), short_url)
            full_url = extract_full_episode_url(short_url)
            if not full_url:
                LOGGER.warning("    No full episode link for %s", short_url)
                continue
            video_id = extract_video_id(full_url)
            if not video_id:
                LOGGER.warning("    Could not parse full video ID for %s", full_url)
                continue
            entry = full_map.setdefault(
                full_url,
                {"video_id": video_id, "shorts": []},
            )
            if short_url not in entry["shorts"]:
                entry["shorts"].append(short_url)
                LOGGER.info("    Linked to full episode %s", full_url)

    if not full_map:
        LOGGER.info("No full episodes discovered. Nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    if torch is not None and torch.cuda.is_available():  # type: ignore[union-attr]
        device = "cuda"
    LOGGER.info("Loading Whisper model '%s' on %s", model_name, device)
    model = whisper.load_model(model_name, device=device)
    fp16 = device == "cuda"

    for full_url, payload in full_map.items():
        video_id = payload["video_id"]
        shorts = payload["shorts"]
        target_dir = output_dir / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Processing full episode %s (%d clips)", full_url, len(shorts))

        full_txt_path = target_dir / "full.txt"
        if not full_txt_path.exists():
            audio_path = download_audio(full_url, target_dir / "full_audio")
            if audio_path is None:
                LOGGER.error("Skipping full episode %s because audio download failed", full_url)
            else:
                try:
                    LOGGER.info("  Transcribing full episode %s", full_url)
                    transcript = transcribe_audio(model, audio_path, fp16=fp16)
                    full_txt_path.write_text(transcript + "\n", encoding="utf-8")
                except Exception as exc:  # whisper may raise generic exceptions
                    LOGGER.error("  Transcription failed for %s: %s", full_url, exc)
                finally:
                    if audio_path.exists():
                        audio_path.unlink()
        else:
            LOGGER.info("  full.txt already exists for %s, skipping transcription", full_url)

        for clip_index, short_url in enumerate(shorts, start=1):
            clip_txt_path = target_dir / f"clip{clip_index}.txt"
            if clip_txt_path.exists():
                LOGGER.info("  %s already exists, skipping", clip_txt_path.name)
                continue
            audio_path = download_audio(short_url, target_dir / f"clip{clip_index}_audio")
            if audio_path is None:
                LOGGER.error("  Skipping clip %s (download failed)", short_url)
                continue
            try:
                LOGGER.info("  Transcribing clip %s", short_url)
                transcript = transcribe_audio(model, audio_path, fp16=fp16)
                clip_txt_path.write_text(transcript + "\n", encoding="utf-8")
            except Exception as exc:
                LOGGER.error("  Transcription failed for clip %s: %s", short_url, exc)
            finally:
                if audio_path.exists():
                    audio_path.unlink()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels-file",
        default="list-channels.txt",
        help="Path to text file with one YouTube channel URL per line",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Folder where the dataset will be written",
    )
    parser.add_argument(
        "--shorts-limit",
        type=int,
        default=500,
        help="Number of shorts to fetch per channel",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Whisper model size to use (default: medium)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        channels = read_channels(Path(args.channels_file))
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1
    if not channels:
        LOGGER.error("Channel list is empty")
        return 1
    LOGGER.info("Loaded %d channels", len(channels))

    try:
        build_dataset(
            channels,
            Path(args.output_dir),
            shorts_limit=args.shorts_limit,
            model_name=args.whisper_model,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 1
    except Exception as exc:
        LOGGER.exception("Unhandled error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
