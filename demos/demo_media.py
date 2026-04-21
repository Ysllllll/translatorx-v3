"""media — 媒体操作演示。

展示 probe (媒体探测)、extract_audio (音频提取)、YtdlpSource (下载) 的使用方法。

运行:
    python demos/demo_media.py

注意：
  - probe / extract_audio 需要系统安装 ffmpeg + ffprobe
  - YtdlpSource 需要安装 yt-dlp (pip install yt-dlp)
  - 以下示例仅展示 API 结构，实际执行需要真实文件/URL
"""

import _bootstrap  # noqa: F401

import asyncio

from adapters.media import (
    MediaInfo,
    PlaylistInfo,
    DownloadResult,
    MediaFileInfo,
    MediaSource,
    YtdlpSource,
    probe,
    extract_audio,
)


async def demo_probe():
    """演示 ffprobe 探测媒体文件信息。"""
    print("=== probe (ffprobe) ===")
    print("""
    用法:
        info = await probe("/path/to/video.mp4")
        print(info.duration)    # 秒数
        print(info.has_audio)   # 是否含音频
        print(info.has_video)   # 是否含视频
        print(info.audio_codec) # 音频编码格式
        print(info.video_codec) # 视频编码格式

    返回 MediaFileInfo 数据类:
        - duration: float
        - has_audio: bool
        - has_video: bool
        - audio_codec: str | None
        - video_codec: str | None
        - format_name: str | None
    """)


async def demo_extract_audio():
    """演示从视频中提取音频。"""
    print("=== extract_audio ===")
    print("""
    用法:
        # 基本提取 (默认 WAV 格式)
        output = await extract_audio(
            input_path="/path/to/video.mp4",
            output_path="/path/to/audio.wav",
        )

        # 指定格式和采样率 (WhisperX 推荐 16kHz mono)
        output = await extract_audio(
            input_path="/path/to/video.mp4",
            output_path="/path/to/audio.wav",
            sample_rate=16000,
            mono=True,
        )
    """)


async def demo_ytdlp_source():
    """演示 YtdlpSource 下载媒体。"""
    print("=== YtdlpSource ===")
    print("""
    用法:
        source = YtdlpSource()

        # 获取媒体信息 (不下载)
        info = await source.info("https://youtube.com/watch?v=xxx")
        print(info.title)
        print(info.duration)
        print(info.subtitles)  # 可用字幕列表

        # 下载音频
        result = await source.download(
            url="https://youtube.com/watch?v=xxx",
            output_dir="/tmp/downloads",
            format="bestaudio",
        )
        print(result.filepath)  # 下载的文件路径

        # 获取播放列表信息 (课程批量处理)
        playlist = await source.playlist_info("https://youtube.com/playlist?list=xxx")
        for entry in playlist.entries:
            print(f"  {entry.title} ({entry.duration}s)")

    数据类型:
        MediaInfo    — 单个媒体的元信息 (title, duration, subtitles, ...)
        PlaylistInfo — 播放列表 (title, entries: list[MediaInfo])
        DownloadResult — 下载结果 (filepath, format, ...)
    """)


async def demo_protocol():
    """演示 MediaSource Protocol 接口。"""
    print("=== MediaSource Protocol ===")
    print("""
    MediaSource 定义了媒体源的标准接口:

        class MediaSource(Protocol):
            async def info(self, url: str) -> MediaInfo: ...
            async def download(self, url: str, output_dir: str, **kwargs) -> DownloadResult: ...
            async def playlist_info(self, url: str) -> PlaylistInfo: ...

    实现:
        - YtdlpSource: yt-dlp 后端 (YouTube, Bilibili, etc.)
        - 未来可添加: BilibiliSource, PodcastSource 等

    新增媒体源只需实现 MediaSource Protocol 即可无缝接入系统。
    """)


async def main():
    await demo_probe()
    print()
    await demo_extract_audio()
    print()
    await demo_ytdlp_source()
    print()
    await demo_protocol()


if __name__ == "__main__":
    asyncio.run(main())
