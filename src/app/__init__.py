"""App package — top-level user-facing facade.

Provides :class:`App` (config + resolver cache + Builder factories) and
the three Builder types: :class:`VideoBuilder`, :class:`CourseBuilder`,
and :class:`StreamBuilder` / :class:`LiveStreamHandle`.
"""

from app._app import App
from app._course import CourseBuilder
from app._stream import LiveStreamHandle, StreamBuilder
from app._video import VideoBuilder

__all__ = [
    "App",
    "CourseBuilder",
    "LiveStreamHandle",
    "StreamBuilder",
    "VideoBuilder",
]
