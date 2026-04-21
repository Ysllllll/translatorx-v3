"""App package — top-level user-facing facade.

Provides :class:`App` (config + resolver cache + Builder factories) and
the three Builder types: :class:`VideoBuilder`, :class:`CourseBuilder`,
and :class:`StreamBuilder` / :class:`LiveStreamHandle`.
"""

from api.app._app import App
from api.app._course import CourseBuilder
from api.app._stream import LiveStreamHandle, StreamBuilder
from api.app._video import VideoBuilder

__all__ = [
    "App",
    "CourseBuilder",
    "LiveStreamHandle",
    "StreamBuilder",
    "VideoBuilder",
]
