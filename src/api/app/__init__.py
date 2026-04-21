"""App package — top-level user-facing facade.

Provides :class:`App` (config + resolver cache + Builder factories) and
the three Builder types: :class:`VideoBuilder`, :class:`CourseBuilder`,
and :class:`StreamBuilder` / :class:`LiveStreamHandle`.
"""

from api.app.app import App
from api.app.course import CourseBuilder
from api.app.stream import LiveStreamHandle, StreamBuilder
from api.app.video import VideoBuilder

__all__ = [
    "App",
    "CourseBuilder",
    "LiveStreamHandle",
    "StreamBuilder",
    "VideoBuilder",
]
