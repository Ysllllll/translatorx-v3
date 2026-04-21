"""Built-in punc backends.

Importing this subpackage registers every bundled backend with the
global :class:`~adapters.preprocess.punc.registry.PuncBackendRegistry`.
External libraries can register additional backends by importing
:class:`PuncBackendRegistry` and decorating their own factory.
"""

from adapters.preprocess.punc.backends import (  # noqa: F401
    deepmultilingualpunctuation,
    llm,
    remote,
)
