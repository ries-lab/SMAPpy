from typing import Any

from . import emitter
from ..generic import protocols


class ForwardableEmitter(protocols.Forwardable):
    def forward(self, em: emitter.EmitterSet) -> Any:
        ...


class EmitterProcessor(ForwardableEmitter):
    def forward(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        ...
