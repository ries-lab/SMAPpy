from typing import Any

from ..generic import protocols
from . import emitter


class ForwardableEmitter(protocols.Forwardable):
    def forward(self, em: emitter.EmitterSet) -> Any:
        ...


class EmitterProcessor(ForwardableEmitter):
    def forward(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        ...
