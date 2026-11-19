from ... import emitter

from abc import abstractmethod


class EmitterValidator:
    @abstractmethod
    def validate(self, em: emitter.EmitterSet):
        raise NotImplementedError
