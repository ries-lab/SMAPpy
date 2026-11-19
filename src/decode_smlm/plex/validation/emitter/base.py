from abc import abstractmethod

from ... import emitter


class EmitterValidator:
    @abstractmethod
    def validate(self, em: emitter.EmitterSet):
        raise NotImplementedError
