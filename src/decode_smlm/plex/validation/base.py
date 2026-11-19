from abc import abstractmethod
from typing import Any, Sequence


class Validator:
    @abstractmethod
    def validate(self, x: Any) -> None:
        raise NotImplementedError


class NoOpValidator(Validator):
    def validate(self, x: Any) -> None:
        return


class SparseSteppedValidation:
    def __init__(self, validation: Sequence[Validator], step_size: int, limit: int):
        self.validation = validation
        self.runs = 0
        self.step_size = step_size
        self.limit = limit

    def validate(self, x: Any, step: int) -> None:
        do = self._do_validation(step)
        if not do:
            return
        _ = [v.validate(x) for v in self.validation]
        self.runs += 1

    def reset_(self):
        self.runs = 0

    def _do_validation(self, step: int) -> bool:
        if step % self.step_size != 0:
            return False
        if self.runs >= self.limit:
            return False
        return True


class PipelineValidator:
    def __init__(
        self,
        val_raw: SparseSteppedValidation
        | Sequence[SparseSteppedValidation]
        | None = None,
        val_model_in: SparseSteppedValidation
        | Sequence[SparseSteppedValidation]
        | None = None,
        val_model_out: SparseSteppedValidation
        | Sequence[SparseSteppedValidation]
        | None = None,
        val_post: SparseSteppedValidation
        | Sequence[SparseSteppedValidation]
        | None = None,
    ):
        self.val_raw = val_raw
        self.val_model_in = val_model_in
        self.val_model_out = val_model_out
        self.val_post = val_post

    def validate(self, x: Any) -> None:
        raise NotImplementedError

    def validate_raw(self, x: Any, step: int) -> None:
        self._validate_kernel(x, self.val_raw, step=step)

    def validate_model_in(self, x: Any, step: int) -> None:
        self._validate_kernel(x, self.val_model_in, step=step)

    def validate_model_out(self, x: Any, step: int) -> None:
        self._validate_kernel(x, self.val_model_out, step=step)

    def validate_post(self, x: Any, step: int) -> None:
        self._validate_kernel(x, self.val_post, step=step)

    @classmethod
    def from_validator(
        cls,
        steps: int,
        limit: int,
        val_raw: Sequence[Validator] | None = None,
        val_model_in: Sequence[Validator] | None = None,
        val_model_out: Sequence[Validator] | None = None,
        val_post: Sequence[Validator] | None = None,
    ):
        if val_raw is not None:
            val_raw = SparseSteppedValidation(val_raw, steps, limit)
        if val_model_in is not None:
            val_model_in = SparseSteppedValidation(val_model_in, steps, limit)
        if val_model_out is not None:
            val_model_out = SparseSteppedValidation(val_model_out, steps, limit)
        if val_post is not None:
            val_post = SparseSteppedValidation(val_post, steps, limit)
        return cls(val_raw, val_model_in, val_model_out, val_post)

    @classmethod
    def no_op(cls):
        return cls(val_raw=None, val_model_in=None, val_model_out=None, val_post=None)

    @staticmethod
    def _validate_kernel(
        x: Any,
        validators: SparseSteppedValidation | Sequence[SparseSteppedValidation] | None,
        step: int,
    ) -> None:
        if validators is not None:
            if isinstance(validators, Sequence):
                _ = [v.validate(x, step=step) for v in validators]
            else:
                validators.validate(x, step=step)
