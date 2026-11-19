class InferenceLogger:
    def __init__(self, interval: int, max_steps: int):
        """
        Helper to debug during inference

        Args:
            interval: log interval
            max_steps: maximum number of steps for logging
        """
        self.interval = interval
        self.max_steps = max_steps

        self.raw = None
        # model_in and out are always filled, therefore we don't set None
        self.model_in = []
        self.model_out = []
        self.post = None

        self.clear()

    def _min_len(self):
        # get minimum length of all filled lists, we need this because
        # we don't know which list is filled, and theirs lengths can differ during
        # logging, cause there are filled one after another
        filled = [getattr(self, s) for s in ["raw", "model_in", "model_out", "post"] if getattr(self, s) is not None]
        return min([len(f) for f in filled])

    def clear(self):
        self.raw = None
        self.model_in = []
        self.model_out = []
        self.post = None

    def _log_step(self, step) -> bool:
        return (step % self.interval == 0) and (self._min_len() <= self.max_steps)

    def log_raw(self, data, step: int):
        if self._log_step(step):
            if self.raw is None:
                self.raw = []
            self.raw.append(data)

    def log_model_in(self, data, step: int):
        if self._log_step(step):
            self.model_in.append(data)

    def log_model_out(self, data, step: int):
        if self._log_step(step):
            self.model_out.append(data)

    def log_post(self, data, step: int):
        if self._log_step(step):
            if self.post is None:
                self.post = []
            self.post.append(data)


class NoOpLogger(InferenceLogger):
    def __init__(self):
        # this is somewhat brittle as we depend on the implementation of the parent
        super().__init__(interval=1, max_steps=-1)
