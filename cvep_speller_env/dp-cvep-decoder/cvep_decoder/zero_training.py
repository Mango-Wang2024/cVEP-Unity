import numpy as np
from numpy.typing import NDArray


class ZeroTrainingClassifier:
    """A no-calibration cVEP classifier based only on known stimulus codes.

    This is intentionally small: it builds code templates from the configured
    event definition and compares incoming EEG to those templates without
    fitting subject-specific spatial or temporal filters.
    """

    def __init__(
        self,
        stimulus: NDArray,
        fs: int,
        event: str = "contrast",
        onset_event: bool = True,
        response_length: float = 0.3,
        tmin: float = 0.1,
        min_time: float | None = None,
        max_time: float | None = None,
    ) -> None:
        self.fs = fs
        self.event = event
        self.onset_event = onset_event
        self.response_length = response_length
        self.tmin = tmin
        self.min_time = min_time
        self.max_time = max_time
        self.set_stimulus(stimulus)

    @property
    def estimator(self):
        return self

    def fit(self, X: NDArray, y: NDArray):
        return self

    def set_stimulus(self, stimulus: NDArray) -> None:
        self.stimulus = np.asarray(stimulus, dtype="float32")
        self.templates_ = self._make_templates(self.stimulus)
        self.Ts_ = self.templates_[:, np.newaxis, :]
        self.Tw_ = self.Ts_.copy()

    def decision_function(self, X: NDArray) -> NDArray:
        if X.ndim == 2:
            X = X[np.newaxis, :, :]

        templates = self._templates_for_length(X.shape[2])
        Xn = self._normalize(X.astype("float32"), axis=2)
        Tn = self._normalize(templates.astype("float32"), axis=1)

        scores = np.einsum("ncs,ks->nkc", Xn, Tn)
        return np.mean(np.abs(scores), axis=2)

    def predict(self, X: NDArray) -> NDArray:
        if X.ndim == 2:
            X = X[np.newaxis, :, :]

        ctime = X.shape[2] / self.fs
        if self.min_time is not None and ctime <= self.min_time:
            return np.full(X.shape[0], -1, dtype="int64")

        return np.argmax(self.decision_function(X), axis=1)

    def _make_templates(self, stimulus: NDArray) -> NDArray:
        events, labels = self._event_matrix(stimulus, self.event, self.onset_event)
        self.event_labels_ = labels

        kernel = self._canonical_response()
        delay = int(round(self.tmin * self.fs))
        templates = np.zeros((events.shape[0], events.shape[2]), dtype="float32")

        for i_event, label in enumerate(labels):
            if label == "fall":
                polarity = -1.0
            elif label == "onset":
                polarity = 0.25
            else:
                polarity = 1.0

            for i_class in range(events.shape[0]):
                response = np.convolve(events[i_class, i_event, :], polarity * kernel, mode="full")
                response = response[: events.shape[2]]
                if delay > 0:
                    response = np.concatenate((np.zeros(delay), response[:-delay]))
                elif delay < 0:
                    response = np.concatenate((response[-delay:], np.zeros(-delay)))
                templates[i_class, :] += response

        return self._normalize(templates, axis=1)

    def _canonical_response(self) -> NDArray:
        n_samples = max(1, int(round(self.response_length * self.fs)))
        t = np.arange(n_samples) / self.fs
        response = (
            np.exp(-0.5 * ((t - 0.10) / 0.035) ** 2)
            - 0.6 * np.exp(-0.5 * ((t - 0.18) / 0.050) ** 2)
        )
        return self._normalize(response.astype("float32"), axis=0)

    def _templates_for_length(self, n_samples: int) -> NDArray:
        n_repeats = int(np.ceil(n_samples / self.templates_.shape[1]))
        return np.tile(self.templates_, (1, n_repeats))[:, :n_samples]

    @staticmethod
    def _event_matrix(stimulus: NDArray, event: str, onset_event: bool) -> tuple[NDArray, tuple[str, ...]]:
        if stimulus.ndim == 1:
            stimulus = stimulus[np.newaxis, :]
        n_stims, n_samples = stimulus.shape

        if event in ["id", "identity", "stim", "stimulus"]:
            events = stimulus[:, np.newaxis, :]
            labels = (event,)
        elif event == "on":
            events = stimulus[:, np.newaxis, :] > 0
            labels = ("on",)
        elif event == "off":
            events = stimulus[:, np.newaxis, :] == 0
            labels = ("off",)
        elif event == "onoff":
            on = stimulus > 0
            off = stimulus == 0
            events = np.concatenate((on[:, np.newaxis, :], off[:, np.newaxis, :]), axis=1)
            labels = ("on", "off")
        elif event in ["re", "rise", "risingedge"]:
            diff = np.diff(np.concatenate((np.zeros((n_stims, 1)), stimulus), axis=1), axis=1)
            events = (diff > 0)[:, np.newaxis, :]
            labels = ("rise",)
        elif event in ["fe", "fall", "fallingedge"]:
            diff = np.diff(np.concatenate((np.zeros((n_stims, 1)), stimulus), axis=1), axis=1)
            events = (diff < 0)[:, np.newaxis, :]
            labels = ("fall",)
        elif event in ["refe", "risefall", "risingedgefallingedge", "contrast"]:
            diff = np.diff(np.concatenate((np.zeros((n_stims, 1)), stimulus), axis=1), axis=1)
            rise = diff > 0
            fall = diff < 0
            events = np.concatenate((rise[:, np.newaxis, :], fall[:, np.newaxis, :]), axis=1)
            labels = ("rise", "fall")
        else:
            raise ValueError(f"Unsupported zero-training event: {event}")

        if onset_event:
            events = np.concatenate((events, np.zeros((n_stims, 1, n_samples))), axis=1)
            events[:, -1, 0] = 1
            labels += ("onset",)

        return events.astype("float32"), labels

    @staticmethod
    def _normalize(x: NDArray, axis: int) -> NDArray:
        x = x - np.mean(x, axis=axis, keepdims=True)
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / np.maximum(norm, np.finfo("float32").eps)
