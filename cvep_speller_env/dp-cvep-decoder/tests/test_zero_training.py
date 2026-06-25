from copy import deepcopy

import numpy as np
import pyntbci

from cvep_decoder.online_decoding import OnlineDecoder
from cvep_decoder.train_decoder import ClassifierMeta, get_zero_training_model


def test_zero_training_model_is_pyntbci_urcca():
    stimulus = np.random.default_rng(3).integers(0, 2, size=(2, 126))

    model = get_zero_training_model(ClassifierMeta(), stimulus)

    assert isinstance(model, pyntbci.classifiers.urCCA)


def test_zero_training_model_applies_tmin_to_structure_matrix():
    stimulus = np.random.default_rng(5).integers(0, 2, size=(2, 126))
    cmeta = ClassifierMeta(sfreq=60, encoding_length=0.3, ctmin=0.15)
    unshifted = pyntbci.classifiers.urCCA(
        stimulus=stimulus,
        fs=60,
        event=cmeta.event,
        onset_event=cmeta.onset_event,
        encoding_length=cmeta.encoding_length,
    )

    model = get_zero_training_model(cmeta, stimulus)

    original = np.concatenate((unshifted.Ms, unshifted.Mw), axis=2)
    shifted = np.concatenate((model.Ms, model.Mw), axis=2)
    shift_samples = 9
    np.testing.assert_array_equal(shifted[:, :, :shift_samples], 0)
    np.testing.assert_array_equal(
        shifted[:, :, shift_samples:], original[:, :, :-shift_samples]
    )
    assert model.tmin == 0.15


def test_online_zero_training_fits_eeg_and_updates_model():
    rng = np.random.default_rng(7)
    stimulus = rng.integers(0, 2, size=(2, 126))
    model = pyntbci.classifiers.urCCA(
        stimulus=stimulus,
        fs=60,
        event="contrast",
        onset_event=True,
        encoding_length=0.3,
    )
    decoder = object.__new__(OnlineDecoder)
    decoder.classifier = model
    decoder.current_trial_id = 2
    decoder.classifier_meta = {"target_accuracy": 0.0}
    decoder.first_trial_target_accuracy = 0.99
    eeg = rng.normal(size=(1, 3, 252))

    prediction = decoder._fit_predict_zero_training(eeg)

    assert 0 <= prediction < stimulus.shape[0]
    assert model.rho.shape == (stimulus.shape[0],)
    assert all(cca.running for cca in model.ccas)


def test_zero_training_confidence_uses_competing_correlations():
    low_separation = np.asarray([0.1, 0.2, 0.3, 0.31])
    high_separation = np.asarray([0.1, 0.2, 0.3, 0.9])

    low = OnlineDecoder._zero_training_confidence(low_separation)
    high = OnlineDecoder._zero_training_confidence(high_separation)

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_low_confidence_does_not_change_cumulative_model():
    rng = np.random.default_rng(11)
    stimulus = rng.integers(0, 2, size=(4, 126))
    model = pyntbci.classifiers.urCCA(
        stimulus=stimulus,
        fs=60,
        event="contrast",
        onset_event=True,
        encoding_length=0.3,
    )
    decoder = object.__new__(OnlineDecoder)
    decoder.classifier = model
    decoder.current_trial_id = 1
    decoder.classifier_meta = {"target_accuracy": 0.95}
    decoder.first_trial_target_accuracy = 1.0
    previous_ccas = deepcopy(model.ccas)

    prediction = decoder._fit_predict_zero_training(
        rng.normal(size=(1, 3, 252))
    )

    assert 0 <= prediction < stimulus.shape[0]
    assert all(
        vars(before) == vars(after)
        for before, after in zip(previous_ccas, model.ccas)
    )


def test_disabled_cumulative_updates_preserve_model():
    rng = np.random.default_rng(13)
    stimulus = rng.integers(0, 2, size=(2, 126))
    model = pyntbci.classifiers.urCCA(
        stimulus=stimulus,
        fs=60,
        event="contrast",
        onset_event=True,
        encoding_length=0.3,
    )
    decoder = object.__new__(OnlineDecoder)
    decoder.classifier = model
    decoder.current_trial_id = 2
    decoder.classifier_meta = {"target_accuracy": 0.0}
    decoder.first_trial_target_accuracy = 0.99
    decoder.zero_training_cumulative_updates = False
    previous_ccas = deepcopy(model.ccas)

    decoder._fit_predict_zero_training(rng.normal(size=(1, 3, 252)))

    assert all(
        vars(before) == vars(after)
        for before, after in zip(previous_ccas, model.ccas)
    )
