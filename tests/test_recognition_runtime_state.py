from src.recognition.recognition_runtime_state import RecognitionRuntimeState


def test_runtime_state_confirms_known_identity_after_min_votes():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=3, reset_after_misses=10)

    state.update(predicted_name="custom:valerio", confidence=20.0, frame_usable=True)
    state.update(predicted_name="custom:valerio", confidence=22.0, frame_usable=True)
    result = state.update(predicted_name="custom:valerio", confidence=24.0, frame_usable=True)

    assert result["final_name"] == "custom:valerio"
    assert result["candidate_name"] == "custom:valerio"
    assert result["candidate_votes"] == 3
    assert result["stable_count"] == 3


def test_runtime_state_tracks_unknown_votes():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=3, reset_after_misses=10)

    state.update(predicted_name="Unknown", confidence=60.0, frame_usable=True)
    state.update(predicted_name="Unknown", confidence=62.0, frame_usable=True)
    result = state.update(predicted_name="Unknown", confidence=65.0, frame_usable=True)

    assert result["final_name"] == "Unknown"
    assert result["candidate_name"] == "Unknown"
    assert result["candidate_votes"] == 3
    assert result["unknown_votes"] == 3


def test_runtime_state_holds_when_frame_is_not_usable():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=3, reset_after_misses=10)

    state.update(predicted_name="custom:valerio", confidence=20.0, frame_usable=True)
    result = state.update(frame_usable=False, recognition_allowed=True, face_missing=False)

    assert result["history_size"] == 1
    assert result["current_name"] is None


def test_runtime_state_resets_after_consecutive_missing_faces():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=2, reset_after_misses=2)

    state.update(predicted_name="custom:valerio", confidence=20.0, frame_usable=True)
    state.update(predicted_name="custom:valerio", confidence=21.0, frame_usable=True)

    state.update(face_missing=True)
    result = state.update(face_missing=True)

    assert result["final_name"] == "Unknown"
    assert result["history_size"] == 0


def test_runtime_state_shifts_candidate_when_new_name_becomes_majority():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=3, reset_after_misses=10)

    state.update(predicted_name="custom:valerio", confidence=20.0, frame_usable=True)
    state.update(predicted_name="custom:valerio", confidence=21.0, frame_usable=True)
    state.update(predicted_name="custom:luca", confidence=18.0, frame_usable=True)
    state.update(predicted_name="custom:luca", confidence=19.0, frame_usable=True)
    result = state.update(predicted_name="custom:luca", confidence=20.0, frame_usable=True)

    assert result["candidate_name"] == "custom:luca"
    assert result["candidate_votes"] == 3
    assert result["final_name"] == "custom:luca"


def test_runtime_state_handles_alternating_known_and_unknown_without_confirming_known():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=3, reset_after_misses=10)

    state.update(predicted_name="custom:valerio", confidence=20.0, frame_usable=True)
    state.update(predicted_name="Unknown", confidence=60.0, frame_usable=True)
    state.update(predicted_name="custom:valerio", confidence=21.0, frame_usable=True)
    result = state.update(predicted_name="Unknown", confidence=62.0, frame_usable=True)

    assert result["final_name"] == "Unknown"
    assert result["candidate_name"] in {"Unknown", "custom:valerio"}
    assert result["unknown_votes"] == 2


def test_runtime_state_holds_and_counts_miss_when_recognition_is_not_allowed():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=2, reset_after_misses=3)

    state.update(predicted_name="custom:valerio", confidence=20.0, frame_usable=True)
    state.update(predicted_name="custom:valerio", confidence=21.0, frame_usable=True)
    result = state.update(frame_usable=True, recognition_allowed=False, face_missing=False)

    assert result["final_name"] == "custom:valerio"
    assert result["history_size"] == 2
    assert result["missing_face_counter"] == 1


def test_runtime_state_does_not_reset_after_non_consecutive_missing_faces():
    state = RecognitionRuntimeState(window_size=5, min_confirmations=2, reset_after_misses=2)

    state.update(predicted_name="custom:valerio", confidence=20.0, frame_usable=True)
    state.update(predicted_name="custom:valerio", confidence=21.0, frame_usable=True)
    state.update(face_missing=True)
    result = state.update(predicted_name="custom:valerio", confidence=22.0, frame_usable=True)

    assert result["missing_face_counter"] == 0
    assert result["history_size"] == 3
    assert result["final_name"] == "custom:valerio"
