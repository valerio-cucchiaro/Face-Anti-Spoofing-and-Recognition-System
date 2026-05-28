from src.pipeline import run_pipeline_pad
from src.pipeline import run_pipeline_recognition
from src.pipeline import run_pipeline_full
from src.pipeline.runtime_session_state import RuntimeSessionState


def _pad_result(final_label):
    return {
        "final_label": final_label,
        "mean_live_probability": None,
    }


def _recognition_result(
    final_name="Unknown",
    current_name=None,
    candidate_name=None,
    candidate_votes=0,
    unknown_votes=0,
    history_size=0,
):
    return {
        "final_name": final_name,
        "current_name": current_name,
        "candidate_name": candidate_name,
        "candidate_votes": candidate_votes,
        "unknown_votes": unknown_votes,
        "history_size": history_size,
        "stable_count": candidate_votes,
        "current_confidence": None,
    }


def test_runtime_session_state_resets_all_counters_and_payloads():
    session = RuntimeSessionState()

    assert session.transition_to(RuntimeSessionState.PAD_CHECK) is True
    assert session.transition_to(RuntimeSessionState.PAD_CHECK) is False

    session.register_missing_face()
    session.register_pad_live_confirmation()
    session.register_locked_movement(True)
    session.register_locked_crop_validity(False)
    session.register_locked_spoof_evidence(True)
    session.store_face_rect((1, 2, 3, 4))
    session.store_pad_result({"final_label": "live"})
    session.store_recognition_result({"final_name": "alice"})

    session.reset_session()

    assert session.current_phase == RuntimeSessionState.DETECT
    assert session.previous_face_rect is None
    assert session.pad_live_confirmation_count == 0
    assert session.missing_face_counter == 0
    assert session.locked_movement_counter == 0
    assert session.locked_invalid_crop_counter == 0
    assert session.locked_spoof_counter == 0
    assert session.live_lock_started_at is None
    assert session.last_pad_result["final_label"] == "unknown"
    assert session.last_recognition_result["final_name"] == "Unknown"


def test_full_pad_display_state_promotes_live_then_decays():
    display_state = run_pipeline_full.PadDisplayState()
    log_messages = []

    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("live"),
        pad_frame_usable=True,
        now=0.0,
        logger=log_messages.append,
    )
    assert display_state.state == run_pipeline_full.PadDisplayState.ANALYZING

    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("live"),
        pad_frame_usable=True,
        now=2.0,
        logger=log_messages.append,
    )
    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("live"),
        pad_frame_usable=True,
        now=2.1,
        logger=log_messages.append,
    )

    assert display_state.state == run_pipeline_full.PadDisplayState.SHOW_LIVE
    assert display_state.get_pad_text() == "PAD: LIVE"
    assert display_state.get_box_color() == (0, 255, 0)

    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("spoof"),
        pad_frame_usable=True,
        now=4.0,
        logger=log_messages.append,
    )
    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("spoof"),
        pad_frame_usable=True,
        now=4.1,
        logger=log_messages.append,
    )
    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("spoof"),
        pad_frame_usable=True,
        now=4.2,
        logger=log_messages.append,
    )

    assert display_state.state == run_pipeline_full.PadDisplayState.ANALYZING
    assert any("live_confirmed" in message for message in log_messages)
    assert any("live_decay" in message for message in log_messages)


def test_pad_only_display_state_promotes_spoof():
    display_state = run_pipeline_pad.PadDisplayState()
    log_messages = []

    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("spoof"),
        pad_frame_usable=True,
        now=0.0,
        logger=log_messages.append,
    )
    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("spoof"),
        pad_frame_usable=True,
        now=2.0,
        logger=log_messages.append,
    )
    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("spoof"),
        pad_frame_usable=True,
        now=2.1,
        logger=log_messages.append,
    )
    display_state.update(
        face_present=True,
        pad_state_result=_pad_result("spoof"),
        pad_frame_usable=True,
        now=2.2,
        logger=log_messages.append,
    )

    assert display_state.state == run_pipeline_pad.PadDisplayState.SHOW_SPOOF
    assert display_state.get_pad_text() == "PAD: SPOOF"
    assert display_state.get_box_color() == (0, 0, 255)
    assert any("spoof_confirmed" in message for message in log_messages)


def test_recognition_only_display_state_promotes_known_identity():
    display_state = run_pipeline_recognition.RecognitionDisplayState()
    log_messages = []

    display_state.update(
        face_present=True,
        recognition_result=_recognition_result(
            final_name="custom:alice",
            current_name="custom:alice",
            candidate_name="custom:alice",
            candidate_votes=3,
            history_size=3,
        ),
        now=0.0,
        logger=log_messages.append,
    )

    assert (
        display_state.state
        == run_pipeline_recognition.RecognitionDisplayState.SHOW_KNOWN
    )
    assert display_state.get_identity_text() == "Identity: custom: alice"
    assert display_state.get_identity_color() == (0, 255, 0)
    assert any("known_confirmed:custom:alice" in message for message in log_messages)


def test_recognition_only_display_state_promotes_unknown_identity():
    display_state = run_pipeline_recognition.RecognitionDisplayState()
    log_messages = []

    display_state.update(
        face_present=True,
        recognition_result=_recognition_result(
            final_name="Unknown",
            candidate_name="Unknown",
            unknown_votes=3,
            history_size=3,
        ),
        now=0.0,
        logger=log_messages.append,
    )

    assert (
        display_state.state
        == run_pipeline_recognition.RecognitionDisplayState.SHOW_UNKNOWN
    )
    assert display_state.get_identity_text() == "Identity: Unknown"
    assert display_state.get_identity_color() == (0, 0, 255)
    assert any("unknown_confirmed" in message for message in log_messages)
