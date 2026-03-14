def test_session_turn_flow(client):
    session_resp = client.post(
        "/api/v1/sessions",
        json={
            "schema_version": "1.0",
            "meta": {"user_id": "u001", "source": "test"},
            "payload": {"user_id": "u001", "avatar_id": "elder_female_01", "profile_seed": {"user_id": "u001"}},
            "extensions": {},
        },
    )
    assert session_resp.status_code == 200
    session_id = session_resp.json()["payload"]["session_id"]

    turn_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={
            "schema_version": "1.0",
            "meta": {"user_id": "u001", "source": "test"},
            "payload": {
                "input_type": "text",
                "text_input": "我最近睡不着，压力很大",
                "client_state": {"mic_on": True},
            },
            "extensions": {"provider_raw": {"asr": "mock"}},
        },
    )
    assert turn_resp.status_code == 200
    body = turn_resp.json()
    assert body["payload"]["output_a"]["response"]["reply_text"]
    assert body["payload"]["output_b"]["dialog_state_update"]["stage"] == "exploration"
    assert body["extensions"]["provider_raw"]["asr"] == "mock"


def test_end_session(client):
    session_resp = client.post(
        "/api/v1/sessions",
        json={
            "schema_version": "1.0",
            "meta": {"user_id": "u001"},
            "payload": {"user_id": "u001"},
            "extensions": {},
        },
    )
    session_id = session_resp.json()["payload"]["session_id"]
    end_resp = client.post(
        f"/api/v1/sessions/{session_id}/end",
        json={
            "schema_version": "1.0",
            "meta": {"user_id": "u001"},
            "payload": {"save_history": True},
            "extensions": {},
        },
    )
    assert end_resp.status_code == 200
    assert end_resp.json()["payload"]["ended"] is True
