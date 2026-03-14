from backend.app.core.envelope import EnvelopeResponse, RequestMeta


def test_request_meta_defaults():
    meta = RequestMeta()
    assert meta.request_id
    assert meta.source == "api"
    assert "T" in meta.timestamp


def test_envelope_response_extensions_default():
    response = EnvelopeResponse(payload={"ok": True})
    assert response.schema_version == "1.0"
    assert response.extensions == {}
