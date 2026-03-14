import os

import pytest


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="External services and .env are not configured yet.",
)
def test_e2e_placeholder():
    # This test is intentionally minimal until Docker and .env are provided.
    assert True
