"""Old endpoint displacement APIs must not remain usable as support gates."""
import pytest
import contact_solver.support_transition as retired


@pytest.mark.parametrize('name', ['endpoint_contact_retention',
    'predicted_contact_retention_loss', 'endpoint_contact_retention_http'])
def test_retired_endpoint_support_api_cannot_be_called(name):
    assert not hasattr(retired, name)
