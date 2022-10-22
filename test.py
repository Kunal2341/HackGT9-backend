from starlette.testclient import TestClient

from app import app

client = TestClient(app)

def test_users_endpoint():
    resp = client.post("/update/1.jpg")

    # assert resp.status_code == 200

test_users_endpoint()