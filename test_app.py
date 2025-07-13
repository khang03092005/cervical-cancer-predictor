import pytest
from app import app
from unittest.mock import patch

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Dự đoán".encode("utf-8") in response.data

@patch("app.ask_openrouter", return_value="Phản hồi giả lập cho test")
def test_ask_api(mocked, client):
    response = client.post("/ask", json={"prompt": "HPV là gì?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "reply" in data
    assert data["reply"] == "Phản hồi giả lập cho test"

def test_history_page(client):
    with client.session_transaction() as sess:
        sess["history"] = [
            {
                "input": {"Age": "30"},
                "result": 1,
                "proba": 60.0,
                "advice": "Lời khuyên mẫu",
                "timestamp": "13-07-2025 10:00"
            }
        ]
    response = client.get("/history")
    assert response.status_code == 200
    assert b"L" in response.data  


def test_monitor_page(client):
    with client.session_transaction() as sess:
        sess["history"] = [
            {"proba": 75.0, "timestamp": "13-07-2025 10:00"}
        ]
    response = client.get("/monitor")
    assert response.status_code == 200
    assert b"L" in response.data

def test_clear_history(client):
    with client.session_transaction() as sess:
        sess["history"] = [{"result": 1}]
    response = client.get("/clear_history", follow_redirects=True)
    with client.session_transaction() as sess_after:
        assert "history" not in sess_after

def test_submit_empty_form(client):
    response = client.post("/", data={})
    html = response.data.decode("utf-8")
    assert "Vui lòng nhập ít nhất một giá trị" in html or "trống" in html

