import pytest
from app import app
from unittest.mock import patch

# Setup client test cho Flask
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

# Test trang chủ load OK
def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Dự đoán".encode("utf-8") in response.data

# Test API /ask hoạt động với mock
@patch("app.ask_openrouter", return_value="Phản hồi giả lập cho test")
def test_ask_api(mocked, client):
    response = client.post("/ask", json={"prompt": "HPV là gì?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "reply" in data
    assert data["reply"] == "Phản hồi giả lập cho test"

# Test hiển thị lịch sử khi có session
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
    assert b"L" in response.data  # Test nhẹ xem có hiện gì
