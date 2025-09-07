import requests

def test_classification():
    """
    Test the classification endpoint of the Flask application.
    """
    url = "http://127.0.0.1:5000/classify"
    sample_text = "This product is fantastic! I love it."
    response = requests.post(url, json={"text": sample_text})
    assert response.status_code == 200
    data = response.json()
    assert data["classification"] == "positive"
    print("Test passed!")


if __name__ == "__main__":
    test_classification()

