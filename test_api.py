import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Image Classification API is running"}
    print("✓ Root endpoint test passed")

def test_predict_endpoint_exists():
    """Check if predict endpoint exists"""
    # Send empty data - should get validation error (422) not 404
    response = client.post("/predict", data={})
    assert response.status_code == 422  # Validation error expected
    print("✓ Predict endpoint exists (validation error as expected)")

def test_api_structure():
    """Basic API structure test"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    print("✓ API structure test passed")

if __name__ == "__main__":
    print("Running API tests...")
    try:
        test_read_root()
        test_predict_endpoint_exists()
        test_api_structure()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)