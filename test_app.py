# test_app.py
# Run: pytest test_app.py
import io
import pytest
from PIL import Image
import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads"))
    app_module.init_db()
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def fake_image_bytes(color=(120, 80, 60)):
    img = Image.new("RGB", (128, 128), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def login(client):
    return client.post("/api/auth", json={"email": "test26101@iiitnr.edu.in", "roll": "261010250"})


def test_me_before_login(client):
    r = client.get("/api/me")
    assert r.get_json()["authed"] is False


def test_auth_rejects_bad_email(client):
    r = client.post("/api/auth", json={"email": "bad@gmail.com", "roll": "261010250"})
    assert r.status_code == 400


def test_auth_rejects_bad_roll(client):
    r = client.post("/api/auth", json={"email": "test26101@iiitnr.edu.in", "roll": "999999999"})
    assert r.status_code == 400


def test_auth_success_sets_session(client):
    r = login(client)
    assert r.status_code == 200
    assert client.get("/api/me").get_json()["authed"] is True


def test_animals_require_login(client):
    r = client.get("/api/animals")
    assert r.status_code == 401


def test_create_list_update_delete_animal(client):
    login(client)

    r = client.post("/api/animals", data={
        "name": "Milo", "type": "Cat", "marks": "white paws", "status": "Healthy",
        "images": (fake_image_bytes(), "milo.jpg"),
    }, content_type="multipart/form-data")
    assert r.status_code == 201
    animal_id = r.get_json()["id"]

    r = client.get("/api/animals")
    assert any(a["id"] == animal_id for a in r.get_json())

    r = client.put(f"/api/animals/{animal_id}", json={"place": "near the library"})
    assert r.status_code == 200
    updated = next(a for a in client.get("/api/animals").get_json() if a["id"] == animal_id)
    assert updated["place"] == "near the library"

    r = client.delete(f"/api/animals/{animal_id}")
    assert r.status_code == 200
    assert not any(a["id"] == animal_id for a in client.get("/api/animals").get_json())


def test_create_animal_rejects_bad_file_type(client):
    login(client)
    r = client.post("/api/animals", data={
        "name": "Bruno", "type": "Dog", "status": "Healthy",
        "images": (io.BytesIO(b"not an image"), "x.txt"),
    }, content_type="multipart/form-data")
    assert r.status_code == 400


def test_scan_requires_login(client):
    r = client.post("/api/scan", data={"image": (fake_image_bytes(), "cat.jpg")},
                     content_type="multipart/form-data")
    assert r.status_code == 401


def test_scan_with_no_registered_cats(client):
    login(client)
    r = client.post("/api/scan", data={"image": (fake_image_bytes(), "cat.jpg")},
                     content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["results"] == []


def test_update_nonexistent_animal(client):
    login(client)
    r = client.put("/api/animals/does-not-exist", json={"status": "Missing"})
    assert r.status_code == 404