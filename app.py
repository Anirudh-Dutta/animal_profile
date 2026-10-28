# app.py
# pip install flask torch torchvision pillow
# Run: python app.py   then open http://127.0.0.1:5000

import os, json, re, sqlite3, uuid
from functools import wraps
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms as T, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, g, session
from werkzeug.utils import secure_filename

DB_PATH = "tracker.db"
CKPT_PATH = "cat_id_best.pt"
UPLOAD_DIR = os.path.join("static", "uploads")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATCH_THRESHOLD = 0.55
IMG_SIZE = 128
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
INSTITUTE_EMAIL = re.compile(r"^[a-zA-Z][a-zA-Z.]*26101@iiitnr\.edu\.in$")
MIN_ROLL, MAX_ROLL = 261010201, 261010302

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("APP_SECRET", "dev-only-change-me")  # set APP_SECRET env var for real use
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # up to 5 photos per animal

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------- DB ----------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS animals (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            marks TEXT,
            status TEXT DEFAULT 'Healthy',
            place TEXT
        );
        CREATE TABLE IF NOT EXISTS animal_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_id TEXT NOT NULL REFERENCES animals(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            vector TEXT
        );
    """)
    conn.commit()
    conn.close()


def new_animal_id(animal_type):
    prefix = (animal_type[:1] or "A").upper()
    return f"{prefix}-{uuid.uuid4().hex[:6].upper()}"


# ---------------------------- Auth ----------------------------
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("authed"):
            return jsonify({"error": "not authenticated"}), 401
        return fn(*args, **kwargs)
    return wrapper


@app.route("/api/me", methods=["GET"])
def me():
    return jsonify({"authed": bool(session.get("authed")), "email": session.get("email")})


@app.route("/api/auth", methods=["POST"])
def auth():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    roll_raw = str(data.get("roll", "")).strip()

    if not INSTITUTE_EMAIL.match(email):
        return jsonify({"error": "Invalid email ID. Use name26101@iiitnr.edu.in."}), 400
    if not roll_raw.isdigit() or len(roll_raw) != 9:
        return jsonify({"error": "Invalid roll number format."}), 400
    roll = int(roll_raw)
    if not (MIN_ROLL <= roll <= MAX_ROLL):
        return jsonify({"error": f"Roll number must be between {MIN_ROLL} and {MAX_ROLL}."}), 400

    session["authed"] = True
    session["email"] = email
    return jsonify({"status": "ok"})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "ok"})


# ---------------------------- Model ----------------------------
class CatNet(nn.Module):
    def __init__(self, num_classes, emb_dim=128):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.emb = nn.Sequential(
            nn.Linear(backbone.last_channel, emb_dim, bias=False), nn.BatchNorm1d(emb_dim))
        self.W = nn.Parameter(torch.zeros(num_classes, emb_dim))  # unused at inference

    def embed(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return F.normalize(self.emb(x), dim=1)


_ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
_model = CatNet(num_classes=len(_ckpt["names"])).to(DEVICE)
_model.load_state_dict(_ckpt["model"])
_model.eval()

_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def embed_image(pil_img):
    x = _tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _model.embed(x)
    return emb.squeeze(0).cpu().tolist()


def validate_image(file_storage):
    if file_storage is None or file_storage.filename == "":
        return "image is required"
    if file_storage.mimetype not in ALLOWED_IMAGE_TYPES:
        return f"unsupported file type: {file_storage.mimetype}"
    return None


def save_photo(animal_id, file_storage):
    folder = os.path.join(UPLOAD_DIR, animal_id)
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file_storage.filename)}"
    disk_path = os.path.join(folder, filename)
    file_storage.stream.seek(0)
    img = Image.open(file_storage.stream).convert("RGB")
    img.save(disk_path)
    url_path = f"/uploads/{animal_id}/{filename}"
    return disk_path, url_path, img


# ---------------------------- Static ----------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------- Animals: CRUD ----------------------------
@app.route("/api/animals", methods=["GET"])
@login_required
def list_animals():
    db = get_db()
    rows = db.execute("SELECT * FROM animals").fetchall()
    out = []
    for r in rows:
        photos = db.execute(
            "SELECT path FROM animal_photos WHERE animal_id = ?", (r["id"],)
        ).fetchall()
        a = dict(r)
        a["photos"] = [p["path"] for p in photos]
        out.append(a)
    return jsonify(out)


@app.route("/api/animals", methods=["POST"])
@login_required
def create_animal():
    name = request.form.get("name", "").strip()
    animal_type = request.form.get("type", "Other").strip() or "Other"
    marks = request.form.get("marks", "").strip()
    status = request.form.get("status", "Healthy").strip() or "Healthy"
    files = request.files.getlist("images")

    if not name:
        return jsonify({"error": "name is required"}), 400

    for f in files:
        err = validate_image(f)
        if err:
            return jsonify({"error": err}), 400

    animal_id = new_animal_id(animal_type)
    db = get_db()
    db.execute(
        "INSERT INTO animals (id, name, type, marks, status) VALUES (?,?,?,?,?)",
        (animal_id, name, animal_type, marks, status),
    )

    for f in files:
        try:
            _, url_path, pil_img = save_photo(animal_id, f)
            vector = json.dumps(embed_image(pil_img)) if animal_type == "Cat" else None
            db.execute(
                "INSERT INTO animal_photos (animal_id, path, vector) VALUES (?,?,?)",
                (animal_id, url_path, vector),
            )
        except Exception:
            db.rollback()
            return jsonify({"error": "could not process one of the uploaded images"}), 400

    db.commit()
    return jsonify({"id": animal_id}), 201


@app.route("/api/animals/<animal_id>", methods=["PUT"])
@login_required
def update_animal(animal_id):
    data = request.get_json(silent=True) or {}
    allowed = {"name", "type", "marks", "status", "place"}
    fields = {k: v for k, v in data.items() if k in allowed}
    if not fields:
        return jsonify({"error": f"no updatable fields provided (allowed: {sorted(allowed)})"}), 400

    db = get_db()
    exists = db.execute("SELECT 1 FROM animals WHERE id = ?", (animal_id,)).fetchone()
    if not exists:
        return jsonify({"error": "animal not found"}), 404

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    db.execute(f"UPDATE animals SET {set_clause} WHERE id = ?", (*fields.values(), animal_id))
    db.commit()
    return jsonify({"status": "ok"})


@app.route("/api/animals/<animal_id>", methods=["DELETE"])
@login_required
def delete_animal(animal_id):
    db = get_db()
    cur = db.execute("DELETE FROM animals WHERE id = ?", (animal_id,))
    db.commit()
    if cur.rowcount == 0:
        return jsonify({"error": "animal not found"}), 404
    return jsonify({"status": "ok"})


# ---------------------------- Scan (identify a cat) ----------------------------
@app.route("/api/scan", methods=["POST"])
@login_required
def scan():
    file = request.files.get("image")
    err = validate_image(file)
    if err:
        return jsonify({"error": err}), 400

    try:
        query_img = Image.open(file.stream)
        query = torch.tensor(embed_image(query_img))
    except Exception:
        return jsonify({"error": "could not process image"}), 400

    db = get_db()
    rows = db.execute(
        """SELECT animals.id AS animal_id, animals.name, animals.marks,
                  animals.status, animals.place, animal_photos.path, animal_photos.vector
           FROM animal_photos
           JOIN animals ON animals.id = animal_photos.animal_id
           WHERE animals.type = 'Cat' AND animal_photos.vector IS NOT NULL"""
    ).fetchall()

    by_animal = {}
    for row in rows:
        score = float(F.cosine_similarity(query, torch.tensor(json.loads(row["vector"])), dim=0))
        entry = by_animal.setdefault(row["animal_id"], {
            "animal": {
                "id": row["animal_id"], "name": row["name"], "marks": row["marks"],
                "status": row["status"], "place": row["place"],
            },
            "photos": [], "best_photo": None, "score": -1.0,
        })
        entry["photos"].append(row["path"])
        if score > entry["score"]:
            entry["score"] = score
            entry["best_photo"] = row["path"]

    results = sorted(by_animal.values(), key=lambda e: e["score"], reverse=True)
    for r in results:
        r["score"] = round(r["score"], 3)
        r["confident"] = r["score"] >= MATCH_THRESHOLD

    return jsonify({"results": results})


if __name__ == "__main__":
    init_db()
    app.run(debug=False, port=5000)