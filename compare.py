# compare.py
# Back-end perbandingan 4 LLM (Llama, Qwen, Mistral, Gemma) untuk website TA.
# Dipasang ke app.py lewat Blueprint, jadi halaman dan rute lama tidak tersentuh.
#
# Rute:
#   GET  /bandingkan   -> halaman perbandingan
#   POST /api/gt       -> cari ground truth untuk pertanyaan
#   POST /api/jawab    -> tanya SATU model + hitung skor cosine + simpan log
import csv
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request

bp = Blueprint("compare", __name__)

BASE = os.path.dirname(os.path.abspath(__file__))
FILE_DATASET = os.path.join(BASE, "pertanyaan.csv")
FILE_LOG = os.path.join(BASE, "log_perbandingan.csv")
COL_P = "pertanyaan"        # nama kolom pertanyaan di pertanyaan.csv
COL_GT = "ground_truth"     # nama kolom ground truth di pertanyaan.csv

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_CHAT = "http://localhost:11434/api/chat"
TIMEOUT = 300
AMBANG_PENCARIAN = 0.75     # kemiripan minimum pertanyaan pengguna vs dataset

# (nama tampilan, nama model di Ollama). Cocokkan dengan hasil: ollama list
MODEL_LLM = [
    ("Llama 3.1 8B", "llama3.1:8b"),
    ("Qwen 3 8B", "qwen3:8b"),
    ("Mistral (Ministral 3 8B)", "ministral-3:8b"),
    ("Gemma 3 4B", "gemma3:4b"),
]

# Parameter HARUS sama untuk semua model agar perbandingan adil.
PARAM = {"temperature": 0.2, "top_p": 0.9, "num_predict": 300,
         "num_ctx": 4096, "seed": 42}
SYSTEM_PROMPT = (
    "Kamu adalah asisten informasi wisata. Jawab dalam bahasa Indonesia, "
    "singkat, dan hanya berdasarkan fakta yang kamu ketahui. "
    "Jika tidak tahu, katakan tidak tahu."
)

_emb_model = None
_rows = None
_emb_rows = None


def _muat():
    """Muat model embedding dan dataset (sekali saja)."""
    global _emb_model, _rows, _emb_rows
    if _emb_model is None:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(EMBED_MODEL)
    if _rows is None:
        with open(FILE_DATASET, encoding="utf-8-sig", newline="") as f:
            teks = f.read()
        baris1 = teks.split("\n", 1)[0]
        pemisah = ";" if baris1.count(";") > baris1.count(",") else ","
        reader = csv.DictReader(io.StringIO(teks), delimiter=pemisah)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        rows = [r for r in reader if r.get(COL_P) and r.get(COL_GT)]
        if not rows:
            raise ValueError(
                f"pertanyaan.csv kosong atau kolom '{COL_P}'/'{COL_GT}' tidak ada. "
                f"Kolom yang terbaca: {reader.fieldnames}"
            )
        _emb_rows = _emb_model.encode([r[COL_P] for r in rows], convert_to_tensor=True)
        _rows = rows


def cari_gt(pertanyaan):
    """Kembalikan (baris_dataset atau None, kemiripan_pertanyaan)."""
    from sentence_transformers import util
    _muat()
    q = _emb_model.encode(pertanyaan, convert_to_tensor=True)
    sims = util.cos_sim(q, _emb_rows)[0]
    i = int(sims.argmax())
    s = float(sims[i])
    return (_rows[i], s) if s >= AMBANG_PENCARIAN else (None, s)


def tanya_model(tag, pertanyaan):
    """Kirim pertanyaan ke satu model. Return (jawaban, error, detik)."""
    payload = {
        "model": tag,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pertanyaan},
        ],
        "options": PARAM,
    }
    if tag.startswith("qwen3"):
        payload["think"] = False
    req = urllib.request.Request(
        OLLAMA_CHAT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    mulai = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            teks = json.loads(resp.read().decode("utf-8"))["message"]["content"]
        teks = re.sub(r"<think>.*?</think>", "", teks, flags=re.DOTALL).strip()
        return teks, None, time.time() - mulai
    except urllib.error.HTTPError as e:
        isi = e.read().decode("utf-8", errors="ignore")[:200]
        return None, f"Model {tag} bermasalah: {isi}", 0.0
    except urllib.error.URLError:
        return None, "Ollama tidak dapat dihubungi. Pastikan Ollama berjalan.", 0.0
    except TimeoutError:
        return None, "Waktu tunggu habis. Coba lagi.", 0.0
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {e}", 0.0


def _simpan_log(baris):
    """Simpan satu baris ke log_perbandingan.csv. Gagal simpan tidak boleh merusak chat."""
    try:
        baru = not os.path.exists(FILE_LOG)
        with open(FILE_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(baris.keys()))
            if baru:
                w.writeheader()
            w.writerow(baris)
    except Exception as e:  # noqa: BLE001
        print("Gagal menyimpan log:", e)


@bp.route("/bandingkan")
def halaman():
    return render_template("bandingkan.html")


@bp.route("/api/gt", methods=["POST"])
def api_gt():
    data = request.get_json(silent=True) or {}
    pesan = str(data.get("message", "")).strip()
    if not pesan:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400
    try:
        row, s = cari_gt(pesan)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Gagal membaca dataset: {e}"}), 500
    return jsonify({
        "ada": row is not None,
        "ground_truth": row[COL_GT] if row else "",
        "pertanyaan_dataset": row[COL_P] if row else "",
        "destinasi": row.get("destinasi", "") if row else "",
        "kategori": row.get("kategori_dokumentasi", "") if row else "",
        "kemiripan": round(s, 3),
        "model": [nama for nama, _ in MODEL_LLM],
    })


@bp.route("/api/jawab", methods=["POST"])
def api_jawab():
    data = request.get_json(silent=True) or {}
    pesan = str(data.get("message", "")).strip()
    try:
        idx = int(data.get("model", -1))
        nama, tag = MODEL_LLM[idx]
    except (ValueError, IndexError):
        return jsonify({"error": "Model tidak dikenal."}), 400
    if not pesan:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    jawaban, error, detik = tanya_model(tag, pesan)

    skor, row, s = None, None, 0.0
    try:
        row, s = cari_gt(pesan)
        if jawaban and row:
            from sentence_transformers import util
            a = _emb_model.encode(jawaban, convert_to_tensor=True)
            b = _emb_model.encode(row[COL_GT], convert_to_tensor=True)
            skor = round(float(util.cos_sim(a, b)), 4)
    except Exception as e:  # noqa: BLE001
        error = error or f"Gagal menghitung skor: {e}"

    _simpan_log({
        "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id_dataset": row.get("id", "") if row else "",
        "destinasi": row.get("destinasi", "") if row else "",
        "kategori_dokumentasi": row.get("kategori_dokumentasi", "") if row else "",
        "pertanyaan_pengguna": pesan,
        "pertanyaan_dataset": row[COL_P] if row else "",
        "kemiripan_pertanyaan": round(s, 4),
        "ground_truth": row[COL_GT] if row else "",
        "model": nama,
        "tag_model": tag,
        "respons": jawaban or "",
        "error": error or "",
        "skor_cosine": skor if skor is not None else "",
        "detik": round(detik, 1),
        "parameter": json.dumps(PARAM),
    })
    return jsonify({"model": nama, "jawaban": jawaban, "error": error,
                    "skor": skor, "detik": round(detik, 1)})
