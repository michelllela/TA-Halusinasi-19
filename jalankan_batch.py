# jalankan_batch.py
# Menjalankan SEMUA pertanyaan di pertanyaan.csv ke keempat model, memakai
# fungsi dan parameter yang SAMA dengan compare.py (website), lalu menyimpan
# hasilnya ke log_batch.csv. Letakkan file ini di folder yang sama dengan compare.py.
#
# Cara pakai (di terminal VS Code, folder proyek):
#   python jalankan_batch.py --uji 2          # uji coba: 2 pertanyaan saja
#   python jalankan_batch.py                  # jalankan semuanya
#   python jalankan_batch.py --data pertanyaan_dipecah.csv
#   python jalankan_batch.py --model 0 1      # hanya model ke-0 dan ke-1
#
# Aman dihentikan (Ctrl + C) dan dilanjutkan: pasangan (id, model) yang sudah
# berhasil tidak diulang. Urutan: model per model, supaya Ollama tidak bolak-balik
# memuat model (di CPU itu lambat).
import argparse
import csv
import io
import json
import os
import re
import time
from datetime import datetime

import compare  # memakai MODEL_LLM, PARAM, tanya_model, EMBED_MODEL yang sama

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "log_batch.csv")
KOLOM = [
    "waktu", "id_dataset", "destinasi", "kategori_dokumentasi", "kategori_pertanyaan",
    "pertanyaan", "ground_truth", "fakta_kunci", "model", "tag_model", "respons",
    "error", "skor_cosine", "fakta_cocok", "jumlah_fakta", "skor_fakta",
    "saran_label", "catatan_otomatis", "label_final", "detik", "parameter",
]

# kata umum yang diabaikan saat mencocokkan fakta kunci
ABAIKAN = {
    "kecamatan", "desa", "kabupaten", "toba", "sekitar", "per", "orang", "malam", "dengan",
    "yang", "dan", "di", "ke", "dari", "atau", "hingga", "mulai", "rp", "km", "meter",
    "dewasa", "kendaraan", "setiap", "hari", "untuk", "pada", "itu", "ini", "adalah",
}
POLA_MENOLAK = re.compile(
    r"(tidak tahu|tidak mengetahui|tidak memiliki informasi|tidak punya informasi|"
    r"tidak ada informasi|informasi (tersebut )?tidak (tersedia|ada)|"
    r"saya tidak (tahu|dapat|bisa|memiliki)|tidak dapat memberikan|tidak bisa memberikan)"
)


def norm(teks):
    t = str(teks).lower()
    t = re.sub(r"(?<=\d)\.(?=\d{3})", "", t)   # 5.000 -> 5000
    t = re.sub(r"(?<=\d),(?=\d)", "", t)       # 1,7 -> 17 (sama di kedua sisi)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.split()


def cocok(fakta, jawaban_tokens):
    """True jika >= 75% kata penting dari satu fakta ada di jawaban."""
    kata = [k for k in norm(fakta) if k not in ABAIKAN and len(k) > 1]
    if not kata:
        return False
    ada = sum(1 for k in kata if k in jawaban_tokens)
    return ada / len(kata) >= 0.75


def nilai_fakta(fakta_kunci, jawaban):
    daftar = [f.strip() for f in fakta_kunci.split(";") if f.strip()]
    tok = set(norm(jawaban))
    hit = sum(1 for f in daftar if cocok(f, tok))
    return hit, len(daftar)


def saran(jawaban, error, hit, total):
    if error or not jawaban:
        return "error", "Jawaban kosong atau error; jalankan ulang."
    catatan = ""
    menolak = bool(POLA_MENOLAK.search(jawaban.lower()))
    if menolak and hit == 0:
        return "menolak", "Terdeteksi frasa tidak tahu."
    if menolak:
        catatan = "Ada frasa menolak tetapi juga ada fakta cocok; cek manual."
    if total and hit == total:
        return "benar", catatan
    if hit > 0:
        return "benar sebagian", catatan
    return "salah", catatan


def baca_dataset(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        teks = f.read()
    baris1 = teks.split("\n", 1)[0]
    pemisah = ";" if baris1.count(";") > baris1.count(",") else ","
    rd = csv.DictReader(io.StringIO(teks), delimiter=pemisah)
    rd.fieldnames = [h.strip() for h in rd.fieldnames]
    return [r for r in rd if r.get("pertanyaan") and r.get("ground_truth")]


def sudah_selesai():
    selesai = set()
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    selesai.add((r["id_dataset"], r["tag_model"]))
    return selesai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pertanyaan.csv")
    ap.add_argument("--uji", type=int, default=0, help="jalankan hanya N pertanyaan pertama")
    ap.add_argument("--model", type=int, nargs="*", help="nomor model (0 sampai 3)")
    a = ap.parse_args()

    rows = baca_dataset(os.path.join(BASE, a.data))
    if a.uji:
        rows = rows[: a.uji]
    pilihan = a.model if a.model else list(range(len(compare.MODEL_LLM)))
    selesai = sudah_selesai()

    from sentence_transformers import SentenceTransformer, util
    emb = SentenceTransformer(compare.EMBED_MODEL)

    tugas = [(i, r) for i in pilihan for r in rows
             if (r.get("id", ""), compare.MODEL_LLM[i][1]) not in selesai]
    print(f"{len(rows)} pertanyaan x {len(pilihan)} model. Sisa dikerjakan: {len(tugas)}")
    if not tugas:
        print("Semua sudah selesai.")
        return

    baru = not os.path.exists(LOG)
    f = open(LOG, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f, fieldnames=KOLOM)
    if baru:
        w.writeheader()

    total_detik, n = 0.0, 0
    try:
        for i, r in tugas:
            nama, tag = compare.MODEL_LLM[i]
            jawaban, error, detik = compare.tanya_model(tag, r["pertanyaan"])
            skor = ""
            if jawaban:
                x = emb.encode(jawaban, convert_to_tensor=True)
                y = emb.encode(r["ground_truth"], convert_to_tensor=True)
                skor = round(float(util.cos_sim(x, y)), 4)
            hit, total = nilai_fakta(r.get("fakta_kunci", ""), jawaban or "")
            label, catatan = saran(jawaban, error, hit, total)
            w.writerow({
                "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id_dataset": r.get("id", ""), "destinasi": r.get("destinasi", ""),
                "kategori_dokumentasi": r.get("kategori_dokumentasi", ""),
                "kategori_pertanyaan": r.get("kategori_pertanyaan", ""),
                "pertanyaan": r["pertanyaan"], "ground_truth": r["ground_truth"],
                "fakta_kunci": r.get("fakta_kunci", ""), "model": nama, "tag_model": tag,
                "respons": jawaban or "", "error": error or "", "skor_cosine": skor,
                "fakta_cocok": hit, "jumlah_fakta": total,
                "skor_fakta": round(hit / total, 2) if total else "",
                "saran_label": label, "catatan_otomatis": catatan, "label_final": "",
                "detik": round(detik, 1), "parameter": json.dumps(compare.PARAM),
            })
            f.flush()
            n += 1
            total_detik += detik
            sisa = (len(tugas) - n) * (total_detik / n) / 60
            print(f"[{n}/{len(tugas)}] {nama} | {r.get('id','')} | {label} | "
                  f"{round(detik)} dtk | perkiraan sisa {sisa:.0f} menit")
    except KeyboardInterrupt:
        print("\nDihentikan. Jalankan lagi perintah yang sama untuk melanjutkan.")
    finally:
        f.close()


if __name__ == "__main__":
    main()
