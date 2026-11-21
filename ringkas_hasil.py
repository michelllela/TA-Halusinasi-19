# ringkas_hasil.py
# Membaca log_batch.csv lalu menghitung ringkasan untuk Bab IV dan seminar.
# Label yang dipakai: kolom label_final kalau sudah diisi tim, kalau kosong
# memakai saran_label (otomatis, BELUM diperiksa).
#
#   python ringkas_hasil.py
import csv
import os
from collections import defaultdict
from math import sqrt

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "log_batch.csv")
OUT = os.path.join(BASE, "hasil_ringkasan.csv")
NILAI = {"benar": 1.0, "benar sebagian": 0.5, "salah": 0.0}


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = sqrt(sum((a - mx) ** 2 for a in x))
    sy = sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def rata(v):
    return sum(v) / len(v) if v else None


def fmt(v, persen=False):
    if v is None:
        return "-"
    return f"{v*100:.1f}%" if persen else f"{v:.3f}"


def hitung(baris):
    c = defaultdict(int)
    cos_per = defaultdict(list)
    for r in baris:
        c[r["label"]] += 1
        if r["cos"] is not None:
            cos_per[r["label"]].append(r["cos"])
    n = len(baris)
    jawab = n - c["menolak"]
    pasang = [(r["cos"], NILAI[r["label"]]) for r in baris
              if r["cos"] is not None and r["label"] in NILAI]
    return {
        "n": n, "benar": c["benar"], "sebagian": c["benar sebagian"],
        "salah": c["salah"], "menolak": c["menolak"],
        "tingkat_halusinasi": (c["salah"] / jawab) if jawab else None,
        "tingkat_penolakan": (c["menolak"] / n) if n else None,
        "cos_benar": rata(cos_per["benar"]), "cos_salah": rata(cos_per["salah"]),
        "korelasi_cos_label": pearson([p[0] for p in pasang], [p[1] for p in pasang]),
    }


def main():
    data = []
    with open(LOG, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            label = (r.get("label_final") or "").strip().lower() or r.get("saran_label", "")
            if label not in ("benar", "benar sebagian", "salah", "menolak"):
                continue
            try:
                cos = float(r["skor_cosine"])
            except (ValueError, TypeError):
                cos = None
            data.append({"model": r["model"], "kelompok": r.get("kategori_dokumentasi", ""),
                         "label": label, "cos": cos,
                         "manual": bool((r.get("label_final") or "").strip())})
    if not data:
        print("Belum ada data di log_batch.csv.")
        return
    manual = sum(1 for d in data if d["manual"])
    print(f"{len(data)} jawaban dihitung; {manual} berlabel manual, "
          f"{len(data) - manual} masih memakai saran otomatis.\n")

    hasil = []
    kunci = [("semua", None)] + [(k, k) for k in sorted({d["kelompok"] for d in data if d["kelompok"]})]
    for model in sorted({d["model"] for d in data}):
        for nama_k, k in kunci:
            sub = [d for d in data if d["model"] == model and (k is None or d["kelompok"] == k)]
            if sub:
                h = hitung(sub)
                h.update({"model": model, "kelompok": nama_k})
                hasil.append(h)

    kol = ["model", "kelompok", "n", "benar", "sebagian", "salah", "menolak",
           "tingkat_halusinasi", "tingkat_penolakan", "cos_benar", "cos_salah", "korelasi_cos_label"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=kol)
        w.writeheader()
        for h in hasil:
            w.writerow({k: h[k] for k in kol})

    print(f"{'Model':26}{'Kelompok':10}{'n':>4}{'B':>4}{'BS':>4}{'S':>4}{'M':>4}"
          f"{'Halusinasi':>12}{'Penolakan':>11}{'cosB':>8}{'cosS':>8}{'r':>8}")
    for h in hasil:
        print(f"{h['model']:26}{h['kelompok']:10}{h['n']:>4}{h['benar']:>4}{h['sebagian']:>4}"
              f"{h['salah']:>4}{h['menolak']:>4}{fmt(h['tingkat_halusinasi'], True):>12}"
              f"{fmt(h['tingkat_penolakan'], True):>11}{fmt(h['cos_benar']):>8}"
              f"{fmt(h['cos_salah']):>8}{fmt(h['korelasi_cos_label']):>8}")
    print("\nB=benar, BS=benar sebagian, S=salah, M=menolak, cosB/cosS=rata-rata cosine "
          "jawaban benar/salah, r=korelasi cosine dengan label.")
    print(f"Tersimpan di {OUT}")


if __name__ == "__main__":
    main()
