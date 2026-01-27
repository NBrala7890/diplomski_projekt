import os
import tarfile
from pathlib import Path


def is_probably_text(sample: bytes, min_text_ratio: float = 0.9) -> bool:
    """
    Vrlo gruba heuristika: ako je previše 'čudnih' bajtova, tretiramo kao binarno.
    """
    if not sample:
        return False

    # ako ima null bajtova, gotovo sigurno nije običan tekst
    if b'\x00' in sample:
        return False

    # dopušteni 'normalni' tekstualni znakovi
    text_bytes = sum(
        1
        for b in sample
        if b in (9, 10, 13) or 32 <= b <= 255  # tab, LF, CR ili 'printable'
    )
    ratio = text_bytes / len(sample)
    return ratio >= min_text_ratio


def decode_bytes(data: bytes) -> str:
    """
    Pokušaj dekodiranja s nekoliko kodnih stranica tipičnih za HR tekst.
    """
    for enc in ("utf-8", "cp1250", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # kao fallback, ignoriraj nevalidne znakove
    return data.decode("utf-8", errors="ignore")


def process_tgz_file(tgz_path: Path, out_file, stats: dict):
    """
    Otvara jednu .tgz arhivu, prolazi kroz sve datoteke
    i tekstualne zapise appenda u izlazni fajl.
    """
    print(f"Obrađujem arhivu: {tgz_path.name}")

    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            # pročitaj mali uzorak za heuristiku
            sample = f.read(4096)
            if not is_probably_text(sample):
                continue

            # učitaj ostatak
            rest = f.read()
            data = sample + rest

            text = decode_bytes(data).strip()
            if not text:
                continue

            # normaliziraj nove redove
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            # upiši u izlazni fajl + razdvoji dokumente praznim redom
            out_file.write(text)
            out_file.write("\n\n")

            stats["files_read"] += 1
            stats["bytes_written"] += len(text.encode("utf-8"))


def build_corpus(
    hanza_dir: str = "hanza",
    output_file: str = "corpus_raw.txt",
):
    hanza_path = Path(hanza_dir)
    if not hanza_path.is_dir():
        raise ValueError(f"Folder '{hanza_dir}' ne postoji ili nije direktorij.")

    tgz_files = sorted(hanza_path.glob("*.tgz"))
    if not tgz_files:
        raise ValueError(f"U folderu '{hanza_dir}' nema .tgz datoteka.")

    stats = {"archives": 0, "files_read": 0, "bytes_written": 0}

    with open(output_file, "w", encoding="utf-8") as out_f:
        for tgz_path in tgz_files:
            stats["archives"] += 1
            process_tgz_file(tgz_path, out_f, stats)

    print("\nGotovo!")
    print(f"Obrađeno arhiva: {stats['archives']}")
    print(f"Obrađeno tekstualnih datoteka: {stats['files_read']}")
    print(f"Ukupno upisanih bajtova (UTF-8): {stats['bytes_written']}")
    print(f"Izlazni fajl: {output_file}")


if __name__ == "__main__":
    # po potrebi promijeni putanje
    build_corpus(
        hanza_dir="hanza",
        output_file="corpus_raw.txt",
    )