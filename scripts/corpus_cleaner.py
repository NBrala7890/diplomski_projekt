from pathlib import Path


def build_translation_table():
    """
    Mapiranje znakova na hrvatska slova:
      { -> š
      } -> ć
      ~ -> č
      | -> đ
      ` -> ž
      @ -> ž
      ^ -> č
      \ -> đ
      [ -> š
      ] -> ć
    """
    mapping = {
        ord('{'): 'š',
        ord('}'): 'ć',
        ord('~'): 'č',
        ord('|'): 'đ',
        ord('`'): 'ž',
        ord('@'): 'ž',
        ord('^'): 'č',
        ord('\\'): 'đ',  # backslash!
        ord('['): 'š',
        ord(']'): 'ć',
    }
    return mapping


def clean_line(line: str, translation_table) -> str | None:
    """
    Očisti jedan redak:
    - zamijeni posebne znakove prema tablici
    - izbaci početak retka dok ne dođemo do prvog slova
    - odbaci retke koji nemaju slova ili su prazni
    """
    # Makni newline na kraju, zadrži ostatak
    line = line.rstrip("\n\r")

    # Zamjene znakova -> hrvatska slova
    line = line.translate(translation_table)

    # Pronađi indeks prvog slova (bilo koje Unicode slovo)
    first_letter_idx = None
    for i, ch in enumerate(line):
        if ch.isalpha():
            first_letter_idx = i
            break

    # Ako nema slova uopće, odbaci redak
    if first_letter_idx is None:
        return None

    # Odsijeci sve prije prvog slova
    line = line[first_letter_idx:]

    # Makni leading/trailing whitespace nakon odsijecanja
    line = line.strip()

    # Ako je nakon svega redak prazan, odbaci
    if not line:
        return None

    return line


def clean_corpus(
    raw_corpus_path: str = "corpus_raw.txt",
    extra_corpus_path: str = "jutarnji-corpus-2024-2025-clean.txt",
    output_path: str = "corpus_clean.txt",
):
    raw_path = Path(raw_corpus_path)
    extra_path = Path(extra_corpus_path)

    if not raw_path.is_file():
        raise FileNotFoundError(f"Ne postoji ulazna datoteka: {raw_corpus_path}")

    if not extra_path.is_file():
        raise FileNotFoundError(f"Ne postoji dodatna datoteka: {extra_corpus_path}")

    translation_table = build_translation_table()

    cleaned_lines = 0
    skipped_lines = 0

    with raw_path.open("r", encoding="utf-8") as in_f, \
         Path(output_path).open("w", encoding="utf-8") as out_f:

        for line in in_f:
            cleaned = clean_line(line, translation_table)
            if cleaned is None:
                skipped_lines += 1
                continue

            out_f.write(cleaned + "\n")
            cleaned_lines += 1

        # Osiguraj praznu liniju prije dodavanja dodatnog korpusa
        out_f.write("\n")

        # Dodaj sav tekst iz dodatnog, već očišćenog korpusa
        with extra_path.open("r", encoding="utf-8") as extra_f:
            for line in extra_f:
                out_f.write(line)

    print("Gotovo čišćenje korpusa.")
    print(f"Sačuvano redaka: {cleaned_lines}")
    print(f"Odbijeno/redaka bez slova ili praznih: {skipped_lines}")
    print(f"Izlazni fajl: {output_path}")


if __name__ == "__main__":
    clean_corpus(
        raw_corpus_path="corpus_raw.txt",
        extra_corpus_path="jutarnji-corpus-2024-2025-clean.txt",
        output_path="corpus_clean.txt",
    )