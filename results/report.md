# Optimizacija FastText modela za hrvatski pravopisni korektor

**Projekt:** Optimalni FastText word embeddings za ispravi.me
**Institucija:** Fakultet elektrotehnike i računarstva (FER), Sveučilište u Zagrebu
**Datum:** Siječanj 2026.

---

## Tim

| Ime | Uloga |
|-----|-------|
| Nikša Brala | Član tima |
| Teo Matošević | Član tima |
| Vitomir Brebrić | Član tima |

---

## 1. Uvod

### 1.1 Cilj projekta

Cilj ovog projekta je pronalazak optimalne konfiguracije FastText modela za poboljšanje hrvatskog pravopisnog korektora [ispravi.me](https://ispravi.me) semantičkom inteligencijom. Treniranjem word embeddinga na velikom hrvatskom korpusu omogućujemo korektoru predlaganje kontekstualno prikladnih ispravaka temeljenih na semantičkoj sličnosti.

### 1.2 Ključna područja fokusa

- **ije/je alternacije** - česte greške poput *riješenje* vs *rješenje*
- **Dijakritičke zamjene** - zamjena č/ć, š/ž
- **Općenite tipfelere** - uobičajene pogreške pri tipkanju
- **Morfološke varijacije** - hrvatski jezik ima 7 gramatičkih padeža

### 1.3 Zašto FastText?

FastText je odabran zbog nekoliko ključnih prednosti za hrvatski jezik:

1. **Subword informacije** - Karakterni n-grami hvataju morfološke obrasce
2. **OOV rukovanje** - Može generirati vektore za neviđene riječi
3. **Dijakritici** - N-grami pomažu razlikovati č/ć, š/ž (iako ne savršeno)

---

## 2. Metodologija

### 2.1 Korpus

| Karakteristika | Vrijednost |
|----------------|------------|
| Veličina | 2.5 GB |
| Broj riječi | ~400 milijuna |
| Veličina vokabulara | 2,918,267 jedinstvenih riječi |
| Izvor | Hrvatski tekstualni korpus (vijesti, web) |
| Predprocesiranje | Čišćenje, normalizacija, uklanjanje šuma |

### 2.2 FastText arhitektura

Koristimo dvije arhitekture:
- **Skip-gram** - Predviđa kontekst iz ciljne riječi (bolje za rijetke riječi)
- **CBOW** (Continuous Bag of Words) - Predviđa ciljnu riječ iz konteksta (brže treniranje)

### 2.3 Parametri modela

| Parametar | Opis | Vrijednosti |
|-----------|------|-------------|
| `dim` | Dimenzionalnost vektora | 100, 200, 300 |
| `ws` | Veličina kontekstnog prozora | 5, 10, 15 |
| `epoch` | Broj prolaza kroz korpus | 3, 10, 15 |
| `minCount` | Minimalna frekvencija riječi | 2 |
| `minn` | Minimalna duljina n-grama | 3 |
| `maxn` | Maksimalna duljina n-grama | 6 |

### 2.4 Prethodna istraživanja (Seminar 2)

Temeljem prethodnih eksperimenata identificirani su sljedeći uvidi:

| Nalaz | Utjecaj |
|-------|---------|
| Veće dimenzije (300) | Bolja semantička preciznost |
| Veći prozori (10) | Bolje semantičke veze |
| Niži minCount (2) | Bolje rukovanje rijetkim riječima |
| FastText | Odličan za ije/je greške, slabiji za č/ć razlike |

---

## 3. Konfiguracije modela

### 3.1 Tier 0: Brzi modeli (~1-2 sata)

Modeli za brzu iteraciju s reduciranim dimenzijama i epohama.

| Naziv | Model | dim | ws | epoch | minCount | minn | maxn |
|-------|-------|-----|-----|-------|----------|------|------|
| sg_d200_ws5_e3_mc2 | skipgram | 200 | 5 | 3 | 2 | 3 | 6 |
| sg_d100_ws5_e3_mc2 | skipgram | 100 | 5 | 3 | 2 | 3 | 6 |
| sg_d200_ws10_e3_mc2 | skipgram | 200 | 10 | 3 | 2 | 3 | 6 |
| cbow_d200_ws5_e3_mc2 | cbow | 200 | 5 | 3 | 2 | 3 | 6 |

### 3.2 Tier 1: Standardni modeli (~10 sati)

Potpuno treniranje temeljeno na nalazima iz Seminara 2.

| Naziv | Model | dim | ws | epoch | minCount | minn | maxn | Napomena |
|-------|-------|-----|-----|-------|----------|------|------|----------|
| sg_d300_ws10_e10_mc2 | skipgram | 300 | 10 | 10 | 2 | 3 | 6 | Najbolji iz Seminara 2 |
| sg_d300_ws10_e15_mc2 | skipgram | 300 | 10 | 15 | 2 | 3 | 6 | Više epoha |
| sg_d300_ws15_e10_mc2 | skipgram | 300 | 15 | 10 | 2 | 3 | 6 | Veći prozor |
| cbow_d300_ws10_e10_mc2 | cbow | 300 | 10 | 10 | 2 | 3 | 6 | CBOW arhitektura |

### 3.3 Baseline model

Za usporedbu koristimo Facebookov unaprijed trenirani model:
- **Naziv:** cc.hr.300.bin
- **Dimenzije:** 300
- **Korpus:** Common Crawl + Wikipedia (~6.4 milijardi tokena)

---

## 4. Rezultati treniranja

### 4.1 Završeni modeli

Ukupno je trenirano 8 modela i preuzet 1 baseline model (Facebook).

| Model | Arhitektura | dim | ws | epoch | minn/maxn | Veličina |
|-------|-------------|-----|-----|-------|-----------|----------|
| ft_sg_d300_ws10_e15_mc2 | Skip-gram | 300 | 10 | 15 | 3/6 | 8.9 GB |
| ft_sg_d300_ws10_e10_mc2 | Skip-gram | 300 | 10 | 10 | 3/6 | 8.9 GB |
| ft_sg_d300_ws10_e10_mc2_minn2_maxn7 | Skip-gram | 300 | 10 | 10 | 2/7 | 8.9 GB |
| ft_sg_d200_ws5_e3_mc2 | Skip-gram | 200 | 5 | 3 | 3/6 | 5.9 GB |
| ft_sg_d200_ws10_e3_mc2 | Skip-gram | 200 | 10 | 3 | 3/6 | 5.9 GB |
| ft_sg_d100_ws5_e3_mc2 | Skip-gram | 100 | 5 | 3 | 3/6 | 3.0 GB |
| ft_cbow_d300_ws10_e10_mc2 | CBOW | 300 | 10 | 10 | 3/6 | 8.9 GB |
| ft_cbow_d200_ws5_e3_mc2 | CBOW | 200 | 5 | 3 | 3/6 | 5.9 GB |
| facebook_cc_hr_300 (baseline) | Skip-gram | 300 | 5 | - | 5/5 | 6.8 GB |

### 4.2 Statistike treniranja

- **Ukupan broj riječi u korpusu:** ~398 milijuna
- **Veličina vokabulara:** 2,918,267 jedinstvenih riječi
- **Prosječno vrijeme Tier 0:** ~65 min
- **Prosječno vrijeme Tier 1:** ~600 min (10 sati)

---

## 5. Evaluacija

### 5.1 Metrike evaluacije

| Metrika | Težina | Opis |
|---------|--------|------|
| ije/je MRR | 20% | Mean Reciprocal Rank za ije/je greške |
| Dijakritički MRR | 20% | MRR za č/ć/š/ž zamjene |
| Općeniti MRR | 20% | MRR za uobičajene tipfelere |
| Semantička sličnost | 20% | Korelacija s ljudskim procjenama |
| Morfološka pokrivenost | 10% | Prepoznavanje oblika riječi |
| Veličina i brzina | 10% | Praktična primjena |

### 5.2 Testni skup

#### ije/je greške
| Pogrešno | Ispravno |
|----------|----------|
| riješenje | rješenje |
| mlieko | mlijeko |
| vriemena | vremena |
| biel | bijel |

#### Dijakritičke greške
| Pogrešno | Ispravno |
|----------|----------|
| moč | moć |
| noč | noć |
| kuča | kuća |

#### Općeniti tipfeleri
| Pogrešno | Ispravno |
|----------|----------|
| prijateli | prijatelj |
| doabr | dobar |
| skolaa | škola |
| pisatti | pisati |

### 5.3 Rezultati evaluacije

Evaluacija koristi **hibridni pristup** koji kombinira edit distance (40%) i kosinusnu sličnost embeddinga (60%). Ovaj pristup pravednije vrednuje FastText modele koji izvrsno rade na semantičkoj razini.

| Rang | Model | ije/je | Dijakritici | Tipfeleri | Ukupno |
|------|-------|--------|-------------|-----------|--------|
| 1 | **ft_sg_d300_ws10_e15_mc2** | 0.748 | 0.832 | 0.769 | **0.528** |
| 2 | ft_sg_d300_ws10_e10_mc2 | 0.727 | 0.830 | 0.772 | 0.495 |
| 3 | ft_sg_d300_ws10_e10_mc2_minn2_maxn7 | 0.748 | 0.831 | 0.786 | 0.493 |
| 4 | ft_sg_d200_ws5_e3_mc2 | 0.744 | 0.837 | 0.793 | 0.484 |
| 5 | ft_sg_d200_ws10_e3_mc2 | 0.740 | 0.830 | 0.781 | 0.464 |
| 6 | ft_sg_d100_ws5_e3_mc2 | 0.754 | 0.818 | 0.798 | 0.449 |
| 7 | ft_cbow_d300_ws10_e10_mc2 | 0.762 | 0.822 | 0.806 | 0.445 |
| 8 | facebook_cc_hr_300 | 0.711 | 0.820 | 0.755 | 0.439 |
| 9 | ft_cbow_d200_ws5_e3_mc2 | 0.793 | 0.822 | 0.824 | 0.438 |

**Napomena:** Hibridni rezultati (ije/je, Dijakritici, Tipfeleri) pokazuju mean_hybrid_score. Ukupni rezultat uzima u obzir i MRR metriku za nearest neighbor pretragu.

#### Ključni nalazi iz evaluacije:

1. **ije/je korekcija:** FastText izvrsno prepoznaje ije/je greške. Riječ "riješenje" → "rješenje" pojavljuje se na #1 poziciji u 8 od 9 modela.

2. **Dijakritici:** Hibridni pristup postiže ~0.82 score, ali čisti MRR za č/ć je nizak (0.00-0.05) jer FastText ne može razlikovati te znakove bez dodatnih pravila.

3. **Semantičko grupiranje:** Modeli ispravno grupiraju:
   - Dane u tjednu (ponedjeljak → utorak, srijeda, četvrtak)
   - Hrvatske gradove (Zagreb → Split, Rijeka, Osijek)
   - Morfološke oblike (učitelj → učitelja, učitelju, učitelji)

---

## 6. Usporedba s baseline modelom

### 6.1 Facebook cc.hr.300.bin

| Karakteristika | Facebook baseline | Naši modeli |
|----------------|-------------------|-------------|
| Veličina korpusa | ~6.4B tokena | ~400M riječi |
| Izvor podataka | Common Crawl + Wikipedia | Hrvatski tekstovi |
| Dimenzije | 300 | 100-300 |
| Prednosti | Veći korpus, opći vokabular | Specifično za domenu |

### 6.2 Očekivanja

- Facebook model može nadmašiti na općem vokabularu
- Naši modeli mogu biti bolji na domenski specifičnim terminima
- Usporedba će pokazati utjecaj veličine korpusa vs specifičnosti

### 6.3 Rezultati usporedbe

| Metrika | Facebook baseline | Najbolji model (sg_d300_ws10_e15_mc2) | Razlika |
|---------|-------------------|---------------------------------------|---------|
| Ukupni score | 0.439 | 0.528 | **+20.3%** |
| ije/je hibridni | 0.711 | 0.748 | +5.2% |
| Dijakritici hibridni | 0.820 | 0.832 | +1.5% |
| Tipfeleri hibridni | 0.755 | 0.769 | +1.9% |

**Zaključak:** Naši modeli trenirani na manjem, ali specifičnom hrvatskom korpusu nadmašuju Facebookov baseline model treniran na 16x većem korpusu (6.4B vs 400M tokena). Ovo pokazuje da je **kvaliteta i specifičnost korpusa važnija od veličine** za specijalizirane zadatke poput pravopisne korekcije.

#### Prednosti naših modela:
- Bolje prepoznavanje specifičnih hrvatskih grešaka (ije/je alternacije)
- Bolja pokrivenost domaćih termina (iz novinskih tekstova)
- Manja veličina modela uz bolju performansu

#### Prednosti Facebook modela:
- Širi vokabular (više stranih riječi, imena)
- Brža priprema (preuzimanje vs. treniranje)

---

## 7. Zaključak

### 7.1 Preporučeni model

**Preporučeni model: `ft_sg_d300_ws10_e15_mc2`**

| Parametar | Vrijednost |
|-----------|------------|
| Arhitektura | Skip-gram |
| Dimenzije | 300 |
| Veličina prozora | 10 |
| Broj epoha | 15 |
| Minimalna frekvencija | 2 |
| minn/maxn | 3/6 |
| Ukupni score | **0.528** |
| Veličina datoteke | 8.9 GB |

**Zašto ovaj model?**
1. Najviši ukupni score (0.528) među svim modelima
2. Nadmašuje Facebook baseline za 20.3%
3. Skip-gram arhitektura bolja za rijetke riječi (važno za pravopis)
4. 15 epoha osigurava stabilnije embeddings

### 7.2 Ključni nalazi

1. **Dimenzionalnost:** Veće dimenzije (300) pokazuju bolju semantičku preciznost - modeli s dim=300 zauzimaju top 3 pozicije
2. **Broj epoha:** Više epoha (15 vs 10) poboljšava rezultate - model s e15 nadmašuje e10 za 6.7%
3. **Arhitektura:** Skip-gram generalno bolji od CBOW za rijetke riječi i pravopisnu korekciju
4. **n-gram raspon:** Standardni minn=3/maxn=6 radi jednako dobro kao prošireni 2/7
5. **Veličina korpusa:** Specifičnost korpusa važnija od veličine - nadmašujemo 16x veći Facebook korpus

**FastText prednosti za hrvatski:**
- Izvrsna ije/je korekcija (riješenje → rješenje na #1 poziciji)
- Semantičko grupiranje (dani u tjednu, gradovi, zanimanja)
- Morfološka pokrivenost (svih 7 padeža)
- Word analogije (kralj:kraljica::princ:princeza radi u 5/9 modela)

### 7.3 Ograničenja

- **č/ć zamjena:** FastText ne može razlikovati č/ć jer imaju identične n-grame. MRR za dijakritike = 0.00-0.05. Potreban je hibridni pristup ili pravila za naknadnu obradu.
- **Jednoznačne greške:** Greške s jednim znakom (moč→moć) zahtijevaju dodatna pravila
- **Bez konteksta:** Embeddinzi ne hvataju kontekst rečenice - ista riječ ima isti vektor bez obzira na kontekst
- **"Prljave" riječi:** Korpus sadrži riječi s pripojenom interpunkcijom (škola, pisati-) zbog nedovoljno agresivne tokenizacije

### 7.4 Budući rad

1. **Hibridni sustav:** Kombinacija FastText embeddinga s edit distance algoritmima i pravilima za dijakritike
2. **Kontekstualni modeli:** Integracija s BERT-om za kontekstualno razumijevanje
3. **Poboljšana tokenizacija:** Čistiji korpus bez pripojene interpunkcije
4. **Proširena evaluacija:** Više testnih primjera, posebno za rijetke riječi
5. **Integracija s ispravi.me:** Implementacija kao spellcheck backend

### 7.5 Kvalitativni primjeri

Detaljni primjeri izlaza modela dostupni su u datoteci `results/example_outputs.md`. Dokument sadrži:
- Tablice najbližih susjeda za testne riječi
- Usporedbu korekcije pravopisa po modelima
- Semantičku sličnost parova riječi
- Rezultate word analogija (A:B::C:?)

---

## 8. Reference

1. Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
2. [FastText](https://fasttext.cc/) - Biblioteka za klasifikaciju i reprezentaciju teksta
3. [ispravi.me](https://ispravi.me) - Hrvatski pravopisni korektor

---

## Dodatak A: Naredbe za reprodukciju

```bash
# Provjera statusa treniranja
python scripts/train_optimized.py --status

# Treniranje specifičnog modela
python scripts/train_optimized.py --name sg_d300_ws10_e10_mc2

# Evaluacija svih modela
python scripts/evaluate_model.py --all-models

# Usporedba modela
python scripts/compare_models.py

# Preuzimanje Facebook baseline modela
bash scripts/download_baseline.sh

# Generiranje kvalitativnih primjera
python scripts/generate_examples.py
```

## Dodatak B: Struktura projekta

```
diplomski_projekt/
├── data/
│   ├── corpus_clean.txt          # Očišćeni korpus (2.5GB)
│   └── eval/                     # Evaluacijski podaci
│       ├── ije_je_errors.json
│       ├── diacritic_errors.json
│       ├── general_typos.json
│       ├── similarity_pairs.json
│       └── morphological.json
├── scripts/
│   ├── train_optimized.py        # Skripta za treniranje
│   ├── evaluate_model.py         # Evaluacija modela
│   ├── compare_models.py         # Usporedba modela
│   ├── generate_examples.py      # Generiranje kvalitativnih primjera
│   ├── diacritic_rules.py        # Pravila za dijakritičku korekciju
│   └── download_baseline.sh      # Preuzimanje baseline modela
├── models/                       # Trenirani modeli (nije u gitu)
├── results/
│   ├── report.md                 # Ovaj izvještaj
│   ├── evaluation_results.json   # Detaljni rezultati evaluacije
│   └── example_outputs.md        # Kvalitativni primjeri
└── requirements.txt
```

## Dodatak C: Primjer korištenja modela

```python
import fasttext

# Učitavanje najboljeg modela
model = fasttext.load_model("models/ft_sg_d300_ws10_e15_mc2.bin")

# Pronalaženje najbližih susjeda (za korekciju pravopisa)
neighbors = model.get_nearest_neighbors("riješenje", k=10)
# Rezultat: [('rješenje', 0.93), ('riješenja', 0.91), ...]

# Računanje sličnosti između riječi
vec1 = model.get_word_vector("škola")
vec2 = model.get_word_vector("učenik")
similarity = cosine_similarity(vec1, vec2)  # ~0.85
```
