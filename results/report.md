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

| Model | Tip | Vrijeme treniranja | Veličina vokabulara | Datum |
|-------|-----|-------------------|---------------------|-------|
| sg_d200_ws5_e3_mc2 | Tier 0 | 81 min | 2,918,267 | 2026-01-28 |
| cbow_d200_ws5_e3_mc2 | Tier 0 | 52 min | 2,918,267 | 2026-01-28 |
| sg_d300_ws10_e10_mc2 | Tier 1 | 596 min (~10h) | 2,918,267 | 2026-01-29 |

### 4.2 Modeli u tijeku

| Model | Tip | Status |
|-------|-----|--------|
| sg_d100_ws5_e3_mc2 | Tier 0 | U tijeku |
| sg_d200_ws10_e3_mc2 | Tier 0 | U tijeku |
| sg_d300_ws10_e15_mc2 | Tier 1 | Planirano |
| sg_d300_ws15_e10_mc2 | Tier 1 | Planirano |
| cbow_d300_ws10_e10_mc2 | Tier 1 | Planirano |

### 4.3 Statistike treniranja

- **Ukupan broj riječi u korpusu:** ~398 milijuna
- **Veličina vokabulara:** 2,918,267 jedinstvenih riječi
- **Prosječno vrijeme Tier 0:** ~65 min
- **Prosječno vrijeme Tier 1:** ~600 min

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

*[Rezultati će biti dodani nakon završetka evaluacije]*

| Model | ije/je MRR | Diak. MRR | Općeniti MRR | Sem. sličnost | Ukupno |
|-------|------------|-----------|--------------|---------------|--------|
| sg_d200_ws5_e3_mc2 | - | - | - | - | - |
| cbow_d200_ws5_e3_mc2 | - | - | - | - | - |
| sg_d300_ws10_e10_mc2 | - | - | - | - | - |
| facebook_cc_hr_300 | - | - | - | - | - |

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

*[Rezultati će biti dodani nakon evaluacije]*

---

## 7. Zaključak

### 7.1 Preporučeni model

*[Bit će određeno nakon završetka svih evaluacija]*

### 7.2 Ključni nalazi

1. **Dimenzionalnost:** Veće dimenzije (300) pokazuju bolju semantičku preciznost
2. **Veličina prozora:** Veći prozori (10-15) bolje hvataju kontekst u morfološki bogatom jeziku
3. **Minimalna frekvencija:** minCount=2 omogućuje bolje rukovanje rijetkim riječima
4. **Arhitektura:** Skip-gram generalno bolji za rijetke riječi

### 7.3 Ograničenja

- **č/ć zamjena:** FastText vidi ove znakove kao slične zbog dijeljenih n-grama
- **Jednoznačne greške:** Potrebna naknadna pravila za greške s jednim znakom
- **Bez konteksta:** Embeddinzi ne hvataju kontekst rečenice

### 7.4 Budući rad

1. Kombinacija s kontekstualnim modelima (BERT)
2. Pravila za naknadnu obradu č/ć grešaka
3. Proširenje evaluacijskog skupa
4. Integracija s ispravi.me

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
```

## Dodatak B: Struktura projekta

```
diplomski_projekt/
├── data/
│   ├── corpus_clean.txt          # Očišćeni korpus (2.5GB)
│   └── eval/                     # Evaluacijski podaci
├── scripts/
│   ├── train_optimized.py        # Skripta za treniranje
│   ├── evaluate_model.py         # Evaluacija modela
│   ├── compare_models.py         # Usporedba modela
│   └── download_baseline.sh      # Preuzimanje baseline modela
├── models/                       # Trenirani modeli (nije u gitu)
├── results/                      # Rezultati treniranja i evaluacije
└── requirements.txt
```
