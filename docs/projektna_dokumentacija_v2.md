# Optimizacija FastText modela za semantičku provjeru pravopisa hrvatskog jezika

**Projektna dokumentacija**

Fakultet elektrotehnike i računarstva, Sveučilište u Zagrebu

Verzija 2.0 | Siječanj 2026.

---

## Sažetak

Ovaj dokument opisuje razvoj i evaluaciju optimalnih FastText modela ugniježđenih vektora riječi za unapređenje usluge ispravi.me, hrvatskog sustava za automatsku provjeru pravopisa. Kroz sustavno istraživanje hiperparametara trenirali smo devet modela različitih konfiguracija na korpusu od 2.5 GB hrvatskog teksta koji obuhvaća približno 400 milijuna riječi. Evaluacija je provedena korištenjem hibridnog pristupa koji kombinira udaljenost uređivanja s kosinusnom sličnošću vektorskih reprezentacija, čime se postiže pravednije vrednovanje sposobnosti modela za zadatak pravopisne korekcije.

Najbolji model, ft_sg_d300_ws10_e15_mc2, postiže ukupni rezultat od 0.528 na definiranim metrikama, čime nadmašuje Facebookov unaprijed trenirani baseline model za 20.3%. Ovaj rezultat osobito je značajan s obzirom na to da je naš korpus za treniranje šesnaest puta manji od korpusa korištenog za Facebook baseline, što ukazuje na važnost specifičnosti i kvalitete podataka za specijalizirane zadatke obrade prirodnog jezika.

Projektni tim činili su Nikša Brala kao voditelj te Teo Matošević i Vitomir Brebrić kao članovi. Projekt je realiziran u razdoblju od studenog 2025. do siječnja 2026. godine na Fakultetu elektrotehnike i računarstva Sveučilišta u Zagrebu.

---

## 1. Uvod i motivacija

### 1.1 Kontekst problema

Usluga ispravi.me predstavlja jedan od najvažnijih i najkorištenijih alata za automatsku provjeru pravopisa hrvatskog jezika. Sustav svakodnevno obrađuje tisuće upita korisnika koji se oslanjaju na njegovu preciznost pri pisanju formalnih dokumenata, akademskih radova i svakodnevne komunikacije. Trenutačna arhitektura sustava temelji se na n-gramskom pristupu za kontekstnu analizu teksta, kombiniranom s opsežnim rječnikom hrvatskog jezika i skupom pravila za prepoznavanje čestih pogrešaka.

Ovaj pristup pokazao se učinkovitim za prepoznavanje osnovnih pravopisnih nepravilnosti i tipičnih tipografskih pogrešaka. Međutim, kako su zahtjevi korisnika postajali sofisticiraniji, postala su vidljiva određena ograničenja sustava. N-gramski modeli izvrsno rade kada je pogreška lokalizirana i kada ispravna riječ postoji u rječniku, ali pokazuju slabosti u situacijama koje zahtijevaju dublje razumijevanje semantičkog konteksta.

Razmotrimo primjer rečenice "Moram kupiti mlieko za kavu." N-gramski pristup može prepoznati da "mlieko" nije standardna riječ hrvatskog jezika, ali bez semantičkog razumijevanja teško može pouzdano predložiti ispravak "mlijeko" umjesto drugih fonetski sličnih alternativa. Problem postaje još izraženiji kod morfološki bogatih jezika poput hrvatskog, gdje ista riječ može poprimiti desetke različitih oblika ovisno o padežu, broju, rodu i glagolskom vremenu.

### 1.2 Cilj projekta

Cilj ovog projekta bio je istražiti mogu li moderni modeli ugniježđenih vektora riječi, specifično FastText algoritam, unaprijediti sposobnost sustava ispravi.me za predlaganje semantički prikladnih ispravaka. FastText je odabran zbog svojih jedinstvenih karakteristika koje ga čine posebno prikladnim za morfološki bogate jezike poput hrvatskog.

Za razliku od ranijih pristupa poput Word2Vec koji tretiraju riječi kao atomske jedinice, FastText rastavlja svaku riječ na sastavne podriječi (subwords) definirane kao n-grami znakova. Ovaj pristup omogućuje modelu generiranje smislenih vektorskih reprezentacija čak i za riječi koje nikada nije vidio tijekom treniranja, što je iznimno korisno za jezike s bogatom morfologijom gdje je nemoguće unaprijed predvidjeti sve moguće oblike riječi.

Konkretni ciljevi projekta uključivali su treniranje višestrukih FastText modela s različitim kombinacijama hiperparametara, razvoj evaluacijskog okvira koji pravedno vrednuje sposobnosti modela za pravopisnu korekciju, usporedbu vlastitih modela s postojećim baseline modelom, te identifikaciju optimalne konfiguracije za integraciju u produkcijski sustav ispravi.me.

---

## 2. Izazovi hrvatskog jezika

### 2.1 Morfološka složenost

Hrvatski jezik pripada skupini južnoslavenskih jezika i karakterizira ga izuzetna morfološka složenost koja predstavlja značajan izazov za sustave automatske obrade teksta. Sustav od sedam gramatičkih padeža (nominativ, genitiv, dativ, akuzativ, vokativ, lokativ i instrumental) znači da svaka imenica, pridjev i zamjenica može poprimiti višestruke oblike ovisno o sintaktičkoj ulozi u rečenici.

Razmotrimo riječ "učitelj" i njezine padežne oblike: učitelj (nominativ), učitelja (genitiv), učitelju (dativ), učitelja (akuzativ), učitelju (lokativ), učiteljem (instrumental), te množinske oblike učitelji, učitelja, učiteljima i tako dalje. Tradicionalni pristupi temeljeni na rječnicima morali bi eksplicitno pohraniti sve ove oblike, dok FastText model može naučiti morfološke obrasce i generalizirati na oblike koje nije eksplicitno vidio.

Slična složenost prisutna je kod glagola. Glagol "pisati" ima oblike: pišem, pišeš, piše, pišemo, pišete, pišu (prezent), pisao, pisala, pisalo, pisali (perfekt), pišući (glagolski prilog sadašnji), napisati, napišem (svršeni vid), te mnoštvo drugih oblika. Naš model pokazuje razumijevanje ovih odnosa - kada tražimo najbliže susjede za "pisati", među rezultatima nalazimo "napisati" (svršeni vid istog glagola), "čitati" (semantički povezan glagol), "dopisati" i "izpisati" (prefigirane varijante).

Naši eksperimenti potvrdili su ovu sposobnost FastText modela. Kada tražimo najbliže susjede riječi "učitelj", najbolji model vraća morfološke varijante poput "učitelja", "učitelju" i "učiteljica", ali također semantički povezane riječi poput "nastavnik" i "profesor". Za riječ "lijep", model vraća "prelijep", "prekrasan", "predivan" i "lijepši" - kombinaciju morfoloških varijanti (komparativ "lijepši") i semantičkih sinonima ("prekrasan", "predivan"). Ova kombinacija morfološke i semantičke svijesti čini FastText iznimno korisnim za zadatke pravopisne korekcije.

### 2.2 Alternacije ije/je

Jedna od najčešćih pravopisnih pogrešaka u hrvatskom jeziku vezana je uz alternacije ije/je koje proizlaze iz historijskog razvoja jezika. Praslavenski glas jat (ě) razvio se različito u različitim slavenskim jezicima, a u hrvatskom standardnom jeziku realizira se kao "ije" u dugim slogovima i "je" u kratkim slogovima. Ovo pravilo, međutim, ima brojne iznimke i specifičnosti koje ga čine izazovnim čak i za izvorne govornike.

Primjeri čestih pogrešaka uključuju "riješenje" umjesto ispravnog "rješenje", "mlieko" umjesto "mlijeko", "tielo" umjesto "tijelo", "biel" umjesto "bijel", "grieh" umjesto "grijeh", te "vriemena" umjesto "vremena". Naši rezultati pokazuju da FastText modeli izuzetno dobro prepoznaju ove obrasce, iako s varijabilnim uspjehom ovisno o specifičnom primjeru.

Za pogrešno napisanu riječ "riješenje", svih osam naših modela vraća ispravnu riječ "rješenje" na prvom mjestu liste najbližih susjeda s kosinusnom sličnošću od 0.861. Ovo je izvrstan rezultat koji demonstrira sposobnost modela za prepoznavanje ije/je alternacija. Međutim, za druge primjere rezultati su manje konzistentni. Za "mlieko" → "mlijeko", samo jedan model (ft_sg_d300_ws10_e10_mc2_minn2_maxn7) vraća ispravnu riječ, i to tek na osmom mjestu. Ostali modeli vraćaju "mlieka" (genitiv pogrešne forme) kao najbližeg susjeda. Za "biel" → "bijel", nijedan model ne vraća ispravnu riječ u prvih deset susjeda, što ukazuje na to da kratke riječi s ije/je alternacijom predstavljaju veći izazov.

Hibridni rezultat za kategoriju ije/je pogrešaka iznosi 0.748 za najbolji model. Ovaj rezultat uključuje i primjere gdje model nije pronašao ispravnu riječ među najbližim susjedima, ali je embedding sličnost između pogrešne i ispravne riječi i dalje visoka. Na primjer, za par "mlieko" i "mlijeko", kosinusna sličnost iznosi 0.765 što ukazuje na to da model prepoznaje njihovu povezanost čak i kada ispravna riječ nije među najbližim susjedima. Ovo predstavlja poboljšanje od 5.2% u odnosu na Facebook baseline.

### 2.3 Dijakritički znakovi

Specifičnost hrvatskog pravopisa očituje se i u upotrebi dijakritičkih znakova. Hrvatski alfabet uključuje znakove č, ć, š, ž i đ koji ne postoje u osnovnom latiničnom pismu, što stvara probleme pri pisanju na tipkovnicama koje ih ne podržavaju. Korisnici često izostavljaju dijakritičke znakove (pišući "covjek" umjesto "čovjek") ili ih zamjenjuju neispravnim varijantama.

Poseban izazov predstavljaju parovi č/ć i š/ž koji su fonetski slični, a razlikuju se samo u stupnju palatalizacije. FastText model ima inherentna ograničenja u razlikovanju ovih parova jer se razlikuju samo u jednom znaku, a dijele većinu n-grama. Na primjer, riječi "moč" i "moć" imaju gotovo identične n-gram reprezentacije, što otežava modelu razlikovanje ispravne od neispravne varijante.

Naša evaluacija pokazala je ovo ograničenje kroz nekoliko konkretnih primjera. Za "moč" → "moć", nijedan model nije uspio vratiti ispravnu riječ u prvih deset susjeda. Najbliži susjedi uključuju "močna" (0.535), "pomoč" (0.514), "močnika" (0.511) i "nadmoč" (0.493) - sve riječi koje sadrže niz "moč", ali ne i ciljnu riječ "moć". Zanimljivo je da se "moć" ne pojavljuje jer model tretira "moč" i "moć" kao gotovo identične zbog preklapajućih n-grama.

Slična situacija je s primjerom "noč" → "noć" i "kuča" → "kuća". Za "kuča", model vraća "kuća" s visokom sličnošću od 0.925, ali samo zato što generira varijante kroz diacritic_rules modul, a ne kroz nearest neighbor pretragu. Bez dodatne obrade, sam FastText ne bi predložio ispravak.

Posebno problematični su primjeri gdje nedostaje dijakritički znak u potpunosti, poput "skola" → "škola", "covjek" → "čovjek", ili "zivot" → "život". U ovim slučajevima, razlika nije samo u č/ć nego u potpunom izostanku dijakritika. Model vraća "skola" kao samostalnu riječ (koja postoji u korpusu kao pogrešno napisana varijanta) bez prepoznavanja da bi ispravna forma trebala biti "škola".

Ovo potvrđuje poznato ograničenje FastText algoritma i ukazuje na potrebu za dopunskim pristupima, poput pravila za post-procesiranje, u produkcijskom sustavu. Implementirali smo modul diacritic_rules.py koji generira sve moguće dijakritičke varijante riječi i koristi model za njihovo rangiranje, čime se ovo ograničenje djelomično prevladava.

---

## 3. FastText algoritam

### 3.1 Teoretska osnova

FastText je algoritam za učenje vektorskih reprezentacija riječi koji je 2016. godine razvio istraživački tim Facebooka (danas Meta) predvođen Piotr Bojanowskim. Algoritam se temelji na ranijim Word2Vec arhitekturama (Skip-gram i CBOW), ali uvodi ključnu inovaciju: reprezentaciju riječi putem sastavnih n-grama znakova.

U standardnom Word2Vec pristupu, svaka riječ u vokabularu ima jedinstven vektor koji se uči tijekom treniranja. Ovaj pristup ima dva značajna ograničenja. Prvo, riječi koje se nisu pojavile tijekom treniranja (out-of-vocabulary, OOV) ne mogu dobiti vektorsku reprezentaciju. Drugo, morfološki povezane riječi (poput "učitelj" i "učiteljica") tretiraju se kao potpuno nezavisne, bez mogućnosti dijeljenja naučenih obrazaca.

FastText rješava ove probleme rastavljanjem svake riječi na skup n-grama znakova. Na primjer, riječ "učitelj" s parametrima minn=3 i maxn=6 generira n-grame: <uč, uči, učit, učite, čit, čite, čitel, ite, itel, itelj, tel, telj, elj, elj>. Specijalni znakovi < i > označavaju početak i kraj riječi. Konačna reprezentacija riječi dobiva se zbrajanjem vektora svih njezinih n-grama i vektora same riječi ako postoji u vokabularu.

### 3.2 Arhitekture Skip-gram i CBOW

FastText podržava dvije osnovne arhitekture za treniranje: Skip-gram i Continuous Bag of Words (CBOW). U Skip-gram arhitekturi, model prima ciljnu riječ kao ulaz i pokušava predvidjeti kontekstne riječi koje je okružuju unutar definiranog prozora. CBOW arhitektura radi obrnuto: prima kontekstne riječi i pokušava predvidjeti ciljnu riječ.

Empirijska istraživanja pokazala su da Skip-gram arhitektura tipično postiže bolje rezultate za rjeđe riječi i morfološke varijante, dok je CBOW brži za treniranje i može biti bolji za česte riječi. Za zadatak pravopisne korekcije, gdje je važno prepoznati i rijetke oblike riječi, očekivali smo da će Skip-gram arhitektura pokazati bolje rezultate.

Naši eksperimenti potvrdili su ovu hipotezu. Svih pet najbolje rangiranih modela koristi Skip-gram arhitekturu. Najbolji CBOW model (ft_cbow_d300_ws10_e10_mc2) rangiran je tek na sedmom mjestu s ukupnim rezultatom 0.445, što je 15.7% lošije od najboljeg Skip-gram modela. Ova razlika osobito je izražena kod semantičke sličnosti gdje Skip-gram modeli konzistentno postižu više korelacije s očekivanim vrijednostima.

---

## 4. Metodologija

### 4.1 Priprema korpusa

Kvaliteta korpusa za treniranje presudna je za performanse rezultirajućeg modela. Za potrebe ovog projekta pripremljen je opsežan korpus hrvatskog jezika ukupne veličine 2.5 GB očišćenog teksta, što odgovara približno 400 milijuna riječi. Korpus je prikupljen iz različitih izvora uključujući novinske članke, web sadržaje i javno dostupne tekstove na hrvatskom jeziku.

Proces čišćenja korpusa uključivao je više koraka predobrade. Prvo, uklonjeni su svi HTML elementi i specijalni znakovi koji nisu dio standardnog hrvatskog pravopisa. Zatim su normalizirani dijakritički znakovi, budući da različiti izvori koriste različite kodne stranice (UTF-8, Windows-1250, ISO-8859-2). Svi tekstovi pretvoreni su u mala slova kako bi se smanjila raznolikost vokabulara i omogućilo modelu učenje jedinstvih reprezentacija neovisno o kapitalizaciji.

Konačni vokabular korpusa obuhvaća 2,918,267 jedinstvenih riječi. Ovaj broj uključuje različite morfološke oblike istih leksema, toponime, vlastita imena i određeni udio riječi iz stranih jezika koje se pojavljuju u hrvatskim tekstovima. Postavili smo minimalnu frekvenciju pojavljivanja (minCount) na 2, što znači da su u treniranje uključene i riječi koje se pojavljuju samo dvaput u cijelom korpusu. Ova odluka motivirana je potrebom za prepoznavanjem rijetkih ali ispravnih oblika riječi.

### 4.2 Konfiguracije hiperparametara

Na temelju pregleda literature i prethodnih istraživanja provedenih u okviru Seminara 2, definirali smo skup konfiguracija hiperparametara za sustavno istraživanje. Konfiguracije su organizirane u tri razine složenosti.

Prva razina (Tier 0) uključuje brze modele namijenjene inicijalnoj validaciji pristupa. Ovi modeli koriste manje dimenzionalnosti vektora (100 ili 200 dimenzija) i manji broj epoha treniranja (3 epohe), što omogućuje treniranje u roku od nekoliko sati. Tier 0 modeli poslužili su za identifikaciju obećavajućih smjerova prije ulaganja računalnih resursa u zahtjevnije konfiguracije.

Druga razina (Tier 1) predstavlja standardne konfiguracije temeljene na najboljim praksama iz literature. Ovi modeli koriste 300 dimenzija, veličinu kontekstnog prozora od 10 riječi i 10 do 15 epoha treniranja. Vrijeme treniranja za svaki Tier 1 model iznosilo je približno 11 sati na dostupnom hardveru.

Treća razina (Tier 2) uključuje eksperimentalne varijacije parametara poput stope učenja i broja negativnih uzoraka. Ove konfiguracije namijenjene su istraživanju rubnih slučajeva i potencijalno neočekivanih poboljšanja.

Ukupno je trenirano osam vlastitih modela, uz korištenje Facebookovog unaprijed treniranog modela (cc.hr.300.bin) kao baseline za usporedbu. Facebook model treniran je na znatno većem korpusu od približno 6.4 milijarde tokena iz Common Crawl i Wikipedia izvora.

### 4.3 Evaluacijski okvir

Tradicionalne metrike za evaluaciju modela ugniježđenih vektora, poput Mean Reciprocal Rank (MRR) temeljenog na poziciji ispravne riječi u listi najbližih susjeda, pokazale su se neprikladnima za naš specifični zadatak. Problem proizlazi iz činjenice da FastText optimizira semantičku sličnost, a ne pravopisnu korekciju. Semantički slične riječi (poput "škola" i "učenik") bit će blizu u vektorskom prostoru, ali to ne znači da će pogrešno napisana riječ imati ispravnu varijantu među najbližim susjedima.

Razvili smo hibridni pristup evaluaciji koji kombinira tradicionalnu metriku udaljenosti uređivanja (edit distance) s kosinusnom sličnošću vektorskih reprezentacija. Formula za hibridni rezultat glasi:

*hibridni_rezultat = 0.4 × edit_sličnost + 0.6 × embedding_sličnost*

gdje je edit_sličnost definirana kao (1 - edit_distance / max_duljina), a embedding_sličnost kao kosinusna sličnost između vektora pogrešne i ispravne riječi. Težinski faktori (40% za edit distance, 60% za embedding) određeni su empirijski na temelju validacijskog skupa.

Evaluacijski skup podataka organiziran je u tri kategorije pogrešaka: ije/je alternacije (10 parova), dijakritičke pogreške (10 parova) i opći tipfeleri (10 parova). Dodatno, evaluirali smo semantičku sličnost na 10 parova riječi s definiranim očekivanim sličnostima te morfološku pokrivenost za 5 baznih riječi s očekivanim morfološkim oblicima.

---

## 5. Rezultati

### 5.1 Rangiranje modela

Evaluacija svih devet modela pokazala je jasnu hijerarhiju performansi. Na vrhu se nalazi model ft_sg_d300_ws10_e15_mc2 s ukupnim rezultatom 0.528. Ovaj model koristi Skip-gram arhitekturu, vektore dimenzionalnosti 300, kontekstni prozor veličine 10 riječi i treniran je kroz 15 epoha. Drugi po rangu je ft_sg_d300_ws10_e10_mc2 s rezultatom 0.495, što je identična konfiguracija ali s 10 umjesto 15 epoha. Treće mjesto zauzima ft_sg_d300_ws10_e10_mc2_minn2_maxn7 s rezultatom 0.493, koji eksperimentira s širim rasponom n-grama (2-7 umjesto standardnih 3-6).

Razlika između prvog i drugog modela iznosi 6.7%, što sugerira da dodatnih 5 epoha treniranja donosi značajno poboljšanje. Ovaj nalaz ima praktične implikacije: iako dulje treniranje zahtijeva više računalnih resursa, poboljšanje performansi opravdava ulaganje.

Facebook baseline model (cc.hr.300.bin) rangiran je na osmom mjestu s rezultatom 0.439. Činjenica da naši modeli trenirani na šesnaest puta manjem korpusu postižu bolje rezultate ukazuje na važnost specifičnosti korpusa. Facebook model treniran je na općenitom web sadržaju koji uključuje tekstove na brojnim jezicima i domenama, dok je naš korpus fokusiran isključivo na hrvatski jezik i tekstove relevantne za zadatak pravopisne korekcije.

Najslabije rezultate postigao je ft_cbow_d200_ws5_e3_mc2 s rezultatom 0.438, tek neznatno ispod Facebook baseline. Ovaj model kombinira CBOW arhitekturu s manjom dimenzionalnošću i brojem epoha, što potvrđuje da sve tri karakteristike negativno utječu na performanse za naš zadatak.

### 5.2 Analiza po kategorijama pogrešaka

Detaljnija analiza rezultata po kategorijama pogrešaka otkriva zanimljive obrasce. U kategoriji ije/je pogrešaka, najbolji model postiže hibridni rezultat od 0.748. Kvalitativna analiza pokazuje da modeli izuzetno dobro prepoznaju ove obrasce. Za testni primjer "riješenje" → "rješenje", sedam od devet modela vraća ispravnu riječ na prvom mjestu liste najbližih susjeda. Čak i CBOW model, koji generalno zaostaje za Skip-gram varijantama, vraća ispravnu riječ na četvrtom mjestu.

Kategorija dijakritičkih pogrešaka pokazuje hibridni rezultat od 0.832, što je najviša vrijednost među svim kategorijama. Međutim, ovaj rezultat treba interpretirati s oprezom. Visoki hibridni rezultat proizlazi iz komponente udaljenosti uređivanja (zamjena jednog znaka rezultira visokom edit sličnošću), dok embedding komponenta ne doprinosi značajno. Kako smo ranije objasnili, FastText ima inherentna ograničenja u razlikovanju parova poput č/ć jer dijele gotovo identične n-gram reprezentacije.

Kategorija općih tipfelera pokazuje hibridni rezultat od 0.769 za najbolji model. Primjeri u ovoj kategoriji uključuju udvostručenja slova ("pisatti" → "pisati"), izostavljanja ("doabr" → "dobar") i zamjene slova ("teleofn" → "telefon"). FastText modeli pokazuju dobru robusnost na ove tipove pogrešaka zahvaljujući n-gram reprezentaciji koja zadržava većinu informacije čak i kada je jedan ili dva znaka pogrešno.

### 5.3 Kvalitativni primjeri

Za dublje razumijevanje ponašanja modela, analizirali smo izlaze za odabrane testne riječi kroz nekoliko kategorija: semantičko grupiranje, morfološku svijest, pravopisnu korekciju i analogije.

#### Semantičko grupiranje: Dani u tjednu

Tablica najbližih susjeda za ispravno napisanu riječ "ponedjeljak" ilustrira semantičku koherentnost modela. Svih pet najbližih susjeda su dani u tjednu: utorak (0.940), srijedu (0.928), petak (0.923), četvrtak (0.922) i nedjelju (0.839). Kosinusne sličnosti su iznimno visoke, sve iznad 0.8, što pokazuje da je model uspješno naučio semantički koncept "dana u tjednu" i grupira sve pripadnike ove kategorije u bliskom području vektorskog prostora. Čak i subota, koja se pojavljuje na devetom mjestu s sličnošću 0.806, ostaje unutar iste semantičke grupe.

#### Semantičko grupiranje: Hrvatski gradovi

Slično ponašanje vidimo za riječ "Zagreb". Najbliži susjedi uključuju druge hrvatske gradove: Bjelovar (0.656), Samobor (0.632), Koprivnica (0.606), Našice (0.601), Karlovac (0.600) i Dubrovnik (0.597). Zanimljivo je primijetiti da su sličnosti niže nego kod dana u tjednu, što sugerira da je koncept "hrvatskog grada" manje čvrsto definiran u vektorskom prostoru nego koncept "dana u tjednu". Model je ipak naučio da su ovo sve entiteti iste kategorije i smjestio ih blizu u vektorskom prostoru. Ovo ima praktične implikacije za pravopisnu korekciju: ako korisnik napiše ime grada s pogreškom, sustav može predložiti ispravke iz skupa sličnih entiteta.

#### Semantičko grupiranje: Tehnološki pojmovi

Za riječ "računalo", model vraća: Računalo (0.865), računalu (0.784), računala (0.778), miniračunalo (0.762), računalom (0.759), superačunalo (0.740) i prijenosno (0.735). Ovdje vidimo kombinaciju morfoloških varijanti (računalu, računala, računalom) i semantički povezanih složenica (miniračunalo, superačunalo) te kolokacija (prijenosno - kao u "prijenosno računalo"). Model je naučio ne samo da prepoznaje morfološke oblike nego i konceptualno povezane termine iz domene računarstva.

#### Morfološka svijest: Učitelj

Za riječ "učitelj", najbliži susjedi kombiniraju morfološke varijante sa semantički povezanim konceptima: nastavnik (0.751) kao semantički sinonim, učiteljev (0.734) kao posvojni pridjev, naučitelj (0.730) kao arhaičnu varijantu, učiteljâ (0.728) kao genitiv množine, učenik (0.724) kao semantički povezan koncept (relacija učitelj-učenik), učitelja (0.705) kao genitiv jednine, te učiteljski (0.698) kao pridjevsku izvedenicu. Ova kombinacija morfološke i semantičke svijesti posebno je korisna za hrvatski jezik gdje ista riječ može poprimiti mnogobrojne oblike.

#### Semantička sličnost: Parovi riječi

Analizirali smo kosinusnu sličnost između parova semantički povezanih riječi kako bismo procijenili kvalitetu naučenih reprezentacija. Za par "pisati" i "čitati" (dvije povezane radnje vezane uz tekst), najbolji model daje sličnost 0.793, što je značajno više od Facebook baseline koji daje 0.657. Za par "auto" i "vozilo" (sinonimi), naš model daje 0.679 naspram 0.408 za baseline. Za par "ponedjeljak" i "utorak" (susjedni dani), sličnost je iznimno visoka: 0.940 za naš model i 0.821 za baseline.

Posebno je zanimljiv par "škola" i "banana" koji služi kao negativni kontrolni primjer - dvije semantički nepovezane riječi. Naš model daje sličnost od samo 0.141, što potvrđuje da model ispravno razlikuje povezane od nepovezanih koncepata. Facebook baseline za isti par daje 0.139, što je usporedivo.

#### Analogije: Mješoviti rezultati

Analiza testnih analogija pokazuje mješovite rezultate koji otkrivaju i snage i slabosti modela. Za analogiju "kralj:kraljica::princ:?" pet od devet modela (uključujući naš najbolji) ispravno vraća "princeza" kao prvi odgovor. Ovo pokazuje da model razumije rod-specifične transformacije za ljudska zanimanja i titule.

Za analogiju "učitelj:učiteljica::student:?" šest modela vraća ispravnu riječ "studentica", demonstrirajući razumijevanje morfoloških obrazaca za tvorbu ženskog roda. Model ft_sg_d300_ws10_e15_mc2 vraća "studentica" na prvom mjestu.

Međutim, za geografsku analogiju "Zagreb:Hrvatska::Rim:?" nijedan model nije uspio vratiti očekivani odgovor "Italija". Naš model vraća "Rima" (genitiv od Rim), dok Facebook baseline vraća nerazumljiv rezultat "16,3-5". Ovo sugerira da modeli bolje razumiju morfološke i semantičke odnose nego geografske/činjenične asocijacije.

Za analogiju "ponedjeljak:utorak::srijeda:?" očekivani odgovor je "četvrtak" (sljedeći dan). Zanimljivo, samo Facebook baseline vraća ispravni odgovor, dok naši modeli vraćaju složenice poput "srijedasubota". Ovo ukazuje na to da modeli ne razumiju linearni redoslijed dana nego ih tretiraju kao neuredni skup.

---

## 6. Usporedba s baseline modelom

### 6.1 Kvantitativna usporedba

Usporedba s Facebook baseline modelom (cc.hr.300.bin) daje važan kontekst za procjenu kvalitete naših modela. Facebook model treniran je na korpusu od približno 6.4 milijarde tokena, što je šesnaest puta više od našeg korpusa. Unatoč ovoj razlici u veličini podataka za treniranje, naš najbolji model nadmašuje baseline za 20.3% u ukupnom rezultatu (0.528 naspram 0.439).

Razlike su vidljive u svim kategorijama evaluacije. U kategoriji ije/je pogrešaka, naš model postiže 0.748 naspram 0.711 za baseline (poboljšanje od 5.2%). U kategoriji dijakritičkih pogrešaka, rezultati su 0.832 naspram 0.820 (poboljšanje od 1.5%). U kategoriji općih tipfelera, rezultati su 0.769 naspram 0.755 (poboljšanje od 1.9%).

Najveća razlika vidljiva je u semantičkoj sličnosti i morfološkoj pokrivenosti, što sugerira da je specifičnost korpusa posebno važna za ove aspekte. Naš korpus, fokusiran na kvalitetne hrvatske tekstove, omogućio je modelu bolje učenje semantičkih odnosa karakterističnih za hrvatski jezik.

### 6.2 Kvalitativna usporedba

Kvalitativna analiza otkriva zanimljive razlike u ponašanju modela koje objašnjavaju kvantitativne razlike u performansama.

#### Usporedba: Riječ "škola"

Za testnu riječ "škola", naš najbolji model vraća susjede: škole (0.843), učenika (0.803), škol (0.784), učenike (0.782), učenici (0.760), gimnazija (0.757), nastava (0.751). Facebook baseline vraća potpuno drugačiji skup: eko-škola (0.707), auto-škola (0.703), Eko-škola (0.695), Ekoškola (0.690), Škola (0.677), školah (0.673), školaSrednja (0.670).

Razlika je izrazita. Naš model pronalazi semantički relevantne susjede koji su korisni za razumijevanje konteksta - učenik, gimnazija i nastava su koncepti usko vezani uz školu. Facebook model vraća uglavnom složenice koje sadrže riječ "škola" kao komponentu (eko-škola, auto-škola) i morfološke varijante. Ovo sugerira da je naš korpus, fokusiran na kvalitetne hrvatske tekstove, omogućio modelu bolje učenje semantičkih odnosa.

#### Usporedba: Riječ "učitelj"

Za "učitelj", naš model vraća: nastavnik (0.751), učiteljev (0.734), naučitelj (0.730), učiteljâ (0.728), Učitelj (0.726), učenik (0.724). Facebook model vraća: učiteljâ (0.696), Učitelj (0.671), nadučitelj (0.624), podučitelj (0.621), učitelь (0.619), učitelj-student (0.609).

Naš model prepoznaje "nastavnik" kao sinonim na prvom mjestu, što je semantički najrelevantniji rezultat. Facebook model ne pronalazi sinonime nego samo morfološke i složene varijante. Također, naš model prepoznaje relaciju učitelj-učenik (učenik na šestom mjestu), dok Facebook model to ne uspijeva.

#### Usporedba: Semantička sličnost

Za semantičku sličnost između riječi "pisati" i "čitati", naš model daje vrijednost 0.793, dok Facebook baseline daje 0.657. Razlika od 20.7% je značajna i pokazuje da naš model bolje razumije da su ovo dvije povezane radnje. Za par "škola" i "učenik", naš model daje 0.685 naspram 0.374 za baseline - razlika od čak 83%. Za par "učitelj" i "profesor", naš model daje 0.542 naspram 0.570 za baseline, što je jedan od rijetkih primjera gdje Facebook model pokazuje bolji rezultat.

Za par "Zagreb" i "Split" (dva najveća hrvatska grada), naš model daje 0.578 naspram 0.587 za baseline. Ova usporediva sličnost sugerira da oba modela podjednako dobro prepoznaju geografske entitete iste kategorije.

#### Usporedba: Pravopisna korekcija

Za pogrešno napisanu riječ "riješenje", oba modela vraćaju ispravnu riječ "rješenje" na prvom mjestu, ali s različitim sličnostima: 0.861 za naš model i 0.758 za baseline. Viša sličnost kod našeg modela znači veću sigurnost u ispravak.

Za "prijateli" → "prijatelj" (tipfeler s izostavljenim slovom), naš model daje hibridni rezultat 0.798 naspram 0.798 za baseline - identične performanse. Međutim, za "kompjutr" → "kompjuter", naš model daje 0.869 naspram 0.868 za baseline, opet vrlo slično.

Ove usporedbe ukazuju na to da za čiste tipfelere (mehanička zamjena ili izostavljanje slova) oba modela postižu slične rezultate jer se oslanjaju na n-gram sličnost. Razlika postaje izraženija za semantičke zadatke gdje naš specijalizirani korpus pruža prednost.

---

## 7. Ograničenja i buduća poboljšanja

### 7.1 Identificirana ograničenja

Tijekom evaluacije identificirali smo nekoliko ograničenja trenutnog pristupa. Najznačajnije ograničenje odnosi se na razlikovanje dijakritičkih parova č/ć i š/ž. FastText algoritam temelji reprezentaciju riječi na n-gramima znakova, a riječi koje se razlikuju samo u jednom znaku dijele većinu svojih n-grama. Ovo inherentno ograničenje algoritma ne može se prevladati promjenom hiperparametara ili povećanjem korpusa za treniranje.

Drugo ograničenje vezano je uz kvalitetu korpusa. Analiza najbližih susjeda pokazala je da model uči i "prljave" varijante riječi s pridodanom interpunkcijom. Na primjer, za riječ "učitelj" među najbližim susjedima pojavljuje se "učitelj." (s točkom), "učitelji." i "učitelj-student". Za riječ "pisati" vidimo "pisati.", "pisati.--" i slično. Ovo proizlazi iz nedostataka u procesu tokenizacije tijekom predobrade korpusa - interpunkcija nije konzistentno odvojena od riječi.

Konkretno, za riječ "škola" u rezultatima Facebook modela vidimo ekstremne primjere poput "školaSrednja" i "školaŽupna" - očito spojene riječi gdje je nedostajao razmak u izvornom tekstu. Naš korpus pokazuje manje ovakvih artefakata, ali oni su i dalje prisutni. U budućim iteracijama ovo se može adresirati poboljšanim procesom čišćenja koji uključuje agresivniju tokenizaciju i filtriranje riječi koje sadrže interpunkciju.

Treće ograničenje je činjenica da FastText modeli ne uzimaju u obzir kontekst rečenice. Vektor riječi je isti neovisno o tome u kakvoj se rečenici pojavljuje. Za punu semantičku analizu i razumijevanje konteksta potrebni bi bili složeniji modeli poput BERT ili GPT arhitektura koje uzimaju u obzir cijelu rečenicu.

### 7.2 Preporuke za produkcijsku implementaciju

Na temelju rezultata ovog istraživanja, preporučujemo hibridni pristup za integraciju u produkcijski sustav ispravi.me. FastText modeli trebali bi služiti kao jedan od slojeva u višeslojnoj arhitekturi za predlaganje ispravaka.

Prvi sloj bi činila tradicionalna rječnička provjera koja identificira riječi koje nisu u rječniku. Drugi sloj bi koristio FastText model za generiranje kandidata za ispravak na temelju vektorske sličnosti. Treći sloj bi primijenio pravila za post-procesiranje, posebno za dijakritičke parove gdje FastText pokazuje slabosti. Primjerice, pravilo može provjeriti oba varijante (moč i moć) u rječniku i preferirati onu koja postoji kao standardna hrvatska riječ.

Dodatno preporučujemo implementaciju kontekstne analize koja bi uzela u obzir okolne riječi pri rangiranju kandidata za ispravak. Iako FastText ne modelira kontekst izravno, kosinusna sličnost između kandidata i kontekstnih riječi može poslužiti kao dodatni signal za rangiranje.

---

## 8. Zaključak

Ovaj projekt demonstrirao je potencijal FastText modela ugniježđenih vektora za unapređenje sustava za provjeru pravopisa hrvatskog jezika. Kroz sustavno istraživanje hiperparametara i razvoj prilagođenog evaluacijskog okvira, identificirali smo optimalnu konfiguraciju modela koja nadmašuje javno dostupni baseline za 20.3%.

Ključni nalazi projekta mogu se sažeti u nekoliko točaka. Prvo, Skip-gram arhitektura konzistentno nadmašuje CBOW za zadatak pravopisne korekcije. Drugo, veća dimenzionalnost vektora (300 naspram 100 ili 200) i dulje treniranje (15 naspram 10 epoha) rezultiraju boljim performansama. Treće, specifičnost i kvaliteta korpusa važnija je od same veličine za specijalizirane zadatke. Četvrto, FastText ima inherentna ograničenja za jednoznačne zamjene poput č/ć koja zahtijevaju dopunske pristupe.

Rezultati ovog istraživanja bit će integrirani u sljedeću verziju sustava ispravi.me, čime će korisnici dobiti pristup semantički inteligentnijem sustavu za provjeru pravopisa koji bolje razumije kontekst i morfologiju hrvatskog jezika.

---

## 9. Projektni tim i zahvale

### 9.1 Članovi tima

**Nikša Brala** (voditelj tima) - student Fakulteta elektrotehnike i računarstva, odgovoran za arhitekturu rješenja, koordinaciju aktivnosti i završnu integraciju komponenti. Kontakt: nb53922@fer.hr

**Teo Matošević** - student Fakulteta elektrotehnike i računarstva, zadužen za razvoj evaluacijskog okvira, analizu rezultata i izradu dokumentacije. Kontakt: tm54277@fer.hr

**Vitomir Brebrić** - student Fakulteta elektrotehnike i računarstva, zadužen za prikupljanje i predobradu korpusa te treniranje modela. Kontakt: vitomir.brebric@fer.hr

### 9.2 Vremenska crta projekta

Projekt je realiziran u razdoblju od studenog 2025. do siječnja 2026. godine. Prva faza (studeni - početak prosinca) obuhvatila je prikupljanje i čišćenje korpusa. Druga faza (prosinac) uključivala je pripremu okruženja za treniranje i inicijalne eksperimente. Treća faza (siječanj) fokusirala se na sustavno treniranje svih konfiguracija, evaluaciju i izradu dokumentacije. Projekt je uspješno završen i isporučen 30. siječnja 2026. godine.

### 9.3 Zahvale

Zahvaljujemo mentoru prof. dr. sc. Gordanu Gledecu na usmjeravanju i stručnim savjetima tijekom projekta. Također zahvaljujemo timu usluge ispravi.me na pristupu infrastrukturi i povratnim informacijama o praktičnim zahtjevima sustava.

---

## 10. Tehnički dodaci

### 10.1 Parametri najboljeg modela

Najbolji model ft_sg_d300_ws10_e15_mc2 koristi sljedeće parametre: arhitektura Skip-gram, dimenzionalnost vektora 300, veličina kontekstnog prozora 10 riječi, broj epoha 15, minimalna frekvencija riječi 2, minimalna duljina n-grama 3, maksimalna duljina n-grama 6, stopa učenja 0.05 (standardna), broj negativnih uzoraka 5 (standardni). Konačna veličina modela iznosi 8.9 GB, a vokabular obuhvaća 2,918,267 jedinstvenih riječi.

### 10.2 Popis svih treniranih modela

| Model | Arhitektura | Dimenzije | Prozor | Epohe | Rezultat |
|-------|-------------|-----------|--------|-------|----------|
| ft_sg_d300_ws10_e15_mc2 | Skip-gram | 300 | 10 | 15 | 0.528 |
| ft_sg_d300_ws10_e10_mc2 | Skip-gram | 300 | 10 | 10 | 0.495 |
| ft_sg_d300_ws10_e10_mc2_minn2_maxn7 | Skip-gram | 300 | 10 | 10 | 0.493 |
| ft_sg_d200_ws5_e3_mc2 | Skip-gram | 200 | 5 | 3 | 0.484 |
| ft_sg_d200_ws10_e3_mc2 | Skip-gram | 200 | 10 | 3 | 0.464 |
| ft_sg_d100_ws5_e3_mc2 | Skip-gram | 100 | 5 | 3 | 0.449 |
| ft_cbow_d300_ws10_e10_mc2 | CBOW | 300 | 10 | 10 | 0.445 |
| facebook_cc_hr_300 (baseline) | Skip-gram | 300 | 5 | - | 0.439 |
| ft_cbow_d200_ws5_e3_mc2 | CBOW | 200 | 5 | 3 | 0.438 |

### 10.3 Rezultati po kategorijama evaluacije

| Model | ije/je | Dijakritici | Tipfeleri | Ukupno |
|-------|--------|-------------|-----------|--------|
| ft_sg_d300_ws10_e15_mc2 | 0.748 | 0.832 | 0.769 | 0.528 |
| ft_sg_d300_ws10_e10_mc2 | 0.727 | 0.830 | 0.772 | 0.495 |
| ft_sg_d300_ws10_e10_mc2_minn2_maxn7 | 0.748 | 0.831 | 0.786 | 0.493 |
| ft_sg_d200_ws5_e3_mc2 | 0.744 | 0.837 | 0.793 | 0.484 |
| ft_cbow_d300_ws10_e10_mc2 | 0.762 | 0.822 | 0.806 | 0.445 |
| facebook_cc_hr_300 | 0.711 | 0.820 | 0.755 | 0.439 |

### 10.4 Primjer izlaza: Korekcija ije/je pogrešaka

Za pogrešno napisanu riječ "riješenje", najbolji model vraća sljedeće najbliže susjede:

1. **rješenje** (0.861) - ispravna riječ
2. riješenjem (0.796)
3. Riješenje (0.781)
4. rješenjema (0.740)
5. ješenje (0.732)
6. riješenja (0.718)
7. riješenju (0.709)
8. razriješenje (0.690)
9. dješenje (0.681)
10. rješenjenja (0.666)

Ispravna riječ "rješenje" nalazi se na prvom mjestu s kosinusnom sličnošću 0.861, što je izvrstan rezultat za zadatak pravopisne korekcije.

### 10.5 Primjer izlaza: Semantičko grupiranje dana u tjednu

Za riječ "ponedjeljak", najbolji model vraća sljedeće najbliže susjede:

1. utorak (0.940)
2. srijedu (0.928)
3. petak (0.923)
4. četvrtak (0.922)
5. nedjelju (0.839)
6. ponedjeljako (0.837)
7. ponedjeljaka (0.817)
8. uponedjeljak (0.808)
9. subotu (0.806)
10. ponedjeljakutorak (0.801)

Svih sedam dana u tjednu nalazi se među prvih deset susjeda (utorak, srijeda, petak, četvrtak, nedjelja, subota, te sam ponedjeljak u varijanti "ponedjeljako"), što demonstrira izvrsno semantičko grupiranje koncepata iste kategorije.

### 10.6 Primjer izlaza: Najbliži susjedi za "učitelj"

1. nastavnik (0.751) - semantički sinonim
2. učiteljev (0.734) - morfološka varijanta (posvojni pridjev)
3. naučitelj (0.730) - arhaična varijanta
4. učiteljâ (0.728) - morfološka varijanta (genitiv množine)
5. Učitelj (0.726) - kapitalizirana varijanta
6. učenik (0.724) - semantički povezan koncept
7. učitelja (0.705) - morfološka varijanta (genitiv jednine)
8. učiteljski (0.698) - morfološka varijanta (pridjev)
9. učitelj-učenik (0.695) - složenica
10. učitelju (0.689) - morfološka varijanta (dativ/lokativ)

Ovaj primjer ilustrira kombinaciju morfološke svijesti (sedam morfoloških varijanti riječi "učitelj") i semantičke svijesti (sinonim "nastavnik" i povezani koncept "učenik").

### 10.7 Primjer izlaza: Korekcija tipfelera "pisatti" → "pisati"

Za pogrešno napisanu riječ "pisatti" (udvostručeno slovo t), najbolji model vraća:

1. pisati (0.892) - ispravna riječ
2. napisati (0.843)
3. popisati (0.812)
4. dopisati (0.798)
5. ispisati (0.791)
6. prepisati (0.786)
7. zapisati (0.779)
8. pisanja (0.771)
9. opisati (0.768)
10. upisati (0.762)

Ispravna riječ "pisati" nalazi se na prvom mjestu s visokom sličnošću od 0.892. Zanimljivo je da model vraća i sve prefigirane varijante glagola (napisati, popisati, dopisati, itd.) što pokazuje razumijevanje morfološke strukture hrvatskih glagola.

### 10.8 Primjer izlaza: Neuspješna korekcija "moč" → "moć"

Za pogrešno napisanu riječ "moč" (č umjesto ć), najbolji model vraća:

1. močna (0.535)
2. pomoč (0.514)
3. močnika (0.511)
4. močnejša (0.503)
5. močnih (0.501)
6. nadmoč (0.493)
7. močni (0.492)
8. močnim (0.478)
9. močić (0.472)
10. močno (0.464)

Ispravna riječ "moć" ne pojavljuje se među prvih deset susjeda. Svi rezultati sadrže niz "moč" jer FastText tretira č i ć kao gotovo identične znakove s obzirom na preklapajuće n-grame. Ovaj primjer jasno demonstrira ograničenje FastText algoritma za dijakritičke parove.

### 10.9 Primjer izlaza: Usporedba modela za "škola"

Usporedba najbližih susjeda za riječ "škola" između našeg najboljeg modela i Facebook baseline:

**Naš model (ft_sg_d300_ws10_e15_mc2):**
1. škole (0.843)
2. učenika (0.803)
3. škol (0.784)
4. učenike (0.782)
5. učenici (0.760)
6. gimnazija (0.757)
7. nastava (0.751)
8. školam (0.744)
9. učenička (0.743)
10. učionica (0.742)

**Facebook baseline (cc.hr.300.bin):**
1. eko-škola (0.707)
2. auto-škola (0.703)
3. Eko-škola (0.695)
4. Ekoškola (0.690)
5. Škola (0.677)
6. školah (0.673)
7. školaSrednja (0.670)
8. školaŽupna (0.670)
9. e-škola (0.660)
10. autoškola (0.651)

Razlika je jasna: naš model vraća semantički povezane riječi (učenik, gimnazija, nastava, učionica) dok Facebook model vraća složenice i morfološke varijante bez semantičke dubine.

### 10.10 Primjer izlaza: Semantička sličnost parova riječi

| Par riječi | Naš model | Facebook | Očekivano | Razlika |
|------------|-----------|----------|-----------|---------|
| pisati - čitati | 0.793 | 0.657 | visoka | +20.7% |
| auto - vozilo | 0.679 | 0.408 | visoka | +66.4% |
| škola - učenik | 0.685 | 0.374 | visoka | +83.2% |
| Zagreb - Split | 0.578 | 0.587 | srednja | -1.5% |
| ponedjeljak - utorak | 0.940 | 0.821 | visoka | +14.5% |
| učitelj - profesor | 0.542 | 0.570 | visoka | -4.9% |
| škola - banana | 0.141 | 0.139 | niska | +1.4% |

Naš model pokazuje značajno bolje rezultate za većinu parova, posebno za "škola - učenik" gdje je razlika čak 83.2%. Facebook model je bolji samo za par "učitelj - profesor" i "Zagreb - Split", ali razlike su minimalne.

### 10.11 Primjer izlaza: Analogije riječi

| Analogija | Očekivano | Naš model | Facebook |
|-----------|-----------|-----------|----------|
| kralj:kraljica::princ:? | princeza | **princeza** | Šprinc |
| učitelj:učiteljica::student:? | studentica | **studentica** | **studentica** |
| Zagreb:Hrvatska::Rim:? | Italija | Rima | 16,3-5 |
| ponedjeljak:utorak::srijeda:? | četvrtak | srijedasubota | **četvrtak** |

Naš model uspješno rješava analogije bazirane na rodu (kralj/kraljica, učitelj/učiteljica) ali ne uspijeva na geografskim i sekvencijalnim analogijama. Facebook baseline pokazuje suprotno ponašanje - bolji je na sekvencijalnim analogijama ali lošiji na morfološkim.

### 10.12 Hibridni rezultati po testnim primjerima

| Kategorija | Primjer | Naš model | Facebook | Razlika |
|------------|---------|-----------|----------|---------|
| ije/je | riješenje → rješenje | 0.883 | 0.758 | +16.5% |
| ije/je | mlieko → mlijeko | 0.765 | 0.711 | +7.6% |
| ije/je | vriemena → vremena | 0.799 | 0.752 | +6.3% |
| dijakritici | moč → moć | 0.851 | 0.820 | +3.8% |
| dijakritici | kuča → kuća | 0.925 | 0.891 | +3.8% |
| tipfeleri | prijateli → prijatelj | 0.798 | 0.798 | 0.0% |
| tipfeleri | pisatti → pisati | 0.817 | 0.812 | +0.6% |
| tipfeleri | kompjutr → kompjuter | 0.869 | 0.868 | +0.1% |

Najveće razlike vidljive su u kategoriji ije/je pogrešaka, dok su performanse za tipfelere gotovo identične između modela.

---

*Dokument pripremljen za Fakultet elektrotehnike i računarstva, Sveučilište u Zagrebu*

*Verzija 2.0 | 30. siječnja 2026.*
