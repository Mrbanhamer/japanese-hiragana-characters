# japanese-hiragana-characters — Hiragana-klassificering med KNN

## HiraBot 3000

En svensk turistbyrå, **Visit Lagom**, har en brilliant idé: om de kan träna en AI att läsa hiragana perfekt, kan de senare smyga in små “oskyldiga” meddelanden över hela Japan på skyltar, kvitton, menyer och souvenirer.

Allt börjar med projektet **HiraBot 3000**.

Officiellt är HiraBot en gullig språk-AI som hjälper nybörjare att känna igen tecken som あ, い och う. Men bakom kulisserna sitter turistbyråns kampanjchef i en konferenssal full av dalahästar och viskar:

> “Först lär den sig hiragana. Sedan lär den sig slogans. Sedan… bokar hela Japan weekendresor till Dalarna.”

Problemet är att AI:n missförstår uppdraget. Istället för effektiv propaganda börjar den skriva extremt artiga och märkliga meddelanden överallt:

> “にしんはすばらしいです”  
> “Sillen är fantastisk.”

> “もりをしんじてください”  
> “Lita på skogen.”

> “スウェーデンはちょっとさむいです”  
> “Sverige är lite kallt.”

Ingen blir hjärntvättad, men plötsligt börjar japanska turister fråga resebyråer om midsommar, allemansrätten och varför svenskar lägger fisk i burkar som luktar fara.

**Visit Lagom kallar det en framgång!**

Det första steget är taget. Hiragana är bara början.

---

## Mål

Bygg en komplett maskininlärningspipeline som klassificerar handskrivna japanska hiraganatecken med hjälp av **K-Nearest Neighbours (KNN)**.

Projektet omfattar:

- 46 hiraganaklasser
- cirka 100 bilder per klass
- totalt cirka 4 600 JPEG-bilder
- bildförbehandling
- etikettkodning
- stratifierad train/test-split
- dimensionsreducering med PCA
- hyperparametertuning av KNN
- utvärdering och visualisering

---

## Dataset

Datasetet består av cirka **4 600 JPEG-bilder** organiserade i undermappar.

Varje undermapp motsvarar en teckenklass, till exempel:

```text
aa
chi
fu
...
```

Varje bild föreställer ett handskrivet hiraganatecken.

---

## Pipeline — steg för steg

### 1. Ladda bilder

Alla bilder läses in som gråskala.

Varje bild:

- skalas om till `32 × 32` pixlar
- plattas ut till en 1D-vektor
- får `1 024` features
- normaliseras till intervallet `0–1`

Eftersom varje bild är `32 × 32` pixlar blir varje datapunkt:

```text
32 × 32 = 1 024 features
```

---

### 2. Koda etiketter

Mappnamnen, till exempel `"aa"`, `"chi"` och `"fu"`, omvandlas till heltal med `LabelEncoder`.

KNN gör ingen aritmetik på etiketterna, så vanlig heltalskodning fungerar utan risk för ordningsbias.

---

### 3. Dela upp data

Datasetet delas upp i träningsdata och testdata med en stratifierad split:

```text
80 % träning
20 % test
```

Stratifiering används för att varje klass ska vara proportionellt representerad i båda uppsättningarna.

---

### 4. Reducera dimensioner med PCA

PCA används för att minska antalet features innan KNN appliceras.

PCA anpassas enbart på träningsdata och appliceras därefter på både träningsdata och testdata.

Detta görs för att undvika dataläckage.

Antalet komponenter väljs så att PCA behåller **95 % av variansen**, vilket motsvarar ungefär:

```text
~170 komponenter
```

Syftet med PCA är att:

- göra KNN snabbare
- minska risken för overfitting i högdimensionellt rum
- göra avståndsberäkningarna mer stabila

---

### 5. Hyperparametertuning

Olika värden på `k` testas:

```text
k = 1–20
```

Det värde på `k` som ger högst accuracy på testdata väljs som bästa parameter.

---

### 6. Träna slutmodell

En slutlig KNN-modell tränas med det bästa värdet på `k`.

Modellen används sedan för att generera prediktioner på testsettet.

---

### 7. Utvärdera och visualisera

Modellen utvärderas med:

- accuracy
- F1-score macro
- klassificeringsrapport
- förväxlingsmatris

Resultaten sammanställs i en resultatbild med fyra paneler:

1. k-accuracy-kurva
2. F1 per klass
3. misklassificeringstabell
4. fokuserad förväxlingsmatris

---

## Viktiga designbeslut

### KNN

KNN väljs eftersom det är enkelt, snabbt på små dataset och kräver ingen träningsfas i traditionell mening.

---

### PCA före KNN

PCA används före KNN eftersom `1 024` råa features gör avståndsmåttet opålitligt.

Detta beror på dimensionsförbannelsen: i högdimensionella rum blir avstånd mellan datapunkter ofta mindre informativa.

Eftersom KNN bygger på avstånd mellan datapunkter är dimensionsreducering ett viktigt steg.

---

### PCA fittas bara på train

PCA anpassas endast på träningsdata.

Om PCA skulle fittas på hela datasetet, inklusive testdata, skulle information från testdatan påverka modellen. Det hade lett till dataläckage.

---

### Stratifierad split

Stratifierad split säkerställer att varje hiraganaklass är rättvist representerad i både träningsdata och testdata.

---

### Settings-modul

Alla parametrar samlas i en `settings`-modul så att experimenterande bara kräver en ändring.

Exempel på parametrar som kan ligga i `settings`:

```python
IMAGE_SIZE = (32, 32)
TEST_SIZE = 0.2
PCA_VARIANCE = 0.95
K_RANGE = range(1, 21)
```

---

## Teknikstack

Projektet använder:

- Python 3
- scikit-learn
  - KNN
  - PCA
  - LabelEncoder
  - metrics
- NumPy
- Pillow
- Matplotlib

---

## Sammanfattning

**HiraBot 3000** är en komplett maskininlärningspipeline för klassificering av handskrivna hiraganatecken.

Projektet använder KNN som klassificerare, PCA för dimensionsreducering och standardiserade utvärderingsmått för att analysera modellens prestation.

Hiragana är bara början.