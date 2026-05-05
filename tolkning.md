# Tolkning av resultat

## Förväxlingsmatris (Confusion Matrix)

Förväxlingsmatrisen är ett kraftfullt verktyg för att förstå modellens fel. Så här tolkar vi den:

1.  **Diagonalen (Korrekt):** Det blå/röda stråket från uppe-vänster till nere-höger representerar korrekta förutsägelser. Ju starkare färg längs diagonalen, desto bättre är modellen på att pricka rätt klass.
2.  **Utanför diagonalen (Förväxlingar):** Celler utanför diagonalen visar var modellen blandar ihop tecken.
    *   **Rad (i):** Faktisk klass.
    *   **Kolumn (j):** Predikterad klass.
    *   Ett högt värde i cell (i, j) betyder att modellen ofta gissar på klass *j* när det egentligen var klass *i*.
3.  **Fokuserad matris (Topp 5 fel):** Denna visualisering isolerar de klasser som modellen har störst problem med, vilket hjälper oss att identifiera grafiskt liknande hiragana-tecken.

## Utvärderingsmått: Vad betyder de?

*   **Accuracy (Noggrannhet):** Andelen korrekta gissningar av det totala antalet förutsägelser. Ett enkelt mått, men kan vara missvisande om klasserna är obalanserade.
*   **Precision (Precision):** Av alla gånger modellen gissade på klass X, hur ofta var det faktiskt klass X? (Mäter hur "pålitlig" modellen är när den säger att det är klass X).
*   **Recall (Känslighet):** Av alla faktiska fall av klass X, hur många hittade modellen? (Mäter modellens förmåga att inte missa klass X).
*   **F1-score (F1-värde):** Det harmoniska medelvärdet av Precision och Recall. Ett utmärkt sätt att få ett balanserat mått när man vill väga både Precision och Recall lika högt.
*   **AP (Average Precision):** Ett mått som sammanfattar Precision-Recall-kurvan. Ett högre värde (närmare 1) indikerar att modellen är duktig på att bibehålla hög precision vid olika recall-nivåer.

## Analys i KNN-kontext

När vi arbetar med riktig data kan mönster i matrisen bero på:
*   **Grafisk likhet:** Tecknen är för lika i den upplösning vi använder (32x32).
*   **PCA:** De detaljer som behövs för att skilja specifika tecken åt kan ha gått förlorade under dimensionsreduceringen.
*   **K-värdet:** För litet *k* kan göra modellen känslig för brus, medan för stort *k* kan sudda ut beslutsgränserna.

## Rekommenderad åtgärd
Om du ser systematiska förväxlingar:
1. Undersök de specifika tecknen (t.ex. bildkvalitet).
2. Justera antalet PCA-komponenter för att behålla mer varians.
3. Utvärdera om ett annat *k*-värde minskar förväxlingarna mellan just dessa tecken.
