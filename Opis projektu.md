# **Projekt zaliczeniowy – Zastosowania Uczenia Maszynowego**

## **1. Wprowadzenie**
W ramach projektu zaliczeniowego z przedmiotu **Zastosowania Uczenia Maszynowego (ZUM)** zrealizujesz indywidualny projekt obejmujący pełen proces uczenia maszynowego — od pozyskania i przygotowania danych po modelowanie i ewaluację wyników.

Projekt wykonujesz samodzielnie, korzystając z darmowych zasobów (np. Google Colab, Kaggle, HuggingFace).

---

## **2. Pierwszy krok – utworzenie repozytorium**
1. Otwórz szablon projektu:  
   [https://github.com/PJATK-ZUM/Projekt-koncowy-2025-2026](https://github.com/PJATK-ZUM/Projekt-koncowy-2025-2026)
2. Kliknij przycisk **“Use this template”** i wybierz **“Create a new repository”**.
3. Nazwij swoje repozytorium w formacie:  
   `ZUM_2025_TwojeImieNazwisko`
4. Ustaw widoczność repozytorium jako **publiczne** lub **prywatne** (jeśli prywatne, dodaj **dwnuk@pjwstk.edu.pl** jako **collaboratora**).
5. Sklonuj repozytorium lokalnie lub pracuj bezpośrednio w Colabie / Jupyter Notebooku.

---

## **3. Wybór tematu**
### **Krok 1 – wybierz obszar projektu**
Wybierz jeden z trzech obszarów:
- **Computer Vision (CV)** – praca z obrazami, np. klasyfikacja, detekcja.
- **Natural Language Processing (NLP)** – praca z tekstem, np. analiza sentymentu, klasyfikacja wiadomości.
- **Audio Processing (AP)** – praca z dźwiękiem, np. rozpoznawanie emocji, klasyfikacja gatunków muzycznych.

### **Krok 2 – wybierz konkretny temat**
Możesz wybrać własny temat lub skorzystać z poniższych propozycji:

#### **Computer Vision**
- Klasyfikacja owoców lub warzyw (np. Fruits360)
- Rozpoznawanie emocji z twarzy (FER2013)
- Detekcja cyfr odręcznych (MNIST)
- Klasyfikacja znaków drogowych (GTSRB)

#### **NLP**
- Klasyfikacja nastroju w recenzjach filmów (IMDb)
- Wykrywanie fake newsów (FakeNewsNet)
- Klasyfikacja tematów wiadomości (AG News)
- Analiza emocji w tweetach (TweetEval)

#### **Audio**
- Klasyfikacja dźwięków środowiskowych (ESC-50)
- Rozpoznawanie emocji w głosie (RAVDESS)
- Klasyfikacja gatunków muzycznych (GTZAN)
- Wykrywanie słów kluczowych (Google Speech Commands)

---

## **4. Zgłoszenie tematu**
1. Po wybraniu tematu, **zgłoś go na platformie Edux** w odpowiednim zadaniu.  
2. W opisie podaj:
   - wybrany **obszar** (CV / NLP / Audio),
   - **tytuł projektu**,
   - krótki opis pomysłu (2–3 zdania),
   - potencjalne **źródło danych** (jeśli już masz pomysł).
3. Poczekaj na **akceptację prowadzącego** przed rozpoczęciem realizacji projektu.

---

## **5. Realizacja projektu – krok po kroku**

Po akceptacji tematu zrealizuj wszystkie etapy projektu zgodnie z poniższymi instrukcjami.

### **Etap 1: Zebranie i analiza danych (EDA)**
- Znajdź dane z **jasno określoną licencją** (np. HuggingFace Datasets, Kaggle, PapersWithCode).  
- Opisz źródło danych w pliku README (nazwa, link, licencja).  
- Wczytaj dane i przeprowadź **eksploracyjną analizę danych (EDA)**:  
  - przeanalizuj strukturę, typy danych, rozkład klas, braki danych, duplikaty, outliery;  
  - wykonaj wizualizacje (wykresy, histogramy, heatmapy, chmury słów itp.);  
  - sformułuj wstępne wnioski.

Wskazówka: możesz inspirować się przykładami z wcześniejszych notebooków do pracy własnej — sposób pracy z danymi, analizy, wizualizacje i interpretacja.

---

### **Etap 2: Preprocessing i inżynieria cech**
Przygotuj dane do trenowania modeli:

- **NLP:** tokenizacja, czyszczenie tekstu, usunięcie stop słów, lematyzacja, wektoryzacja (TF-IDF, embeddings).
- **CV:** zmiana rozmiaru obrazów, normalizacja, augmentacja (obrót, przycięcie, odbicie).
- **Audio:** standaryzacja próbkowania, ekstrakcja cech (MFCC, spektrogram, Chroma).

Następnie wykonaj **feature engineering** – wyciągnij najistotniejsze cechy i, jeśli trzeba, zastosuj **redukcję wymiarów** (np. PCA).

---

### **Etap 3: Modelowanie**
W projekcie musisz utworzyć **trzy różne modele**:

1. **Model klasyczny ML**  
   - Przykłady: Logistic Regression, SVM, Random Forest, XGBoost.  
   - Traktuj go jako baseline do porównania z modelami głębokiego uczenia.  
   - Wytrenuj, oceń i zapisz wyniki.

2. **Sieć neuronowa zbudowana od zera**  
   - Zbuduj prostą architekturę (np. CNN, MLP, RNN).  
   - Zaprojektuj ją samodzielnie w TensorFlow lub PyTorch.  
   - Przeanalizuj działanie sieci (np. wpływ liczby epok, batch size, learning rate).

3. **Model transformerowy (fine-tuning)**  
   - Wybierz model z biblioteki HuggingFace odpowiedni dla Twojego typu danych:
     - NLP: BERT, DistilBERT, RoBERTa  
     - CV: Vision Transformer (ViT), Swin Transformer  
     - Audio: Wav2Vec2, HuBERT  
   - Wykonaj **fine-tuning** na swoich danych.  
   - Porównaj wyniki z poprzednimi modelami.

---

### **Etap 4: Ewaluacja**
- Oceń wszystkie modele na zestawie testowym.  
- Użyj odpowiednich metryk (accuracy, precision, recall, F1, ROC-AUC, MSE itp.).  
- Przedstaw wyniki w formie tabel i wykresów (np. macierz pomyłek, krzywa ROC, learning curve).  
- Porównaj modele i uzasadnij, który uzyskał najlepszy wynik.  
- Opisz możliwe przyczyny błędów i kierunki dalszego rozwoju.

---

## **6. Struktura repozytorium**

Repozytorium powinno mieć następującą strukturę:

```
projekt_zum_2025/
│
├── data/
│ ├── sample/ # mała próbka danych – kilka rekordów / obrazów
│ ├── raw/ # surowe dane (nie dodawaj dużych plików!)
│ └── processed/ # dane przetworzone
├── notebooks/
│ ├── 1_EDA.ipynb
│ ├── 2_Preprocessing_Features.ipynb
│ ├── 3_Models_Training.ipynb
│ └── 4_Evaluation.ipynb
├── models/
├── results/
├── README.md
└── requirements.txt
```


> **Ważne:** Nie dodawaj pełnego zbioru danych do repozytorium.  
> Repo ma zawierać tylko mały przykład danych (np. `data/sample/`) oraz instrukcję, jak pobrać pełny zbiór w notatniku.
Każdy etap powinien być opisany w osobnym notatniku `.ipynb`, z komentarzami i uzasadnieniem wyborów.

---

## **7. Dokumentacja projektu**
W pliku `README.md` zamieść:
- nazwę projektu i swoje dane,  
- opis danych i ich źródło (z licencją),  
- cel projektu,  
- opis kolejnych etapów,  
- listę modeli i użytych metryk,  
- krótkie podsumowanie wyników i wniosków.

---

## **8. Technologie i biblioteki**
Używaj narzędzi i bibliotek omawianych na zajęciach:

- **Dane:** numpy, pandas  
- **Wizualizacja:** matplotlib, seaborn, plotly  
- **Uczenie klasyczne:** scikit-learn, xgboost  
- **Uczenie głębokie:** TensorFlow, PyTorch  
- **Transformery:** HuggingFace (transformers, datasets)  

---

## **9. Oddanie projektu**
1. Upewnij się, że Twoje repozytorium zawiera wszystkie notatniki i plik README.  
2. Sprawdź, czy wszystkie komórki w notatnikach wykonują się poprawnie.  
3. Upewnij się, że dane nie łamią licencji.  
4. Zgłoś link do repozytorium w zadaniu na Eduxie do wyznaczonego terminu.

---

## **10. Licencja**
Projekt udostępniony na licencji:  
*(np. MIT License, CC-BY 4.0, GPL-3.0)*  

Źródło danych: zgodnie z licencją wskazaną w sekcji **Dane**.

---

## **11. Wskazówki**
- Inspiruj się zadaniami przerabianymi na zajęciach i przykładami z notatników.  
- Staraj się komentować i uzasadniać swoje decyzje (np. wybór modelu, metryk, hiperparametrów).  
- Skup się na jakości kodu i opisie – projekt ma być czytelny i samodokumentujący się.  
- Jeśli napotkasz problemy z danymi lub modelem, pokaż swoje podejście i rozwiązania — to także element oceny.

**Uwaga dotycząca danych:**  
Nie wrzucaj do repozytorium pełnego zbioru danych!  
- Jeśli dane są duże, zamieść **jedynie niewielką próbkę** (np. 100–200 rekordów lub kilka obrazów / plików audio) w folderze `data/sample/`, aby pokazać strukturę danych.  
- Pełen zbiór powinien być pobierany w kodzie (np. z HuggingFace lub Kaggle API) lub dostępny przez link w README.  
- Dzięki temu repozytorium pozostanie lekkie i zgodne z zasadami licencyjnymi.
---

