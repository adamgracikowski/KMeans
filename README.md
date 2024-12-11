# K-Means Clustering:

This project demonstrates the implementation of the K-Means clustering algorithm using three different approaches: 
- CPU-based computation, 
- GPU computation using custom CUDA kernels, 
- and GPU computation utilizing the Thrust library. 

The goal is to compare the performance and efficiency of these methods in clustering large datasets.

The project was implemented as a part of the Graphic Processors in Computational Applications course at Warsaw University of Technology during the winter semester of the 2024-2025 academic year.

<p align="center">
  <img src="Images/01.PNG"/>
</p>

## Specyfikacja danych:
- $N$ - liczba punktów ($1 \leq N \leq 50 \times 10^6$)
- $d$ - liczba wymiarów przestrzeni ($1 \leq d \leq 20$)
- $k$ - liczba centroidów ($1 \leq k \leq 20$)

## Format danych wejściowych:

W programie zaimplementowana dwa formaty danych wejściowych.

### Format tekstowy:

Pierwsza linia pliku zawiera $3$ liczby naturalne, rozdzielone białymi znakami. Liczby te są interpretowane jako $N$, $d$ oraz $k$.
Następne $N$ linii jest interpretowanych jako punkty o $d$ współrzędnych. Współrzędne te powinny być rozdzielone białymi znakami.
Wśród punktów pierwsze $k$ jest interpretowanych jednocześnie jako początkowe położenia centroidów.

Przykładowo, dla $N=4$, $d=3$ oraz $k=2$ plik wejściowy wygląda następująco:

```
4 3 2
12.20  1.12  5.55
34.45  5.23  2.34
65.33  1.10  4.40
4.90   3.34  0.12
```

### Format binarny:

Format ten jest w pełni analogiczny do tekstowego. 
Pierwsze $12$ bajtów to parametry $N$, $d$ i $k$. 
Dalsze $N$ porcji danych to współrzędne kolejnych punktów, zapisane jako $4$-bajtowe liczby rzeczywiste typu `float`.

## Format danych wyjściowych:

Wyniki są zapisywane wyłącznie w formacie tekstowym. 
Pierwsze $k$ linii to współrzędne centroidów, wyznaczone przez algorytm. 
Kolejne $N$ linii zawiera pojedynczą liczbę naturalną, oznaczającą przynależność punktu do odpowiedniego centroidu (kolejność punktów odpowiada danym wejściowym).

Przykładowo, dla $N=4$, $d=3$ oraz $k=2$ plik wyjściowy może wyglądać następująco:

```
12.20  1.12  5.55
34.45  5.23  2.34
0
1
2
1
```

## Uruchomienie programu:

```c
KMeans data_format computation_method input_file output_file
```

Program pobiera 4 parametry pozycyjne:
- `data_format`, który określa format danych wejściowych (`txt`|`bin`)
- `computation_method`, który określa zastosowany algorytm (`cpu`|`gpu1`|`gpu2`)
- `input_file`, który określa ścieżkę do pliku wejściowego w odpowiednim formacie
- `output_file`, który określa ścieżkę do pliku wyjściowego
  - jeżeli plik nie istnieje, to zostanie utworzony
  - w przyciwnym przypadku jego zawartość zostanie nadpisana przez aktualne wywołanie programu

## Features:

## Prerequisites:

## Running the Program:

## Input Format:

## Output Format:

## Visualization:

<p align="center">
  <img src="Images/02.PNG"/>
</p>

<p align="center">
  <img src="Images/03.PNG"/>
</p>

<p align="center">
  <img src="Images/04.PNG"/>
</p>

## Author:

My GitHub: [@adamgracikowski](https://github.com/adamgracikowski)

## Contributing:

All contributions, issues, and feature requests are welcome! 🤝

## Show your support:

Give a ⭐️ if you like this project and its documentation!
