<div align="center">
  <h1>Bagian 2 (DoE + Supervised Learning) dan bagian 3 (Unsupervised Learning)</h1>
</div>

## Author
|Nama|NIM|
|-|-|
|Naufal Adnan|13522116|

## Ringkasan
Reinforcement learning merupakan salah satu jenis machine learning dimana suatu agent melakukan aksi terhadap lingkungannya, kemudian agent akan menerima reward atau penalty terhadap aksi yang telah dilakukannya. Tujuan akhir dari interaksi tersebut adalah membuat agent berperilaku sedemikian sehingga dapat memaksimalkan reward yang diterima.

Repository ini berisi implementasi dari reinforcement learning terhadap suatu suatu game sederhana dengan aturan sebagai berikut:
- Terdapat papan 1 dimensi (gerakan kiri kanan saja) sepanjang 10 kotak
- Terdapat lubang di titik 0, dan apel di titik 9
- Player berada pada titik 2 dan dapat bergerak ke kiri atau kanan
- Jika player jatuh ke dalam lubang, point yang didapatkan -100, jika player mendapatkan apel, point yang didapatkan +100. Jika player menempati titik lain, player akan mendapatkan point -1
- Jika player jatuh ke lubang atau mendapatkan apel, player kembali ke titik 3
- Player menang saat mendapatkan point +500
- Player kalah saat mendapatkan point -200


## Cara Penggunaan
1. Clone repository
``` sh
    git clone https://github.com/nanthedom/Reinforcement_Learning
```
2. Buka Terminal atau CMD
3. Pindah ke directory
``` sh
    cd src
```
3. Run
``` sh
    py reinforcement_learning.py
```

## Reinforcement Learning (Bagian 4)
- [v] Q-LEARNING
- [v] SARSA
