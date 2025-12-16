NASIL ÇALIŞTIRILIR?

1. Gerekli kütüphaneleri yükleyin

```bash
pip install -r requirements.txt
```
Kurulumun tamamlanmasını bekleyin.

2. Veri üretimi

Bağımlılıklar yüklendikten sonra aşağıdaki komutu çalıştırın:

```bash
python generate_data.py
```

Bu dosya, patients.csv içerisinde bulunan sentetik veriler üzerine
modelin daha sağlıklı çalışabilmesi için ek veri üretimi yapar.
Üretilen veriler dosyaya kaydedilir.

3. Model eğitimi

Veri üretimi tamamlandıktan sonra modeli eğitmek için:

```bash
python train.py
```

Bu adımda veriler kullanılarak başarılı şekilde çalışan modeller üretilir.
Eğitim tamamlandığında modeller ve çıktı dosyaları kaydedilir.

4. Uygulamanın çalıştırılması

Son olarak Streamlit uygulamasını başlatmak için:

```bash
streamlit run app.py
```

Bu komut ile web tabanlı uygulama çalıştırılır.
