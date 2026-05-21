# -Enviromental-Sound-Classification-using-Wav2vec2-
Enviromental Sound Classification using Fine-Tuning Wav2Vec2 on ESC-50 dataset.
# 🎧 Environmental Sound Classification using Wav2Vec2 (Audio ML)

Ushbu loyiha atrof-muhitdagi tabiiy va sun'iy tovushlarni (shovqinlarni) chuqur o'rganish (Deep Learning) algoritmlari yordamida avtomatik tarzda klassifikatsiya qilishga mo'ljallangan. Loyihada Meta (Facebook) kompaniyasining audio signallari bilan ishlashda g'oyat yuqori samaradorlik ko'rsatuvchi Wav2Vec2 arxitekturasi asos qilib olindi va maxsus datasetda qayta o'rgatildi (Fine-Tuning).

---

## 📌 Loyiha Haqida (Project Overview)
An'anaviy kompyuter ko'rish (Computer Vision) yoki matnli NLP loyihalaridan farqli o'laroq, ushbu loyiha Audio Machine Learning (Ovozli ma'lumotlar tahlili) yo'nalishida amaliy ko'nikma hosil qilish uchun yaratilgan. Loyihaning asosiy maqsadi — uzluksiz raqamli ovoz to'lqinlarini neyron tarmoq tushunadigan akustik xususiyatlar (features) ko'rinishiga keltirish va ularni 50 xil sinf bo'yicha yuqori aniqlikda tasniflashdir.

### 🔥 Asosiy Imkoniyatlari:
- Ovoz to'lqinlarini standart chastotaga keltirish (Audio Resampling).
- Audio signallarini raqamli matritsaga o'girish (Feature Extraction).
- Oldindan o'rgatilgan SOTA (State-of-the-Art) transformer modelini moslashtirish.
- GPU (CUDA T4) muhitida optimallashtirilgan o'rgatish (Fine-Tuning) quvuri.

---

## 📊 Dataset Ma'lumotlari (Data Source)
Loyihani amalga oshirishda Hugging Face ekotizimidagi mashhur **ashraq/esc50** (Environmental Sound Classification) ma'lumotlar to'plamidan foydalanildiSinflar soni (Classes):):** 50 ta har xil tovush toifasi (masalan: it hurishi, yomg'ir ovozi, mashina signali, dengiz to'lqinlari, chaqmoq va h.k.)Audio formati:i:** Amplituda to'lqinlari massivi (Waveform raw array)Sampling Rate (Chastota):):** Modelning ichki arxitektura talabidan kelib chiqib, barcha audio fayllar majburiy ravis16,000 Gts (16 kHz)z)** chastotaga o'tkazildi.

---

## 🏗️ Texnologiyalar Steki (Tech StackDasturlash tili:i:** Python Deep Learning Freymvorki:i:** PyTorch (Torch AudioKutubxonalar:r:** Hugging Face Transformers, Hugging Face DatasetHisoblash muhiti:i:** Google Colab (T4 GPU Accelerator



# Audiolarni 16000 Gts chastotaga o'tkazish
sampling_rate = extractor.sampling_rate
small_train = small_train.cast_column("audio", Audio(sampling_rate=sampling_rate))
