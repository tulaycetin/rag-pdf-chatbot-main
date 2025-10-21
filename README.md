# 📚 Basit RAG Chatbot — README

Bu doküman, Streamlit ile hazırlanmış **Basit RAG (Retrieval-Augmented Generation) Chatbot** uygulamasının tüm bileşenlerini, mimarisini, kullanılan modelleri, vektör veritabanını, kurulum ve dağıtım bilgilerini açıklar. Eğitimsel amaçlı sade bir örnek olup gerçek üretim kullanımı için ek güvenlik ve ölçeklendirme adımları gereklidir.

## 🎥 Demo Video

[![RAG PDF Chatbot Demo](https://img.youtube.com/vi/7gzGqXkb3yw/maxresdefault.jpg)](https://youtu.be/7gzGqXkb3yw)

**[▶️ Demo Videoyu İzle](https://youtu.be/7gzGqXkb3yw)**

> Uygulamanın çalışma şeklini gösteren demo videosu


**[uygulama Deploy Link] ([rag-pdf-chatbot_Streamlit](https://rag-pdf-chatbotapp.streamlit.app/))
---

## İçindekiler

1. Proje Özeti
2. Mimari (RAG akışı)
3. Kullanılan Kütüphaneler ve Modeller
4. Veri Akışı ve Chunklama
5. Vektör Veritabanı (FAISS)
6. Embedding (Vektörleştirme)
7. LLM Entegrasyonu (Google Gemini)
8. Uygulama Dosya Açıklaması
9. Ortam Değişkenleri (.env örneği)
10. Kurulum ve Çalıştırma
11. Dağıtım Notları
12. Güvenlik, Gizlilik ve Maliyet Uyarıları
13. Sık Karşılaşılan Hatalar & Çözümleri
14. Geliştirme Önerileri ve İyileştirmeler
15. Lisans

---

## 1) Proje Özeti

Bu uygulama, kullanıcıdan yüklenen bir PDF dosyasını okuyup parçalara (chunk) ayırır, bu parçaları embedding modeline sokarak bir vektör veritabanı (FAISS) oluşturur. Kullanıcının sorduğu sorular için en alakalı chunk'lar geri getirildikten sonra bir LLM (burada Google Gemini - `langchain_google_genai.ChatGoogleGenerativeAI`) kullanılarak cevap üretilir. Ayrıca tamamı veya sayfa sayfa özetleme ve soru üretme (quiz generation) özellikleri bulunur.

## 2) Mimari (RAG akışı)

1. **PDF yükleme** — Kullanıcı Streamlit üzerinden PDF yükler.
2. **Metin çıkarımı** — `PyPDF2` ile sayfa bazlı metin çekilir.
3. **Chunklama** — `RecursiveCharacterTextSplitter` ile uzun metinler mantıklı uzunlukta parçalara ayrılır.
4. **Embedding** — Her chunk için `HuggingFaceEmbeddings` (sentence-transformers/all-MiniLM-L6-v2) ile vektörler oluşturulur.
5. **Vektör Veritabanı** — FAISS içine chunk embedding'leri yerleştirilir.
6. **Retriever** — Benzerlik araması (k = 4) ile en alakalı chunk'lar seçilir.
7. **LLM (GenAI)** — Seçilen chunk'lar ve kullanıcı promptu LLM'e verilerek cevap oluşturulur (RAG step).
8. **Sunum** — Streamlit UI üzerinden özet, sayfa özeti, soru üretme ve chat cevapları gösterilir.

## 3) Kullanılan Kütüphaneler ve Modeller

* `streamlit` — Web arayüzü
* `PyPDF2` — PDF'ten metin çıkartma
* `langchain` — RAG için zincir, text-splitting, retriever vs.
* `langchain_community.embeddings.HuggingFaceEmbeddings` — SentenceTransformers tabanlı embedding
* `FAISS` (via `langchain_community.vectorstores`) — Vektör veritabanı (in-memory)
* `langchain_google_genai.ChatGoogleGenerativeAI` — Google Gemini ile entegrasyon (LLM)
* `dotenv` — Ortam değişkenleri yönetimi

**Modeller:**

* Embedding: `sentence-transformers/all-MiniLM-L6-v2` (küçük, hızlı, CPU dostu)
* LLM: `gemini-2.0-flash` (örnek), Google Generative API anahtar ile kullanılır



## 4) Veri Akışı ve Chunklama

* `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` ile metinler 1000 karakterlik parçalar halinde kesilir ve 200 karakter overlap bırakılır.
* Neden overlap? Overlap, anlam bütünlüğünü korur ve aranan bilgi sınırlarında kesilmeyi azaltır.
* Chunk boyutu ve overlap uygulamaya göre ayarlanmalıdır: çok büyük chunk LLM contexte yükünü artırır; çok küçük chunk bağlamı kaybettirir.

## 5) Vektör Veritabanı (FAISS)

* FAISS, bellek içi (in-memory) bir vektör arama motorudur. Bu uygulamada küçük/orta ölçekli belgeler için uygundur.


## 6) Embedding (Vektörleştirme)

* Kullanılan embedding modeli `all-MiniLM-L6-v2` — küçük, hızlı ve yeterli kalitede semantik benzerlik sağlar.


## 7) LLM Entegrasyonu (Google Gemini)

* Uygulama `langchain_google_genai.ChatGoogleGenerativeAI` üzerinden `gemini-2.0-flash` modelini çağırıyor.


## 8) Uygulama Dosya Açıklaması

* `Basic_Rag.py` (veya mevcut Streamlit dosyanız): Uygulama arayüzü, PDF yükleme, chunklama, vektör DB oluşturma, LLM çağrıları
* `requirements.txt`: Gereken Python paketleri
* `.env`: Gizli anahtarlar (API_KEY, GOOGLE_APPLICATION_CREDENTIALS vs.)

## 9) Ortam Değişkenleri (.env örneği)

```env
# Google Generative AI
GOOGLE_API_KEY=AIza...REPLACE_WITH_YOURS

# Opsiyonel: HuggingFace TOKEN (private model kullanıyorsanız)
HUGGINGFACE_API_TOKEN=

# Streamlit konfigürasyonu (opsiyonel)
STREAMLIT_SERVER_PORT=8501
```

 

## 10) Kurulum ve Çalıştırma

 

**2. Gereksinimleri yükleyin (`requirements.txt` örneği aşağıda):**

```bash
pip install -r requirements.txt
```

**3. `.env` dosyasını oluşturun ve API anahtarınızı ekleyin.**

**4. Uygulamayı başlatın:**

```bash
streamlit run Basic_Rag.py
```

**Örnek `requirements.txt`**

```
streamlit
PyPDF2
langchain
langchain-community
faiss-cpu
python-dotenv
google-generativeai
sentence-transformers
transformers
torch
```

