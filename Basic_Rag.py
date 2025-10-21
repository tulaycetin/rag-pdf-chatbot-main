# -------------------------------
#  RAG Tabanlı PDF Analiz ve Sohbet Botu
# -------------------------------
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

# -------------------------------
# ⚙️ Streamlit Sayfa Yapılandırması (HER ZAMAN EN BAŞTA OLMALI!)
# -------------------------------
st.set_page_config(page_title="📚 Basit RAG Chatbot", layout="wide")

# -------------------------------
# 🔐 .env Dosyasını Yükle
# -------------------------------
load_dotenv()

# -------------------------------
# 🔑 API Key Yönetimi (Streamlit Cloud + Local)
# -------------------------------
def get_api_key():
    """
    API key'i önce Streamlit Secrets'tan almaya çalışır (Streamlit Cloud için),
    bulamazsa .env dosyasından alır (local development için)
    """
    try:
        # Streamlit Cloud'da secrets kullan
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Local development'ta .env dosyasından oku
        return os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY = get_api_key()

# Başlık
st.title("📚 Basit RAG Chatbot (Eğitimsel Versiyon)")

# API Key kontrolü
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin veya Streamlit Cloud'da Secrets ayarlarını yapın.")
    st.info("**Streamlit Cloud kullanıyorsanız:** Settings → Secrets kısmından GOOGLE_API_KEY ekleyin")
    st.info("**Local kullanıyorsanız:** .env dosyasında GOOGLE_API_KEY tanımlayın")
    st.stop()



# -------------------------------
#  PDF Yükleme
# -------------------------------
pdf = st.file_uploader("Bir PDF yükle", type=["pdf"])
if not pdf:
    st.stop()

# -------------------------------
#  PDF → Metin Dönüşümü
# -------------------------------
pdf_reader = PdfReader(pdf)
text = ""
pages_text = []  # Her sayfanın metnini ayrı tutmak için

for idx, page in enumerate(pdf_reader.pages, 1):
    page_text = page.extract_text()
    if page_text:
        text += page_text
        pages_text.append({
            'page_num': idx,
            'text': page_text
        })

st.success(f" PDF başarıyla okundu! Toplam {len(pages_text)} sayfa bulundu.")

# 🔧 Yardımcı Fonksiyon
# -------------------------------
def get_text_content(response):
    """
    LLM (Large Language Model) yanıtlarından yalnızca metin içeriğini çıkarır.
    - Eğer response bir nesne ve 'content' özniteliğine sahipse, onu döndürür.
    - Eğer doğrudan string tipindeyse string'i döndürür.
    - Diğer durumlarda response'u string'e çevirip döndürür.
    """

    if hasattr(response, 'content'):
        # Bazı LLM yanıtlarında content alanı olabilir (örneğin: OpenAI objesi)
        return response.content
    elif isinstance(response, str):
        # Response doğrudan metinse
        return response
    else:
        # Diğer olası tipler (örneğin dict veya Response objesi)
        return str(response)

# -------------------------------
# Text Splitting (Chunklama)
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

# -------------------------------
# Embedding Oluşturma
# -------------------------------
with st.spinner("🔍 Embedding (vektörleştirme) işlemi..."):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

st.success(" Vektör veritabanı hazır!")

# -------------------------------
# Soru-Cevap Zinciri (RAG)
# -------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# -------------------------------
#  Özetleme ve Soru Üretme
# -------------------------------
st.subheader(" PDF Özeti ve Soru Üretimi")

ozet_turu = st.radio(
    "İşlem Türünü Seç:",
    [" Komple PDF Özeti", " Her Sayfa Ayrı Özet", " Sorular Üret"],
    horizontal=True,
    help="Komple özet: Tüm PDF'i özetler | Sayfa sayfa: Her sayfayı ayrı özetler | Sorular: Test soruları üretir"
)

if st.button(" Başlat", type="primary", use_container_width=True):

    # ========== KOMPLE PDF ÖZETİ ==========
    if ozet_turu == " Komple PDF Özeti":
        with st.spinner(" PDF özetleniyor... (RAG ile tüm PDF taranıyor)"):
            summary_query = """Bu PDF'in kapsamlı bir özetini Türkçe olarak çıkar.
            Ana konuları, önemli noktaları ve sonuçları içersin.
            Mümkün olduğunca detaylı ama öz ol."""

            response = qa_chain.run(summary_query)
            summary = get_text_content(response)

        st.success(" Komple PDF Özeti Hazır!")
        st.write(summary)
        st.caption(f" RAG kullanıldı: Toplam {len(chunks)} chunk tarandı, en alakalı 4 chunk seçildi")

        # İndirme butonu
        st.download_button(
            label=" Özeti İndir (TXT)",
            data=summary,
            file_name="komple_pdf_ozeti.txt",
            mime="text/plain"
        )

    # ========== HER SAYFA AYRI ÖZET ==========
    elif ozet_turu == " Her Sayfa Ayrı Özet":
        st.info(f" {len(pages_text)} sayfa tek tek özetleniyor...")

        all_summaries = []
        progress_bar = st.progress(0)

        for idx, page_data in enumerate(pages_text):
            page_num = page_data['page_num']
            page_content = page_data['text']

            # Progress güncelle
            progress = (idx + 1) / len(pages_text)
            progress_bar.progress(progress)

            with st.spinner(f" Sayfa {page_num}/{len(pages_text)} özetleniyor..."):
                # Her sayfa için LLM ile özet
                page_summary_prompt = f"""Aşağıdaki metni Türkçe olarak özetle.
                Kısa ve öz olsun (2-3 cümle):

                {page_content[:2000]}"""

                response = llm.invoke(page_summary_prompt)
                page_summary = get_text_content(response)

                all_summaries.append({
                    'page': page_num,
                    'summary': page_summary
                })

        progress_bar.empty()
        st.success(f" Tüm {len(pages_text)} sayfa özetlendi!")

        # Özetleri göster
        for summary_data in all_summaries:
            with st.expander(f" Sayfa {summary_data['page']} Özeti"):
                st.write(summary_data['summary'])

        # Tüm özetleri indir
        all_summaries_text = "\n\n".join([
            f"=== SAYFA {s['page']} ===\n{s['summary']}\n{'-'*60}"
            for s in all_summaries
        ])

        st.download_button(
            label=" Tüm Özetleri İndir (TXT)",
            data=all_summaries_text,
            file_name="sayfa_sayfa_ozetler.txt",
            mime="text/plain"
        )

    # ========== SORULAR ÜRET ==========
    else:  # Sorular Üret
        with st.spinner(" Sorular oluşturuluyor... (RAG ile tüm PDF taranıyor)"):
            questions_query = """Bu PDF içeriğine dayanarak Türkçe olarak 5-7 önemli soru üret.
            Sorular PDF'in farklı bölümlerinden olsun ve kavrama düzeyini test etsin.
            Her soruyu madde işareti ile listele."""

            response = qa_chain.run(questions_query)
            questions = get_text_content(response)

        st.success(" Sorular Hazır!")
        st.write(questions)
        st.caption(f" RAG kullanıldı: Toplam {len(chunks)} chunk tarandı, en alakalı 4 chunk seçildi")

        # İndirme butonu
        st.download_button(
            label=" Soruları İndir (TXT)",
            data=questions,
            file_name="pdf_sorular.txt",
            mime="text/plain"
        )
# -------------------------------
#  Kullanıcı Soru Sorma (RAG Chat)
# -------------------------------
st.subheader(" PDF'e Soru Sor")
user_question = st.text_input("Sorunu buraya yaz:")

if user_question:
    with st.spinner("Yanıt aranıyor..."):
        response = qa_chain.run(user_question)
        answer = get_text_content(response)
    st.write(" **Cevap:**", answer)
    st.caption(f" RAG kullanıldı: En alakalı 4 chunk'tan cevap üretildi")
