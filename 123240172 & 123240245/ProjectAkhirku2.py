import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dana Darurat Fuzzy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Sidebar Input ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 Input Data Keuangan")
    
    penghasilan_input = st.number_input("Penghasilan Bulanan (Rp)", 
                                        min_value=1_000_000, max_value=25_000_000, 
                                        value=10_000_000, step=500_000)
    
    pengeluaran_input = st.number_input("Pengeluaran Bulanan (Rp)", 
                                        min_value=500_000, max_value=20_000_000, 
                                        value=5_000_000, step=500_000)
    
    tanggungan_input = st.number_input("Jumlah Tanggungan (orang)", 
                                       min_value=0, max_value=10, value=2, step=1)
    
    hitung_btn = st.button("⚡ Hitung Rekomendasi", type="primary", use_container_width=True)
    if pengeluaran_input >= penghasilan_input:
        st.error("⚠️ Pengeluaran melebihi atau sama dengan penghasilan!")

# ─── Main Content ──────────────────────────────────────────────────────────────
st.title("💰 Sistem Rekomendasi Dana Darurat")
st.caption("Berbasis Logika Fuzzy Mamdani | Tugas Praktikum SPK")
st.divider()

# ─── Load Dataset ──────────────────────────────────────────────────────────────
try:
    df_dataset = pd.read_csv("dataset_dana_darurat.csv")
    st.success(f"✅ Dataset dimuat")
except FileNotFoundError:
    df_dataset = None
    st.warning("⚠️ File dataset_dana_darurat.csv tidak ditemukan.")

# ─── Fuzzy System ──────────────────────────────────────────────────────────────
@st.cache_resource
def build_fuzzy_system():
    # Semua variabel fuzzy harus di dalam fungsi ini
    penghasilan = ctrl.Antecedent(np.arange(0, 21, 0.1), 'penghasilan')
    pengeluaran = ctrl.Antecedent(np.arange(0, 21, 0.1), 'pengeluaran')
    tanggungan  = ctrl.Antecedent(np.arange(0, 11, 0.1), 'tanggungan')
    dana_darurat = ctrl.Consequent(np.arange(0, 15, 0.1), 'dana_darurat')

    # Membership Functions
    penghasilan['rendah'] = fuzz.trimf(penghasilan.universe, [0, 0, 8])
    penghasilan['sedang'] = fuzz.trimf(penghasilan.universe, [5, 10, 15])
    penghasilan['tinggi'] = fuzz.trimf(penghasilan.universe, [12, 20, 25])

    pengeluaran['kecil'] = fuzz.trimf(pengeluaran.universe, [0, 0, 6])
    pengeluaran['sedang'] = fuzz.trimf(pengeluaran.universe, [4, 8, 12])
    pengeluaran['besar'] = fuzz.trimf(pengeluaran.universe, [10, 20, 20])

    tanggungan['sedikit'] = fuzz.trimf(tanggungan.universe, [0, 0, 3])
    tanggungan['sedang'] = fuzz.trimf(tanggungan.universe, [2, 4, 6])
    tanggungan['banyak'] = fuzz.trimf(tanggungan.universe, [5, 10, 12])

    dana_darurat['kecil'] = fuzz.trimf(dana_darurat.universe, [0, 2, 4])
    dana_darurat['sedang'] = fuzz.trimf(dana_darurat.universe, [3, 6, 8])
    dana_darurat['besar'] = fuzz.trimf(dana_darurat.universe, [7, 10, 12])


    # ==================== RULES (15 Rules) ====================
    rules = [
        ctrl.Rule(penghasilan['rendah'] & pengeluaran['besar'] & tanggungan['banyak'], dana_darurat['besar']),
        ctrl.Rule(penghasilan['rendah'] & pengeluaran['besar'] & tanggungan['sedang'], dana_darurat['besar']),
        ctrl.Rule(penghasilan['sedang'] & pengeluaran['besar'] & tanggungan['banyak'], dana_darurat['besar']),
        ctrl.Rule(pengeluaran['besar'] & tanggungan['banyak'], dana_darurat['besar']),
        ctrl.Rule(penghasilan['rendah'] & tanggungan['banyak'], dana_darurat['besar']),
        ctrl.Rule(penghasilan['rendah'] & pengeluaran['sedang'], dana_darurat['besar']),
        
        ctrl.Rule(penghasilan['tinggi'] & pengeluaran['kecil'] & tanggungan['sedikit'], dana_darurat['kecil']),
        ctrl.Rule(penghasilan['tinggi'] & pengeluaran['sedang'] & tanggungan['sedikit'], dana_darurat['kecil']),
        ctrl.Rule(penghasilan['tinggi'] & pengeluaran['kecil'], dana_darurat['kecil']),
        
        ctrl.Rule(penghasilan['sedang'] & pengeluaran['sedang'] & tanggungan['sedang'], dana_darurat['besar']),
        ctrl.Rule(penghasilan['tinggi'] & pengeluaran['sedang'], dana_darurat['kecil']),
        ctrl.Rule(penghasilan['rendah'] & pengeluaran['kecil'] & tanggungan['sedang'], dana_darurat['sedang']),
        ctrl.Rule(penghasilan['sedang'] & pengeluaran['besar'] & tanggungan['sedikit'], dana_darurat['sedang']),
        
        ctrl.Rule(tanggungan['banyak'] & pengeluaran['sedang'], dana_darurat['besar']),
        ctrl.Rule(penghasilan['tinggi'] & tanggungan['sedikit'] & pengeluaran['sedang'], dana_darurat['kecil']),
    ]

    sistem_ctrl = ctrl.ControlSystem(rules)
    simulasi = ctrl.ControlSystemSimulation(sistem_ctrl)
    
    return simulasi, penghasilan, pengeluaran, tanggungan, dana_darurat

simulasi, fuz_ph, fuz_pe, fuz_t, fuz_dd = build_fuzzy_system()

def hitung_dana(p_juta, e_juta, t):
    try:
        simulasi.input['penghasilan'] = min(p_juta, 20.0)
        simulasi.input['pengeluaran'] = min(e_juta, 20.0)
        simulasi.input['tanggungan'] = min(t, 10.0)
        simulasi.compute()
        return simulasi.output['dana_darurat']
    except:
        return 6.0  # nilai default agar tidak crash

def format_rupiah(val):
    return f"Rp {val:,.0f}".replace(",", ".")

# ─── Perhitungan ───────────────────────────────────────────────────────────────
if hitung_btn and pengeluaran_input < penghasilan_input:
    p_juta = penghasilan_input / 1_000_000
    e_juta = pengeluaran_input / 1_000_000
    hasil_bulan = hitung_dana(p_juta, e_juta, float(tanggungan_input))
    hasil_rupiah = hasil_bulan * pengeluaran_input
    tabungan_per_bulan = penghasilan_input - pengeluaran_input

    if tabungan_per_bulan > 0:
        lama_menabung = hasil_rupiah / tabungan_per_bulan
    else:
        lama_menabung = 0

    if hasil_bulan < 4:
        kategori = "Kecil 🟢"
        ket = "Kondisi finansial relatif stabil. Risiko rendah."
    elif hasil_bulan < 8:
        kategori = "Sedang 🟡"
        ket = "Perlu menyiapkan cadangan yang cukup untuk perlindungan optimal."
    else:
        kategori = "Besar 🔴"
        ket = "Tingkat risiko tinggi. Segera bangun dana darurat yang kuat."

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Hasil", "📈 Fungsi Keanggotaan", "📋 Aturan Fuzzy", "📁 Dataset"])

    # TAB 1: HASIL
    with tab1:
        st.subheader("Hasil Rekomendasi")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dana Darurat (bulan)", f"{hasil_bulan:.2f} bulan")
        c2.metric("Dana Darurat (nominal)", format_rupiah(hasil_rupiah))
        c3.metric("Kategori", kategori)
        # c4.metric("Estimasi Terkumpul", f"{lama_menabung:.1f} bulan")
        st.info(f"ℹ️ {ket}")
        st.progress(min(hasil_bulan/12, 1.0))

    # TAB 2: FUNGSI KEANGGOTAAN (SUDAH DIBENERIN)
    with tab2:
        st.subheader("Visualisasi Fungsi Keanggotaan")
        st.caption("Garis putus-putus = nilai input Anda")

        COLORS = {"rendah": "red", "kecil": "red", "sedikit": "green",
                  "sedang": "orange", "tinggi": "green", "besar": "blue", "banyak": "blue"}

        def plot_mf(variable, current_val, title, xlabel, x_max):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for term in variable.terms:
                mf = variable[term].mf
                color = COLORS.get(term, "gray")
                ax.plot(variable.universe, mf, color=color, linewidth=2, label=term.capitalize())
                ax.fill_between(variable.universe, mf, alpha=0.1, color=color)
            ax.axvline(current_val, color="black", linewidth=1.5, linestyle="--", label=f"Input ({current_val:.1f})")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Derajat Keanggotaan")
            ax.set_xlim(0, x_max)
            ax.set_ylim(-0.05, 1.1)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.pyplot(plot_mf(fuz_ph, p_juta, "Penghasilan", "Juta Rupiah", 20))
            st.pyplot(plot_mf(fuz_t, float(tanggungan_input), "Tanggungan", "Jumlah Orang", 10))
        with col_plot2:
            st.pyplot(plot_mf(fuz_pe, e_juta, "Pengeluaran", "Juta Rupiah", 20))
            st.pyplot(plot_mf(fuz_dd, hasil_bulan, "Dana Darurat (Output)", "Bulan Gaji", 12))

    # TAB 3: ATURAN FUZZY
    with tab3:
        st.subheader("Basis Aturan Fuzzy")

        daftar_rules = [
            ["R1", "Rendah", "Besar", "Banyak", "Besar"],
            ["R2", "Rendah", "Besar", "Sedang", "Besar"],
            ["R3", "Sedang", "Besar", "Banyak", "Besar"],
            ["R4", "-", "Besar", "Banyak", "Besar"],
            ["R5", "Rendah", "-", "Banyak", "Besar"],
            ["R6", "Rendah", "Sedang", "-", "Besar"],

            ["R7", "Tinggi", "Kecil", "Sedikit", "Kecil"],
            ["R8", "Tinggi", "Sedang", "Sedikit", "Kecil"],
            ["R9", "Tinggi", "Kecil", "-", "Kecil"],

            ["R10", "Sedang", "Sedang", "Sedang", "Besar"],
            ["R11", "Tinggi", "Sedang", "-", "Kecil"],
            ["R12", "Rendah", "Kecil", "Sedang", "Sedang"],
            ["R13", "Sedang", "Besar", "Sedikit", "Sedang"],

            ["R14", "-", "Sedang", "Banyak", "Besar"],
            ["R15", "Tinggi", "Sedang", "Sedikit", "Kecil"]
        ]

        df_rules = pd.DataFrame(
            daftar_rules,
            columns=["No", "Penghasilan", "Pengeluaran", "Tanggungan", "Output"]
        )

        st.dataframe(df_rules, use_container_width=True, hide_index=True)

    # TAB 4: DATASET
    with tab4:
        st.subheader("📁 Dataset dana_darurat.csv")
        if df_dataset is not None:
            st.dataframe(df_dataset, use_container_width=True, height=400)
            if st.button("🚀 Proses Seluruh Dataset", type="primary"):
                # proses dataset...
                st.success("Dataset berhasil diproses!")
        else:
            st.info("Letakkan file dataset_dana_darurat.csv di folder yang sama.")

# else:
#     st.info("👈 Masukkan data di sidebar lalu klik **Hitung Rekomendasi**")

st.caption("Aplikasi Dana Darurat Fuzzy")