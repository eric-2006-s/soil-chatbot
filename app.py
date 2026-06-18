import streamlit as st
import pandas as pd
from scipy.spatial import cKDTree
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from deep_translator import GoogleTranslator
import tempfile


# ==========================
# Groq Setup
# ==========================
groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# ==========================
# Load Data (Cached)
# ==========================
@st.cache_data
def load_data():
    df = pd.read_excel("combined.xlsx")
    oc_df = pd.read_excel("soc attribute.xlsx")
    tree = cKDTree(oc_df[["latitude", "longitude"]].values)
    distances, indices = tree.query(df[["latitude", "longitude"]].values, k=1)
    df["Organic_Carbon"] = oc_df.iloc[indices]["RASTERVALU"].values
    return df

@st.cache_resource
def get_soil_tree(_df):
    return cKDTree(_df[["latitude", "longitude"]].values)

df = load_data()
soil_tree = get_soil_tree(df)

# ==========================
# Crop Map
# ==========================
CROP_MAP = {
    "grape":       "Grape_Leg",
    "ragi":        "Ragi_Leg",
    "maize":       "Maize_Leg",
    "mango":       "Mango_Leg",
    "coconut":     "Cocon_Leg",
    "guava":       "Guava_Leg",
    "banana":      "Banana_Leg",
    "onion":       "Onion_Leg",
    "groundnut":   "Gnut_Leg",
    "cotton":      "Cot_Leg",
    "sorghum":     "Sorg_Leg",
    "tomato":      "Tom_Leg",
    "papaya":      "Papaya_Leg",
    "cowpea":      "Cowpea_Leg",
    "orange":      "Orange_Leg",
    "coriander":   "Coriander_Leg",
    "sapota":      "Sapota_Leg",
    "sunflower":   "Sunf_Leg",
    "bajra":       "Bajra_Leg",
    "brinjal":     "Brinjal_Leg",
    "paddy":       "UPaddy_Leg",
    "bengal gram": "Beng_Leg",
    "red gram":    "RG_Leg",
    "horse gram":  "HG_Leg",
    "black gram":  "Blkgrm_Leg",
    "chilli":      "Chil_Leg",
    "pomegranate": "Pome_Leg",
    "bhendi":      "Bhendi_Leg",
    "ginger":      "GG_Leg",
    "garlic":      "GG_Leg",
    "sesame":      "Lseed_Leg",
    "linseed":     "Lseed_Leg",
}

# ==========================
# Field Map
# ==========================
FIELD_MAP = {
    "organic carbon": "Organic_Carbon",
    "oc":             "Organic_Carbon",
    "soil type":      "Soil_Type",
    "depth":          "Depth_Leg",
    "texture":        "Text_Leg",
    "slope":          "Slope_Leg",
    "gravel":         "Gravel_Leg",
    "erosion":        "Eros_Leg",
    "awc":            "AWC",
    "taluk":          "Taluk",
    "village":        "Village",
    "district":       "KGISDistrictName",
    "latitude":       "latitude",
    "longitude":      "longitude",
    "soil series":    "SoilSeries",
    "soil phase":     "Soil_Phase",
}

COMPARE_FIELDS = [
    "Soil_Type", "Depth_Leg", "Text_Leg", "Slope_Leg",
    "Gravel_Leg", "Eros_Leg", "AWC", "Organic_Carbon",
    "SoilSeries", "Soil_Phase", "Taluk", "Village"
]

# ==========================
# Suitability Explanation
# ==========================
def explain_suitability(code):
    if pd.isna(code):
        return "No Data Available"
    code = str(code)
    suitability = {
        "S1": "Highly Suitable", "S2": "Moderately Suitable",
        "S3": "Marginally Suitable", "N1": "Currently Not Suitable",
        "N2": "Permanently Not Suitable", "N": "Not Suitable",
    }
    limitations = {
        "g": "Gravelliness/Stoniness", "l": "Topography",
        "r": "Rooting Condition",      "t": "Texture",
        "w": "Drainage",               "n": "Nutrient Availability",
        "e": "Erosion",                "z": "Excess Salt/Calcareousness",
    }
    if code.startswith("S1"):   base = suitability["S1"]
    elif code.startswith("S2"): base = suitability["S2"]
    elif code.startswith("S3"): base = suitability["S3"]
    elif code.startswith("N2"): base = suitability["N2"]
    elif code.startswith("N1"): base = suitability["N1"]
    else:                        base = suitability["N"]
    lims = [limitations[ch] for ch in code if ch in limitations]
    if lims:
        return f"{base}\n\nLimitations: {', '.join(lims)}"
    return base

# ==========================
# Crop Recommendation
# ==========================
def get_suitable_crops(record):
    highly, moderate, marginal = [], [], []
    for crop, col in CROP_MAP.items():
        if col in record.index:
            value = str(record[col])
            if value.startswith("S1"):   highly.append(crop.title())
            elif value.startswith("S2"): moderate.append(crop.title())
            elif value.startswith("S3"): marginal.append(crop.title())
    response = "## Suitable Crops\n\n"
    if highly:   response += f"### Highly Suitable (S1)\n{', '.join(highly)}\n\n"
    if moderate: response += f"### Moderately Suitable (S2)\n{', '.join(moderate)}\n\n"
    if marginal: response += f"### Marginally Suitable (S3)\n{', '.join(marginal)}"
    return response

# ==========================
# WHERE: Soil Type
# ==========================
def where_soil_type(query_lower):
    col = "Soil_Type" if "soil type" in query_lower else "SoilSeries"
    soil_values = df[col].dropna().unique()
    matched_soil = next((v for v in soil_values if str(v).lower() in query_lower), None)
    if not matched_soil:
        return f"Please mention a specific soil type.\n\n**Available:** {', '.join(sorted(map(str, soil_values)))}", None
    result_df = df[df[col].astype(str) == str(matched_soil)]
    map_df = result_df[["latitude", "longitude"]].dropna()
    response  = f"### {col.replace('_', ' ')}: **{matched_soil}**\n\n"
    response += f"**Found in:** {len(result_df)} plots\n\n"
    response += f"**Taluks:** {', '.join(result_df['Taluk'].dropna().unique())}\n\n"
    response += f"**Villages:** {', '.join(result_df['Village'].dropna().unique())}\n\n"
    return response, map_df if not map_df.empty else None

# ==========================
# WHERE: Crop
# ==========================
def where_grow_crop(query_lower):
    matched_crop = next((c for c in CROP_MAP if c in query_lower), None)
    if not matched_crop:
        return f"Please mention a specific crop.\n\n**Supported:** {', '.join(sorted(CROP_MAP.keys()))}", None
    col = CROP_MAP[matched_crop]
    if col not in df.columns:
        return f"Column `{col}` not found.", None
    s1_df = df[df[col].astype(str).str.startswith("S1")]
    s2_df = df[df[col].astype(str).str.startswith("S2")]
    s3_df = df[df[col].astype(str).str.startswith("S3")]
    response  = f"### Where to Grow **{matched_crop.title()}**\n\n"
    response += f"**S1 (Highly Suitable):** {len(s1_df)} plots\n"
    if not s1_df.empty: response += f"Villages: {', '.join(s1_df['Village'].dropna().unique())}\n\n"
    response += f"**S2 (Moderately Suitable):** {len(s2_df)} plots\n"
    if not s2_df.empty: response += f"Villages: {', '.join(s2_df['Village'].dropna().unique())}\n\n"
    response += f"**S3 (Marginally Suitable):** {len(s3_df)} plots\n"
    if not s3_df.empty: response += f"Villages: {', '.join(s3_df['Village'].dropna().unique())}\n\n"
    map_df = s1_df[["latitude", "longitude"]].dropna()
    return response, map_df if not map_df.empty else None

# ==========================
# Crop Suitability Heatmap
# ==========================
@st.cache_data
def soil_suitability_heatmap(crop_name):
    col = CROP_MAP.get(crop_name)
    if not col or col not in df.columns:
        return None, f"Crop `{crop_name}` not found."
    weight_map = {"S1": 1.0, "S2": 0.6, "S3": 0.3}
    heat_df = df[["latitude", "longitude", col]].dropna().copy()
    heat_df["weight"] = heat_df[col].astype(str).apply(
        lambda x: next((v for k, v in weight_map.items() if x.startswith(k)), 0)
    )
    heat_df = heat_df[heat_df["weight"] > 0]
    if heat_df.empty:
        return None, f"No suitable plots found for {crop_name.title()}."
    center = [heat_df["latitude"].mean(), heat_df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
    HeatMap(
        heat_df[["latitude", "longitude", "weight"]].values.tolist(),
        min_opacity=0.3,
        radius=20,
        blur=15,
        gradient={"0.3": "#ffffb2", "0.6": "#fd8d3c", "1.0": "#bd0026"}
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m, None

# ==========================
# Organic Carbon Heatmap
# ==========================
@st.cache_data
def organic_carbon_heatmap():
    oc_df = df[["latitude", "longitude", "Organic_Carbon"]].dropna().copy()
    oc_df["Organic_Carbon"] = pd.to_numeric(oc_df["Organic_Carbon"], errors="coerce")
    oc_df = oc_df.dropna(subset=["Organic_Carbon"])

    if oc_df.empty:
        return None, "No numeric organic carbon data available."

    oc_min = oc_df["Organic_Carbon"].min()
    oc_max = oc_df["Organic_Carbon"].max()
    oc_df["weight"] = (oc_df["Organic_Carbon"] - oc_min) / (oc_max - oc_min + 1e-9)

    center = [oc_df["latitude"].mean(), oc_df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
    HeatMap(
        oc_df[["latitude", "longitude", "weight"]].values.tolist(),
        min_opacity=0.3,
        radius=20,
        blur=15,
        gradient={"0.3": "#ffffb2", "0.6": "#fd8d3c", "1.0": "#bd0026"}
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m, None

# ==========================
# PDF Report
# ==========================
def generate_pdf(record):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFillColorRGB(0.1, 0.4, 0.1)
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 40, "Information Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 62, f"Survey Number: {record['SurNo_Hissa']}   |   Village: {record['Village']}   |   District: {record['KGISDistrictName']}")

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 110, "Soil Parameters")
    c.setLineWidth(1)
    c.setStrokeColorRGB(0.1, 0.4, 0.1)
    c.line(50, height - 115, 550, height - 115)

    fields = [
        ("District",        record["KGISDistrictName"]),
        ("Taluk",           record["Taluk"]),
        ("Village",         record["Village"]),
        ("Survey Number",   record["SurNo_Hissa"]),
        ("Soil Type",       record["Soil_Type"]),
        ("Soil Series",     record["SoilSeries"]),
        ("Soil Phase",      record["Soil_Phase"]),
        ("Depth",           record["Depth_Leg"]),
        ("Texture",         record["Text_Leg"]),
        ("Slope",           record["Slope_Leg"]),
        ("Gravel",          record["Gravel_Leg"]),
        ("Erosion",         record["Eros_Leg"]),
        ("AWC",             record["AWC"]),
        ("Organic Carbon",  record["Organic_Carbon"]),
    ]

    y = height - 135
    for i, (label, value) in enumerate(fields):
        if i % 2 == 0:
            c.setFillColorRGB(0.95, 0.95, 0.95)
            c.rect(50, y - 4, 500, 18, fill=True, stroke=False)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(60, y, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(220, y, str(value))
        y -= 20

    y -= 15
    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(50, y, "Crop Suitability")
    c.setStrokeColorRGB(0.1, 0.4, 0.1)
    c.line(50, y - 5, 550, y - 5)
    y -= 25

    suitability_colors = {
        "S1": (0.1, 0.6, 0.1),
        "S2": (0.8, 0.6, 0.0),
        "S3": (0.8, 0.4, 0.0),
        "N":  (0.7, 0.0, 0.0),
    }

    for crop, col in CROP_MAP.items():
        if col in record.index:
            code = str(record[col])
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(60, y, f"{crop.title()}:")
            color = next((v for k, v in suitability_colors.items() if code.startswith(k)), (0, 0, 0))
            c.setFillColorRGB(*color)
            explanation = explain_suitability(record[col]).split("\n\n")[0]
            c.drawString(220, y, f"{code} - {explanation}")
            y -= 15
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFillColorRGB(0, 0, 0)

    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.setFont("Helvetica", 8)
    c.drawString(50, 30, "Generated by GIS Soil Information Chatbot | NBSS&LUP")

    c.save()
    buffer.seek(0)
    return buffer

# ==========================
# UI
# ==========================
st.set_page_config(page_title="SoilMitra AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #1a3a1a !important;
}

.stApp {
    background-color: #f5f7f0 !important;
}

p, div, span, label, li, td, th, a {
    color: #1a3a1a !important;
}

h1 {
    color: #1a4d1a !important;
    font-weight: 700 !important;
    font-size: 1.9rem !important;
    border-bottom: 3px solid #4caf50;
    padding-bottom: 10px;
    margin-bottom: 20px !important;
}

h2 { color: #1a4d1a !important; font-weight: 700 !important; }
h3 { color: #2e6b2e !important; font-weight: 600 !important; }

section[data-testid="stSidebar"] {
    background-color: #2d5a2d !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #d4edda !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #90ee90 !important;
    border-bottom: 1px solid #4caf50;
    padding-bottom: 4px;
}

.stRadio > label > div > p {
    color: #1a3a1a !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
div[role="radiogroup"] label {
    color: #1a3a1a !important;
    font-weight: 500 !important;
}

.stSelectbox label p,
.stSelectbox label {
    color: #1a3a1a !important;
    font-weight: 600 !important;
}

.stSelectbox > div > div {
    border: 1.5px solid #a5d6a7 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
}

.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div {
    color: #1a3a1a !important;
    font-weight: 700 !important;
}

.stTextInput label p,
.stTextInput label,
.stNumberInput label p,
.stNumberInput label {
    color: #1a3a1a !important;
    font-weight: 600 !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border: 1.5px solid #a5d6a7 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
    color: #1a3a1a !important;
    font-weight: 700 !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 2px rgba(76,175,80,0.2) !important;
}

.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: background-color 0.2s ease;
}

.stButton > button:hover {
    background-color: #1b5e20 !important;
    color: white !important;
}

.stButton > button p { color: white !important; }

.stDownloadButton > button {
    background-color: #388e3c !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}

.stDownloadButton > button:hover {
    background-color: #1b5e20 !important;
    color: white !important;
}

.stDownloadButton > button p { color: white !important; }

[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    border: 1px solid #c8e6c9 !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] span {
    color: #1a3a1a !important;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown span,
.stMarkdown div {
    color: #1a3a1a !important;
}

.stSuccess, div[data-testid="stNotification"] {
    background-color: #e8f5e9 !important;
    border-left: 4px solid #4caf50 !important;
    border-radius: 6px !important;
    color: #1a3a1a !important;
}

[data-testid="stText"] {
    color: #1a3a1a !important;
    font-weight: 500 !important;
}

.stDataFrame {
    border: 1px solid #c8e6c9 !important;
    border-radius: 8px !important;
    overflow: hidden;
}

.streamlit-expander {
    border: 1px solid #c8e6c9 !important;
    border-radius: 8px !important;
    background: white !important;
}

details summary p {
    color: #1a3a1a !important;
    font-weight: 600 !important;
}

hr { border-color: #c8e6c9 !important; }

[data-testid="stSubheader"] { color: #1a4d1a !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌱 SoilMitra AI – South Indian Soil Intelligence and Advisory System")
st.sidebar.header("Dataset Information")
st.sidebar.write(f"Total Records: {len(df)}")

search_mode = st.radio("Select Search Method", ["Survey Number", "Latitude & Longitude"])

# ==========================
# Search Mode
# ==========================
if search_mode == "Survey Number":
    selected_district = st.selectbox("Select District", sorted(df["KGISDistrictName"].dropna().astype(str).unique()))
    district_df = df[df["KGISDistrictName"].astype(str) == selected_district]
    selected_village = st.selectbox("Select Village", sorted(district_df["Village"].dropna().astype(str).unique()))
    village_df = district_df[district_df["Village"].astype(str) == selected_village]
    selected_survey = st.selectbox("Select Survey Number (SurNo_Hissa)", sorted(village_df["SurNo_Hissa"].dropna().astype(str).unique()))
    record = village_df[village_df["SurNo_Hissa"].astype(str) == selected_survey].iloc[0]
else:
    st.subheader("Search Using Coordinates")
    input_lat = st.number_input("Latitude", format="%.6f")
    input_lon = st.number_input("Longitude", format="%.6f")
    distance, index = soil_tree.query([[input_lat, input_lon]], k=1)
    record = df.iloc[index[0]]
    st.write("### Matched Location")
    st.write(f"District: {record['KGISDistrictName']}")
    st.write(f"Village: {record['Village']}")
    st.write(f"Survey Number: {record['SurNo_Hissa']}")
    st.success(f"Nearest Survey: {record['SurNo_Hissa']} (Distance = {distance[0]:.6f})")

# ==========================
# Dynamic Zoom Map
# ==========================
if "latitude" in record.index and "longitude" in record.index and pd.notna(record["latitude"]) and pd.notna(record["longitude"]):
    st.subheader("📍 Parcel Location")

    if search_mode == "Survey Number":
        district_center = df[df["KGISDistrictName"].astype(str) == selected_district][["latitude","longitude"]].mean()
        village_center  = village_df[["latitude","longitude"]].mean()
        survey_lat      = record["latitude"]
        survey_lon      = record["longitude"]

        m = folium.Map(location=[survey_lat, survey_lon], zoom_start=15, tiles="CartoDB positron")

        folium.Marker(
            location=[district_center["latitude"], district_center["longitude"]],
            tooltip=selected_district,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

        folium.Marker(
            location=[village_center["latitude"], village_center["longitude"]],
            tooltip=selected_village,
            icon=folium.Icon(color="orange", icon="home")
        ).add_to(m)

        folium.Marker(
            location=[survey_lat, survey_lon],
            tooltip=f"Survey: {record['SurNo_Hissa']}",
            icon=folium.Icon(color="green", icon="leaf")
        ).add_to(m)

    else:
        m = folium.Map(location=[record["latitude"], record["longitude"]], zoom_start=15, tiles="CartoDB positron")
        folium.Marker(
            location=[record["latitude"], record["longitude"]],
            tooltip=f"Survey: {record['SurNo_Hissa']}",
            icon=folium.Icon(color="green", icon="leaf")
        ).add_to(m)

    st_folium(m, width="100%", height=450, returned_objects=[])

# ==========================
# PDF Download
# ==========================
st.subheader("📄 Download Soil Report")
pdf_buffer = generate_pdf(record)
st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf_buffer,
    file_name=f"soil_report_{record['SurNo_Hissa']}.pdf",
    mime="application/pdf"
)

# ==========================
# Compare Two Plots
# ==========================
st.sidebar.markdown("---")
st.sidebar.subheader("🔀 Compare Two Plots")
all_surveys = sorted(df["SurNo_Hissa"].dropna().astype(str).unique())
compare_a = st.sidebar.selectbox("Plot A", all_surveys, key="compare_a")
compare_b = st.sidebar.selectbox("Plot B", all_surveys, key="compare_b")

if st.sidebar.button("Compare"):
    rec_a = df[df["SurNo_Hissa"].astype(str) == compare_a].iloc[0]
    rec_b = df[df["SurNo_Hissa"].astype(str) == compare_b].iloc[0]

    st.subheader(f"📊 Comparison: {compare_a} vs {compare_b}")

    rows = []
    for col in COMPARE_FIELDS:
        if col in df.columns:
            rows.append({"Field": col.replace("_", " "), compare_a: rec_a[col], compare_b: rec_b[col]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("#### Crop Suitability Comparison")
    crop_rows = []
    for crop, col in CROP_MAP.items():
        if col in df.columns:
            crop_rows.append({
                "Crop": crop.title(),
                compare_a: str(rec_a[col]) if col in rec_a.index else "N/A",
                compare_b: str(rec_b[col]) if col in rec_b.index else "N/A",
            })
    st.dataframe(pd.DataFrame(crop_rows), use_container_width=True)

    map_points = pd.DataFrame({
        "lat": [rec_a["latitude"], rec_b["latitude"]],
        "lon": [rec_a["longitude"], rec_b["longitude"]],
    }).dropna()
    if not map_points.empty:
        st.subheader("🗺️ Plot Locations")
        st.map(map_points)


# ==========================
# Voice Helpers
# ==========================
def speak_text(text):
    try:
        tts = gTTS(text=str(text))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception:
        return None

voice_query = None
st.subheader("🎤 Voice input")
audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True
)

if audio:
    try:
        st.audio(audio["bytes"])
        with open("voice.wav", "wb") as f:
            f.write(audio["bytes"])

        transcription = groq_client.audio.transcriptions.create(
            file=open("voice.wav", "rb"),
            model="whisper-large-v3"
        )

        voice_query = transcription.text

        try:
            voice_query = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(voice_query)
        except Exception:
            pass

        st.success(f"You said: {voice_query}")

    except Exception as e:
        st.error(f"Voice recognition error: {e}")


# ==========================
# Query Input
# ==========================
text_query = st.text_input("💬 Ask a question about this soil plot")
query = voice_query if voice_query else text_query

# ==========================
# Chatbot Logic
# ==========================
if query:
    query_lower = query.lower()
    st.chat_message("user").write(query)
    response = None
    map_data = None

    # ── 1. WHERE queries ──────────────────────────────────────────────────
    if "where" in query_lower:
        if "soil type" in query_lower or "soil series" in query_lower:
            response, map_data = where_soil_type(query_lower)
        elif any(w in query_lower for w in ["grow", "cultivate", "plant", "suitable for"]):
            response, map_data = where_grow_crop(query_lower)
        elif "village" in query_lower or "taluk" in query_lower or "district" in query_lower:
            response = f"**District:** {record['KGISDistrictName']}\n\n**Taluk:** {record['Taluk']}\n\n**Village:** {record['Village']}"
        else:
            response = "You can ask:\n- *Where is [soil type] found?*\n- *Where can I grow [crop]?*\n- *Where is this village?*"

    # ── 2. Complete profile / summary ─────────────────────────────────────
    elif any(w in query_lower for w in ["complete", "profile", "full", "all details"]):
        response = f"## Complete Profile: {record['SurNo_Hissa']}\n\n"
        for col in df.columns:
            response += f"**{col}:** {record[col]}\n\n"

    elif "summary" in query_lower:
        response = (
            f"## Soil Summary\n\n"
            f"**District:** {record['KGISDistrictName']}\n\n"
            f"**Village:** {record['Village']}\n\n"
            f"**Survey Number:** {record['SurNo_Hissa']}\n\n"
            f"**Taluk:** {record['Taluk']}\n\n"
            f"**Soil Type:** {record['Soil_Type']}\n\n"
            f"**Depth:** {record['Depth_Leg']}\n\n"
            f"**Texture:** {record['Text_Leg']}\n\n"
            f"**Organic Carbon:** {record['Organic_Carbon']}\n\n"
            f"**AWC:** {record['AWC']}"
        )

    # ── 3. Crop recommendations ────────────────────────────────────────────
    elif any(w in query_lower for w in ["what crops", "which crops", "suitable crops", "recommended crops", "suitability", "crop suitability", "all crops"]):
        response = get_suitable_crops(record)

    # ── 4. Heatmap ────────────────────────────────────────────────────────
    elif "heatmap" in query_lower:
        # Organic Carbon heatmap
        if "organic carbon" in query_lower or " oc " in query_lower or query_lower.strip() in ["heatmap oc", "oc heatmap", "organic carbon heatmap", "heatmap organic carbon"]:
            heatmap_obj, err = organic_carbon_heatmap()
            if heatmap_obj:
                st.chat_message("assistant").write(
                    "### 🌡️ Organic Carbon Heatmap\n\n"
                    "Red = High OC · Orange = Medium · Yellow = Low\n\n"
                    f"Range across dataset: {df['Organic_Carbon'].min():.3f} – {df['Organic_Carbon'].max():.3f}"
                )
                st_folium(heatmap_obj, width="100%", height=500, returned_objects=[])
                response = "__rendered__"
            else:
                response = err

        # Crop suitability heatmap
        else:
            matched_crop = next((c for c in CROP_MAP if c in query_lower), None)
            if matched_crop:
                heatmap_obj, err = soil_suitability_heatmap(matched_crop)
                if heatmap_obj:
                    st.chat_message("assistant").write(
                        f"### 🌡️ Suitability Heatmap: {matched_crop.title()}\n\n"
                        "Red = Highly Suitable (S1) · Orange = Moderate (S2) · Yellow = Marginal (S3)"
                    )
                    st_folium(heatmap_obj, width="100%", height=500, returned_objects=[])
                    response = "__rendered__"
                else:
                    response = err
            else:
                response = (
                    f"Mention a crop name or 'organic carbon'.\n\n"
                    f"**Examples:**\n"
                    f"- *heatmap for maize*\n"
                    f"- *organic carbon heatmap*\n\n"
                    f"**Supported crops:** {', '.join(sorted(CROP_MAP.keys()))}"
                )

    # ── 5. Individual crop suitability ────────────────────────────────────
    else:
        for crop, column in CROP_MAP.items():
            if crop in query_lower and column in record.index:
                code = record[column]
                response = f"### {crop.title()} Suitability\n\nCode: **{code}**\n\n{explain_suitability(code)}"
                break

    # ── 6. Soil parameter ─────────────────────────────────────────────────
    if response is None:
        for keyword, column in FIELD_MAP.items():
            if keyword in query_lower and column in record.index:
                response = f"### {keyword.title()}\n\n{record[column]}"
                break

    # ── 7. Groq fallback ──────────────────────────────────────────────────
    if response is None:
        try:
            soil_context = f"""You are a soil and agriculture expert. Answer based on this data:
District: {record['KGISDistrictName']}, Village: {record['Village']}, Survey: {record['SurNo_Hissa']}
Taluk: {record['Taluk']}, Soil Type: {record['Soil_Type']}, Depth: {record['Depth_Leg']}
Texture: {record['Text_Leg']}, Slope: {record['Slope_Leg']}, Gravel: {record['Gravel_Leg']}
Erosion: {record['Eros_Leg']}, Organic Carbon: {record['Organic_Carbon']}, AWC: {record['AWC']}
Soil Series: {record['SoilSeries']}, Soil Phase: {record['Soil_Phase']}

Question: {query}
Give a concise, helpful answer."""
            groq_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": soil_context}],
                max_tokens=500
            )
            response = f"🤖 **Groq (LLaMA):**\n\n{groq_response.choices[0].message.content}"
        except Exception as e:
            response = (
                f"## Soil Summary\n\n"
                f"**District:** {record['KGISDistrictName']}\n\n"
                f"**Village:** {record['Village']}\n\n"
                f"**Survey Number:** {record['SurNo_Hissa']}\n\n"
                f"**Taluk:** {record['Taluk']}\n\n"
                f"**Soil Type:** {record['Soil_Type']}\n\n"
                f"**Depth:** {record['Depth_Leg']}\n\n"
                f"**Texture:** {record['Text_Leg']}\n\n"
                f"**Organic Carbon:** {record['Organic_Carbon']}\n\n"
                f"**AWC:** {record['AWC']}\n\n"
                f"*(Groq unavailable: {e})*"
            )

    # ── Render ────────────────────────────────────────────────────────────
    if response and response != "__rendered__":
        st.chat_message("assistant").write(response)
        
        audio_file = speak_text(str(response).replace("#","").replace("*",""))
        if audio_file:
            st.audio(audio_file)

    if map_data is not None:
        st.subheader("🗺️ Matching Locations")
        st.map(map_data.rename(columns={"latitude": "lat", "longitude": "lon"}))

# ==========================
# Dataset Viewer
# ==========================
with st.expander("📂 View Dataset"):
    st.dataframe(df)