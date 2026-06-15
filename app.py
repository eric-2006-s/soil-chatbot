import streamlit as st
import pandas as pd
from scipy.spatial import cKDTree

# ==========================
# Load Data
# ==========================
df = pd.read_excel("Koppal_Union_Plot_Soil.xlsx")
df = df.drop_duplicates()

# ==========================
# Load Organic Carbon Data
# ==========================
oc_df = pd.read_excel("soc attribute.xlsx")

tree = cKDTree(oc_df[["latitude", "longitude"]].values)
distances, indices = tree.query(df[["latitude", "longitude"]].values, k=1)
df["Organic_Carbon"] = oc_df.iloc[indices]["RASTERVALU"].values

# ==========================
# Single source-of-truth crop map
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

# ==========================
# Suitability Explanation
# ==========================
def explain_suitability(code):
    if pd.isna(code):
        return "No Data Available"
    code = str(code)
    suitability = {
        "S1": "Highly Suitable",
        "S2": "Moderately Suitable",
        "S3": "Marginally Suitable",
        "N1": "Currently Not Suitable",
        "N2": "Permanently Not Suitable",
        "N":  "Not Suitable",
    }
    limitations = {
        "g": "Gravelliness/Stoniness",
        "l": "Topography",
        "r": "Rooting Condition",
        "t": "Texture",
        "w": "Drainage",
        "n": "Nutrient Availability",
        "e": "Erosion",
        "z": "Excess Salt/Calcareousness",
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
# Crop Recommendation (single record)
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
# WHERE: Soil Type across dataset
# ==========================
def where_soil_type(query_lower):
    col = "Soil_Type" if "soil type" in query_lower else "SoilSeries"
    soil_values = df[col].dropna().unique()
    matched_soil = next(
        (v for v in soil_values if str(v).lower() in query_lower), None
    )
    if not matched_soil:
        all_types = ", ".join(sorted(map(str, soil_values)))
        return (
            f"Please mention a specific soil type name.\n\n"
            f"**Available values:** {all_types}"
        ), None

    result_df = df[df[col].astype(str) == str(matched_soil)]
    villages  = result_df["Village"].dropna().unique()
    taluks    = result_df["Taluk"].dropna().unique()
    map_df    = result_df[["latitude", "longitude"]].dropna()

    response  = f"### {col.replace('_', ' ')}: **{matched_soil}**\n\n"
    response += f"**Found in:** {len(result_df)} plots\n\n"
    response += f"**Taluks:** {', '.join(map(str, taluks))}\n\n"
    response += f"**Villages:** {', '.join(map(str, villages))}\n\n"
    return response, map_df if not map_df.empty else None

# ==========================
# WHERE: Crop across dataset
# ==========================
def where_grow_crop(query_lower):
    matched_crop = next((c for c in CROP_MAP if c in query_lower), None)
    if not matched_crop:
        return (
            "Please mention a specific crop name.\n\n"
            f"**Supported crops:** {', '.join(sorted(CROP_MAP.keys()))}"
        ), None

    col = CROP_MAP[matched_crop]
    if col not in df.columns:
        return f"Column `{col}` not found in dataset.", None

    s1_df = df[df[col].astype(str).str.startswith("S1")]
    s2_df = df[df[col].astype(str).str.startswith("S2")]
    s3_df = df[df[col].astype(str).str.startswith("S3")]

    response  = f"### Where to Grow **{matched_crop.title()}**\n\n"
    response += f"**Highly Suitable (S1):** {len(s1_df)} plots\n"
    if not s1_df.empty:
        response += f"Villages: {', '.join(s1_df['Village'].dropna().unique())}\n\n"

    response += f"**Moderately Suitable (S2):** {len(s2_df)} plots\n"
    if not s2_df.empty:
        response += f"Villages: {', '.join(s2_df['Village'].dropna().unique())}\n\n"

    response += f"**Marginally Suitable (S3):** {len(s3_df)} plots\n"
    if not s3_df.empty:
        response += f"Villages: {', '.join(s3_df['Village'].dropna().unique())}\n\n"

    map_df = s1_df[["latitude", "longitude"]].dropna()
    return response, map_df if not map_df.empty else None

# ==========================
# UI
# ==========================
st.title("🌱 GIS Soil Information Chatbot")

st.sidebar.header("Dataset Information")
st.sidebar.write(f"Total Records: {len(df)}")

search_mode = st.radio(
    "Select Search Method",
    ["Survey Number", "Latitude & Longitude"]
)

# ==========================
# Search Mode
# ==========================
if search_mode == "Survey Number":
    selected_district = st.selectbox(
        "Select District",
        sorted(df["KGISDistrictName"].dropna().astype(str).unique())
    )
    district_df = df[df["KGISDistrictName"].astype(str) == selected_district]

    selected_village = st.selectbox(
        "Select Village",
        sorted(district_df["Village"].dropna().astype(str).unique())
    )
    village_df = district_df[district_df["Village"].astype(str) == selected_village]

    selected_survey = st.selectbox(
        "Select Survey Number (SurNo_Hissa)",
        sorted(village_df["SurNo_Hissa"].dropna().astype(str).unique())
    )
    record = village_df[village_df["SurNo_Hissa"].astype(str) == selected_survey].iloc[0]

else:
    st.subheader("Search Using Coordinates")
    input_lat = st.number_input("Latitude",  format="%.6f")
    input_lon = st.number_input("Longitude", format="%.6f")

    soil_tree = cKDTree(df[["latitude", "longitude"]].values)
    distance, index = soil_tree.query([[input_lat, input_lon]], k=1)
    record = df.iloc[index[0]]

    st.write("### Matched Location")
    st.write(f"District: {record['KGISDistrictName']}")
    st.write(f"Village: {record['Village']}")
    st.write(f"Survey Number: {record['SurNo_Hissa']}")
    st.success(f"Nearest Survey: {record['SurNo_Hissa']} (Distance = {distance[0]:.6f})")

# ==========================
# Parcel Map
# ==========================
if (
    "latitude" in record.index
    and "longitude" in record.index
    and pd.notna(record["latitude"])
    and pd.notna(record["longitude"])
):
    st.subheader("📍 Parcel Location")
    st.map(pd.DataFrame({"lat": [record["latitude"]], "lon": [record["longitude"]]}))

# ==========================
# Query Input
# ==========================
query = st.text_input("Ask a question")

# ==========================
# Chatbot Logic
# ==========================
if query:
    query_lower = query.lower()
    st.chat_message("user").write(query)
    response = None
    map_data = None

    # ── 1. WHERE queries (dataset-wide) ──────────────────────────────────
    if "where" in query_lower:

        # Where is this soil type found?
        if "soil type" in query_lower or "soil series" in query_lower:
            response, map_data = where_soil_type(query_lower)

        # Where can I grow X?
        elif any(w in query_lower for w in ["grow", "cultivate", "plant", "suitable for"]):
            response, map_data = where_grow_crop(query_lower)

        # Where is this village / taluk / district?
        elif "village" in query_lower or "taluk" in query_lower or "district" in query_lower:
            response = (
                f"**District:** {record['KGISDistrictName']}\n\n"
                f"**Taluk:** {record['Taluk']}\n\n"
                f"**Village:** {record['Village']}"
            )

        else:
            response = (
                "You can ask:\n"
                "- *Where is [soil type] found?*\n"
                "- *Where can I grow [crop]?*\n"
                "- *Where is this village?*"
            )

    # ── 2. Complete profile / summary ────────────────────────────────────
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

    # ── 3. Crop recommendations for selected record ───────────────────────
    elif any(w in query_lower for w in ["what crops", "which crops", "suitable crops", "recommended crops"]):
        response = get_suitable_crops(record)

    # ── 4. Individual crop suitability for selected record ────────────────
    else:
        for crop, column in CROP_MAP.items():
            if crop in query_lower and column in record.index:
                code = record[column]
                response = (
                    f"### {crop.title()} Suitability\n\n"
                    f"Code: **{code}**\n\n"
                    f"{explain_suitability(code)}"
                )
                break

    # ── 5. Soil parameter for selected record ─────────────────────────────
    if response is None:
        for keyword, column in FIELD_MAP.items():
            if keyword in query_lower and column in record.index:
                response = f"### {keyword.title()}\n\n{record[column]}"
                break

    # ── 6. Fallback ───────────────────────────────────────────────────────
    if response is None:
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

    # ── Render ────────────────────────────────────────────────────────────
    if response:
        st.chat_message("assistant").write(response)
    if map_data is not None:
        st.subheader("🗺️ Matching Locations")
        st.map(map_data)

# ==========================
# Dataset Viewer
# ==========================
with st.expander("View Dataset"):
    st.dataframe(df)