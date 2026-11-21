import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# 한글 폰트 CSS
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CSV 인코딩 처리 (utf-8-sig → cp949)
# ============================================================
def read_csv(file):
    """file.getvalue()로 바이트를 꺼내 인코딩 순차 시도"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ============================================================
# 생육 데이터 표준화 (수경/토양 공통)
#   잎 수(장)  → 잎 수
#   지상부 길이(mm)  → 지상부 길이
#   지하부길이(mm) / 지하부 길이(mm)  → 지하부 길이
#   생중량(g)  → 생중량
#   지상부질량(g) / 지상부 질량(g)  → 지상부 질량
#   지하수부질량(g) / 지하부 질량(g)  → 지하부 질량
#   생중량(g)이 없으면 지상부 질량 + 지하부 질량으로 계산
# ============================================================
def std_cols_growth(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


# ============================================================
# 수경재배 환경 데이터 표준화
#   time, air sensor(1)→temperature, air sensor(2)→humidity,
#   air sensor-co2→co2
# ============================================================
def std_cols_env_hydro(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif k == "co2":
            rename[c] = "co2"
        elif "air" in k and "sensor" in k and "co2" in k:
            rename[c] = "co2"
        elif "air" in k and "sensor" in k and "1" in k:
            rename[c] = "temperature"
        elif "air" in k and "sensor" in k and "2" in k:
            rename[c] = "humidity"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


# ============================================================
# 환경 데이터 매칭 파일에서 토양재배 환경 데이터 추출
#   강화중_time, 강화중_temperature, 강화중_humidity,
#   강화중_ph, 강화중_ec → time, temperature, humidity, ph, ec
# ============================================================
def extract_soil_env_from_match(match_df):
    df = match_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k == "강화중_time":
            col_map[c] = "time"
        elif k == "강화중_temperature":
            col_map[c] = "temperature"
        elif k == "강화중_humidity":
            col_map[c] = "humidity"
        elif k == "강화중_ph":
            col_map[c] = "ph"
        elif k == "강화중_ec":
            col_map[c] = "ec"
    if not col_map:
        return pd.DataFrame()
    df = df[list(col_map.keys())].copy()
    df = df.rename(columns=col_map)
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    for c in df.columns:
        if c == "time":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    return df


# ============================================================
# 사이드바
# ============================================================
st.sidebar.title("📂 데이터 업로드")

env_hydro_file = st.sidebar.file_uploader(
    "수경재배 환경 CSV",
    type="csv",
    help="예: 강화여고_환경데이터.csv",
)
growth_hydro_file = st.sidebar.file_uploader(
    "수경재배 생육 CSV",
    type="csv",
    help="예: 강화여고_생육데이터.csv",
)
match_file = st.sidebar.file_uploader(
    "환경 데이터 매칭 CSV",
    type="csv",
    help="예: 토양_수경_환경데이터_매칭.csv (토양재배 환경값 추출용)",
)
growth_soil_file = st.sidebar.file_uploader(
    "토양재배 생육 CSV",
    type="csv",
    help="예: 강화중_생육데이터.csv",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 생육 측정 항목 선택")
growth_cols_default = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_measures = st.sidebar.multiselect(
    "비교할 생육 항목 선택",
    options=growth_cols_default,
    default=growth_cols_default,
)

st.sidebar.markdown("---")
st.sidebar.caption("ℹ️ 각 CSV는 utf-8-sig / cp949 순서로 읽습니다.")
st.sidebar.caption(
    "ℹ️ 환경 데이터 매칭 CSV가 있으면 토양재배 환경값을 자동 추출해 "
    "수경재배와 함께 비교합니다."
)
st.sidebar.caption("ℹ️ 업로드 전에는 해당 탭 내용이 보이지 않습니다.")


# ============================================================
# 데이터 로딩 (@st.cache_data + spinner)
# ============================================================
@st.cache_data
def load_env_hydro(file):
    return std_cols_env_hydro(read_csv(file))


@st.cache_data
def load_growth_hydro(file):
    return std_cols_growth(read_csv(file))


@st.cache_data
def load_match(file):
    return read_csv(file)


@st.cache_data
def load_growth_soil(file):
    return std_cols_growth(read_csv(file))


env_hydro_df = pd.DataFrame()
growth_hydro_df = pd.DataFrame()
match_df = pd.DataFrame()
soil_env_df = pd.DataFrame()
growth_soil_df = pd.DataFrame()

if env_hydro_file is not None:
    with st.spinner("수경재배 환경 데이터를 읽는 중..."):
        try:
            env_hydro_df = load_env_hydro(env_hydro_file)
        except Exception:
            env_hydro_df = pd.DataFrame()

if growth_hydro_file is not None:
    with st.spinner("수경재배 생육 데이터를 읽는 중..."):
        try:
            growth_hydro_df = load_growth_hydro(growth_hydro_file)
        except Exception:
            growth_hydro_df = pd.DataFrame()

if match_file is not None:
    with st.spinner("환경 데이터 매칭 파일을 읽는 중..."):
        try:
            match_df = load_match(match_file)
            if len(match_df) > 0:
                soil_env_df = extract_soil_env_from_match(match_df)
        except Exception:
            match_df = pd.DataFrame()
            soil_env_df = pd.DataFrame()

if growth_soil_file is not None:
    with st.spinner("토양재배 생육 데이터를 읽는 중..."):
        try:
            growth_soil_df = load_growth_soil(growth_soil_file)
        except Exception:
            growth_soil_df = pd.DataFrame()

# 수경재배 생육 데이터에 생중량 열이 없으면 계산
if len(growth_hydro_df) > 0 and "생중량" not in growth_hydro_df.columns:
    if {"지상부 질량", "지하부 질량"} <= set(growth_hydro_df.columns):
        growth_hydro_df = growth_hydro_df.copy()
        growth_hydro_df["생중량"] = (
            growth_hydro_df["지상부 질량"] + growth_hydro_df["지하부 질량"]
        )

# 학교명 추출 (파일명 앞부분에서 읽기, 하드코딩 금지)
school_name = "우리 학교"
for f in [env_hydro_file, growth_hydro_file, match_file, growth_soil_file]:
    if f is not None:
        fn = os.path.basename(f.name)
        school_name = fn.split("_")[0] if "_" in fn else fn.split(".")[0]
        break


# ============================================================
# 편의성 변수
# ============================================================
has_env_hydro = len(env_hydro_df) > 0
has_growth_hydro = len(growth_hydro_df) > 0
has_match = len(match_df) > 0
has_soil_env = len(soil_env_df) > 0
has_growth_soil = len(growth_soil_df) > 0


# ============================================================
# 페이지 제목 및 통계 용어 설명
# ============================================================
st.title(
    "토양재배와 수경재배 환경이 극지식물의 생육 특성에 미치는 영향 탐구"
)
st.markdown(
    "*[평균]*: 모든 값을 더한 뒤 개수로 나눈 값  "
    "*[표준편차]*: 값이 평균 주변에 얼마나 모여 있는지 나타내는 값 (작을수록 비슷)  "
    "*[CV(변동계수)]* : 표준편차를 평균으로 나눈 값 (%) — 단위가 다른 항목도 상대적 퍼짐을 비교 가능  "
    "*[상관계수 r]* : 두 값이 함께 변하는 정도 (-1~1, 0.7 이상이면 강한 관계)  "
    "*[추세선]* : 산점도의 점들이 대체로 따라가는 직선 (np.polyfit으로 직접 계산)  "
    "*[T/R 비율]* : 지상부 길이 ÷ 지하부 길이 — 지상부/지하부 균형 지표",
)


tab1, tab2, tab3 = st.tabs(
    [
        "📖 실험 개요",
        "🌡️ 환경 데이터 비교 (수경 vs 토양)",
        "📊 생육 결과 비교 (수경 vs 토양)",
    ]
)


# ============================================================
# TAB 1 · 실험 개요
# ============================================================
with tab1:
    try:
        st.subheader("연구 배경 및 목적")
        st.markdown(
            "극지 환경은 기후변화에 민감하게 반응하는 지역 중 하나로, "
            "이곳에 서식하는 식물의 분포와 생육 시기 변화는 극지 생태계의 "
            "변화를 파악하는 중요한 자료가 된다. 또한 일부 극지식물은 낮은 "
            "온도와 짧은 생육기간 등 극한 환경에 적응한 생리적 특성을 지니고 "
            "있다. 이러한 식물의 생존 및 생육 메커니즘을 연구하면 저온 환경에서도 "
            "생육할 수 있는 작물의 특성을 이해하고 내한성 작물 개발과 같은 미래 "
            "농업 연구에 활용할 수 있다. 더불어 극지식물은 극지 생태계의 식생을 "
            "구성하며 토양 표면의 안정화와 유기물 순환 등 생태적 기능에도 기여한다. "
            "따라서 극지식물의 생육 특성을 이해하고 안정적으로 증식할 수 있는 재배 "
            "기술을 탐색하는 것은 생태계 보전과 생명자원 활용 측면에서 의미가 있다.",
        )
        st.markdown(
            "극지식물을 재배하는 방식에는 토양을 이용하는 방법과 양액을 이용하는 "
            "수경재배 방식 등이 있다. 토양재배는 토양의 물리적 구조와 수분 및 양분 "
            "상태 등 다양한 요인이 식물의 생육에 영향을 미치는 반면, 수경재배는 "
            "양액을 통해 물과 양분을 공급하므로 재배 환경을 상대적으로 세밀하게 "
            "관리할 수 있다는 특징이 있다. 두 방식은 식물의 뿌리가 접하는 환경과 "
            "수분 및 양분 공급 방식에서 차이가 있기 때문에 생육 양상에도 차이가 "
            "나타날 수 있다. 따라서 재배 방식별 환경 조건과 식물의 생육 특성을 함께 "
            "비교하여 극지식물에 적합한 재배 환경을 탐색할 필요가 있다.",
        )
        st.markdown(
            "식물의 생육은 재배 방식뿐 아니라 온도, 습도, CO₂ 등 주변 환경 요인의 "
            "영향을 받는다. 온도는 식물의 광합성과 호흡 등 대사 과정에 영향을 미치며, "
            "습도는 증산 작용과 식물의 수분 균형에 관여한다. CO₂는 광합성에 필요한 "
            "주요 기질로, 농도에 따라 광합성 속도가 달라질 수 있다. 이러한 환경 요인은 "
            "서로 독립적으로 작용하기보다 복합적으로 식물 생육에 영향을 미칠 수 있으므로, "
            "각각의 환경 데이터와 변수 간 상관관계를 함께 분석할 필요가 있다. 또한 잎의 "
            "개수와 크기, 지상부와 지하부의 길이, 생체중 등의 생육 지표를 분석하면 식물의 "
            "생장 양상과 생육 특성 간의 관계를 파악할 수 있다.",
        )
        st.markdown(
            "이에 본 연구에서는 **수경재배(스마트팜)**와 **토양재배(노지)** 환경에서 "
            "수집한 환경 데이터를 비교하고, 온도·습도·CO₂·pH·EC 등 환경 변수 간 차이를 "
            "분석한다. 또한 각 재배 방식에서 측정·관찰한 식물의 잎, 줄기, 뿌리 등의 "
            "생육 특성을 비교하고 생육 지표 간 상관관계를 분석하여 재배 방식에 따른 "
            "생육 양상의 차이를 살펴본다. 나아가 환경 데이터 분석 결과와 실제 생육 관찰 "
            "결과를 종합하여 극지식물의 효율적인 재배 환경을 탐색하고, 약용식물 생산이나 "
            "멸종위기 식물의 증식 및 보전 등 재배 목적에 따라 적합한 재배 방식이 무엇인지 "
            "고찰한다.",
        )

        st.subheader("가설 설정")
        st.markdown(
            "**가설 1.** 높은 기온의 환경에서 식물이 더 많은 잎 수를 가질 것이다.\n"
            "→ 온도 상승이 광합성 및 대사 활동을 촉진하여 잎 수 증가로 이어질 것으로 예상.",
        )
        st.markdown(
            "**가설 2.** 습도가 높은 환경에서는 지하부 길이가 짧아지고, "
            "습도가 낮은 환경에서는 지하부 길이가 길어질 것이다.\n"
            "→ 낮은 습도에서 수분 확보를 위해 뿌리 신장이 촉진될 것으로 예상.",
        )

        st.subheader("데이터 요약")
        rows = []
        if has_growth_hydro:
            rows.append(
                {
                    "구분": "수경재배 (스마트팜)",
                    "총 개체 수": len(growth_hydro_df),
                    "환경 데이터": "있음" if has_env_hydro else "업로드 필요",
                    "측정 기간": (
                        f"{env_hydro_df['time'].min().strftime('%Y-%m-%d')} ~ "
                        f"{env_hydro_df['time'].max().strftime('%Y-%m-%d')}"
                        if has_env_hydro and "time" in env_hydro_df.columns
                        else "자료 없음"
                    ),
                    "생육 측정 항목": ", ".join(
                        [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                         if c in growth_hydro_df.columns]
                    )
                    or "자료 없음",
                }
            )
        else:
            rows.append(
                {
                    "구분": "수경재배 (스마트팜)",
                    "총 개체 수": "CSV 업로드 필요",
                    "환경 데이터": "CSV 업로드 필요",
                    "측정 기간": "자료 없음",
                    "생육 측정 항목": "자료 없음",
                }
            )

        if has_growth_soil:
            soil_period = "자료 없음"
            if has_soil_env and "time" in soil_env_df.columns:
                soil_period = (
                    f"{soil_env_df['time'].min().strftime('%Y-%m-%d')} ~ "
                    f"{soil_env_df['time'].max().strftime('%Y-%m-%d')}"
                )
            rows.append(
                {
                    "구분": "토양재배 (노지)",
                    "총 개체 수": len(growth_soil_df),
                    "환경 데이터": "있음 (매칭 파일 추출)" if has_soil_env else "매칭 파일 업로드 필요",
                    "측정 기간": soil_period,
                    "생육 측정 항목": ", ".join(
                        [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                         if c in growth_soil_df.columns]
                    )
                    or "자료 없음",
                }
            )
        else:
            rows.append(
                {
                    "구분": "토양재배 (노지)",
                    "총 개체 수": "CSV 업로드 필요",
                    "환경 데이터": "매칭 파일 업로드 필요",
                    "측정 기간": "자료 없음",
                    "생육 측정 항목": "자료 없음",
                }
            )

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.subheader("주요 지표 카드")
        col1, col2, col3, col4 = st.columns(4)

        if has_growth_hydro and "생중량" in growth_hydro_df.columns:
            gw_h = growth_hydro_df["생중량"].dropna()
            if len(gw_h) >= 2:
                col1.metric("🌿 수경 평균 생중량 (g)", round(gw_h.mean(), 2))
            else:
                col1.metric("🌿 수경 평균 생중량 (g)", "자료 부족")
        else:
            col1.metric("🌿 수경 평균 생중량 (g)", "자료 없음")

        if has_growth_soil and "생중량" in growth_soil_df.columns:
            gw_s = growth_soil_df["생중량"].dropna()
            if len(gw_s) >= 2:
                col2.metric("🌱 토양 평균 생중량 (g)", round(gw_s.mean(), 2))
            else:
                col2.metric("🌱 토양 평균 생중량 (g)", "자료 부족")
        else:
            col2.metric("🌱 토양 평균 생중량 (g)", "자료 없음")

        if has_env_hydro and "temperature" in env_hydro_df.columns:
            t_h = env_hydro_df["temperature"].dropna()
            if len(t_h) >= 2:
                col3.metric("🌡️ 수경 평균 온도 (°C)", round(t_h.mean(), 1))
            else:
                col3.metric("🌡️ 수경 평균 온도 (°C)", "자료 부족")
        else:
            col3.metric("🌡️ 수경 평균 온도 (°C)", "자료 없음")

        if has_soil_env and "temperature" in soil_env_df.columns:
            t_s = soil_env_df["temperature"].dropna()
            if len(t_s) >= 2:
                col4.metric("🌡️ 토양 평균 온도 (°C)", round(t_s.mean(), 1))
            else:
                col4.metric("🌡️ 토양 평균 온도 (°C)", "자료 부족")
        else:
            col4.metric("🌡️ 토양 평균 온도 (°C)", "자료 없음")

        if not has_env_hydro and not has_soil_env:
            st.info(
                "환경 데이터(수경 환경 CSV + 환경 매칭 CSV)를 업로드하면 "
                "온도·습도 정보가 표시됩니다."
            )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# TAB 2 · 환경 데이터 비교 (수경 vs 토양)
# ============================================================
with tab2:
    try:
        if not has_env_hydro and not has_soil_env:
            st.info(
                "수경재배 환경 CSV + 환경 데이터 매칭 CSV를 업로드하면 "
                "수경 vs 토양 환경 비교가 표시됩니다."
            )
        else:
            # ---------- 수경재배 환경 시계열 ----------
            if has_env_hydro and "time" in env_hydro_df.columns:
                st.subheader("수경재배(스마트팜) 환경 시계열 (일별 평균)")
                env_h_daily = env_hydro_df.copy()
                env_h_daily["date"] = env_h_daily["time"].dt.date
                h_date_cols = [
                    c for c in ["temperature", "humidity", "co2"]
                    if c in env_h_daily.columns
                ]
                for dc in h_date_cols:
                    daily = (
                        env_h_daily.groupby("date")[dc]
                        .mean()
                        .dropna()
                        .reset_index()
                    )
                    if len(daily) >= 2:
                        unit = (
                            "°C" if dc == "temperature"
                            else "%" if dc == "humidity"
                            else "ppm"
                        )
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=daily["date"],
                                y=daily[dc],
                                mode="lines+markers",
                                line=dict(color="#4FC3F7", width=2),
                                name=dc,
                            )
                        )
                        fig.update_layout(
                            title=f"수경재배 환경: {dc} 일별 평균 변화",
                            xaxis_title="날짜",
                            yaxis_title=f"{dc} ({unit})",
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(
                            f"수경재배 {dc} 데이터가 충분하지 않아 "
                            "그래프를 건너뛰었어요."
                        )
                if not h_date_cols:
                    st.info(
                        "수경재배 환경에서 온도·습도·CO2 열이 없어 "
                        "시계열을 표시하지 못했어요."
                    )
            else:
                st.info(
                    "수경재배 환경 CSV를 업로드하면 수경 환경 시계열이 "
                    "표시됩니다."
                )

            st.markdown("---")

            # ---------- 토양재배 환경 정보 ----------
            if has_soil_env and "time" in soil_env_df.columns:
                st.subheader(
                    "토양재배(노지) 환경 데이터 (매칭 파일에서 추출)"
                )
                soil_num_cols = [
                    c for c in soil_env_df.columns
                    if c not in ("time", "개체번호")
                    and pd.api.types.is_numeric_dtype(soil_env_df[c])
                ]
                if soil_num_cols and len(soil_env_df[soil_num_cols].dropna()) >= 2:
                    soil_summary = {}
                    for c in soil_num_cols:
                        s = soil_env_df[c].dropna()
                        soil_summary[c] = {
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "측정 횟수": len(s),
                        }
                    st.dataframe(
                        pd.DataFrame(soil_summary).T,
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "※ 토양재배 환경 데이터는 환경 데이터 매칭 파일에서 "
                        "추출한 값이며, 비정기 현장 관측 데이터입니다. "
                        "수경재배의 연속 1시간 간격 데이터와 측정 주기가 다르므로 "
                        "직접 시계열 비교는 주의가 필요합니다."
                    )
                else:
                    st.info(
                        "토양재배 환경 데이터에서 숫자 값을 찾을 수 없어요."
                    )
            else:
                st.info(
                    "환경 데이터 매칭 CSV를 업로드하면 토양재배 환경 요약(표)이 "
                    "표시됩니다. 매칭 파일이 없으면 토양재배 환경은 비교할 수 "
                    "없습니다."
                )

            st.markdown("---")

            # ---------- 4분면 맵: 수경 vs 토양 ----------
            st.subheader(
                "1단계 · 온도-습도 4분면 맵 (수경재배 vs 토양재배)"
            )
            xy_cols = ["temperature", "humidity"]
            has_h_xy = (
                has_env_hydro
                and all(c in env_hydro_df.columns for c in xy_cols)
            )
            has_s_xy = (
                has_soil_env
                and all(c in soil_env_df.columns for c in xy_cols)
            )
            if has_h_xy or has_s_xy:
                fig = go.Figure()
                if has_h_xy:
                    h_xy = env_hydro_df[xy_cols].dropna()
                    if len(h_xy) >= 1:
                        fig.add_trace(
                            go.Scatter(
                                x=h_xy["temperature"],
                                y=h_xy["humidity"],
                                mode="markers",
                                marker=dict(
                                    size=2,
                                    color="#4FC3F7",
                                    opacity=0.4,
                                    line=dict(width=0),
                                ),
                                name=f"수경재배 ({len(h_xy)}개 측정)",
                                hovertemplate=(
                                    "수경 온도 %{x:.1f}°C<br>습도 %{y:.1f}%"
                                ),
                            )
                        )
                if has_s_xy:
                    s_xy = soil_env_df[xy_cols].dropna()
                    if len(s_xy) >= 1:
                        fig.add_trace(
                            go.Scatter(
                                x=s_xy["temperature"],
                                y=s_xy["humidity"],
                                mode="markers",
                                marker=dict(
                                    size=8,
                                    color="#FFB74D",
                                    symbol="square",
                                    line=dict(width=1, color="#E65100"),
                                ),
                                name=f"토양재배 ({len(s_xy)}개 측정)",
                                hovertemplate=(
                                    "토양 온도 %{x:.1f}°C<br>습도 %{y:.1f}%"
                                ),
                            )
                        )
                all_x = []
                all_y = []
                if has_h_xy:
                    all_x.extend(h_xy["temperature"].dropna().tolist())
                    all_y.extend(h_xy["humidity"].dropna().tolist())
                if has_s_xy:
                    all_x.extend(s_xy["temperature"].dropna().tolist())
                    all_y.extend(s_xy["humidity"].dropna().tolist())
                if all_x and all_y:
                    x_min, x_max = min(all_x), max(all_x)
                    y_min, y_max = min(all_y), max(all_y)
                    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1
                    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1
                    fig.update_xaxes(
                        range=[x_min - x_pad, x_max + x_pad]
                    )
                    fig.update_yaxes(
                        range=[y_min - y_pad, y_max + y_pad]
                    )

                    if has_h_xy:
                        h_xmed = h_xy["temperature"].median()
                        h_ymed = h_xy["humidity"].median()
                    else:
                        h_xmed = (x_min + x_max) / 2
                        h_ymed = (y_min + y_max) / 2
                    if has_s_xy:
                        s_xmed = s_xy["temperature"].median()
                        s_ymed = s_xy["humidity"].median()
                    else:
                        s_xmed = (x_min + x_max) / 2
                        s_ymed = (y_min + y_max) / 2

                    fig.add_shape(
                        type="line",
                        x0=h_xmed,
                        x1=h_xmed,
                        y0=y_min - y_pad,
                        y1=y_max + y_pad,
                        line=dict(color="#4FC3F7", width=1, dash="dot"),
                    )
                    fig.add_shape(
                        type="line",
                        x0=x_min - x_pad,
                        x1=x_max + x_pad,
                        y0=h_ymed,
                        y1=h_ymed,
                        line=dict(color="#4FC3F7", width=1, dash="dot"),
                    )
                    fig.add_annotation(
                        x=h_xmed + x_pad * 0.3,
                        y=y_max - y_pad * 0.5,
                        text=(
                            f"수경 중앙값<br>({h_xmed:.1f}°C, {h_ymed:.0f}%)"
                        ),
                        showarrow=False,
                        font=dict(size=9, color="#4FC3F7"),
                    )

                    fig.add_shape(
                        type="line",
                        x0=s_xmed,
                        x1=s_xmed,
                        y0=y_min - y_pad,
                        y1=y_max + y_pad,
                        line=dict(color="#FFB74D", width=1, dash="dot"),
                    )
                    fig.add_shape(
                        type="line",
                        x0=x_min - x_pad,
                        x1=x_max + x_pad,
                        y0=s_ymed,
                        y1=s_ymed,
                        line=dict(color="#FFB74D", width=1, dash="dot"),
                    )
                    fig.add_annotation(
                        x=s_xmed - x_pad * 0.4,
                        y=y_min + y_pad * 0.4,
                        text=(
                            f"토양 중앙값<br>({s_xmed:.1f}°C, {s_ymed:.0f}%)"
                        ),
                        showarrow=True,
                        arrowhead=2,
                        font=dict(size=9, color="#FFB74D"),
                    )

                    fig.update_layout(
                        title=(
                            "온도-습도 4분면 맵 — 수경재배(파랑) vs 토양재배(주황)\n"
                            f"수경: {len(h_xy) if has_h_xy else 0}개 측정 "
                            f"(스마트팜, 1시간 간격)  |  "
                            f"토양: {len(s_xy) if has_s_xy else 0}개 측정 "
                            f"(노지, 비정기 관측)"
                        ),
                        xaxis_title="온도 (°C)",
                        yaxis_title="상대습도 (%)",
                        font=dict(
                            family=(
                                "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            )
                        ),
                        margin=dict(l=40, r=20, t=80, b=40),
                        width=750,
                        height=550,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        ),
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        "※ 수경재배는 스마트팜 내부 air sensor로 1시간 간격 자동 "
                        "측정한 값이고, 토양재배는 현장 실측 온습도계로 비정기 측정한 "
                        "값(매칭 파일 기반)입니다. 점 개수가 크게 다르므로, '분포 "
                        "경향'을 보는 용도로 활용하세요."
                    )
            else:
                st.info(
                    "수경재배 환경의 temperature·humidity 열이 모두 없어 "
                    "4분면 맵을 표시할 수 없어요."
                )

            # ---------- 2단계: 생육 지표 비교 막대그래프 ----------
            st.markdown("---")
            st.subheader(
                "2단계 · 재배 방식별 생육 지표 비교 (막대그래프)"
            )
            if has_growth_hydro or has_growth_soil:
                gw_cols = [
                    c
                    for c in selected_measures
                    if (has_growth_hydro and c in growth_hydro_df.columns)
                    or (has_growth_soil and c in growth_soil_df.columns)
                ]
                if gw_cols:
                    rows_bar = []
                    for col in gw_cols:
                        h_mean = (
                            round(
                                growth_hydro_df[col].dropna().mean(), 2
                            )
                            if (
                                has_growth_hydro
                                and col in growth_hydro_df.columns
                                and len(growth_hydro_df[col].dropna()) >= 2
                            )
                            else None
                        )
                        s_mean = (
                            round(
                                growth_soil_df[col].dropna().mean(), 2
                            )
                            if (
                                has_growth_soil
                                and col in growth_soil_df.columns
                                and len(growth_soil_df[col].dropna()) >= 2
                            )
                            else None
                        )
                        rows_bar.append(
                            {
                                "생육 지표": col,
                                "수경재배 평균": h_mean,
                                "토양재배 평균": s_mean,
                            }
                        )
                    bar_df = pd.DataFrame(rows_bar)
                    if (
                        bar_df["수경재배 평균"].notna().any()
                        or bar_df["토양재배 평균"].notna().any()
                    ):
                        fig_bar = go.Figure()
                        fig_bar.add_trace(
                            go.Bar(
                                x=bar_df["생육 지표"],
                                y=bar_df["수경재배 평균"],
                                name=f"수경재배 (n={len(growth_hydro_df)})",
                                marker_color="#4FC3F7",
                                width=0.4,
                            )
                        )
                        fig_bar.add_trace(
                            go.Bar(
                                x=bar_df["생육 지표"],
                                y=bar_df["토양재배 평균"],
                                name=f"토양재배 (n={len(growth_soil_df)})",
                                marker_color="#FFB74D",
                                width=0.4,
                            )
                        )
                        fig_bar.update_layout(
                            title="재배 방식별 생육 지표 평균 비교",
                            xaxis_title="생육 지표",
                            yaxis_title="평균 값",
                            barmode="group",
                            font=dict(
                                family=(
                                    "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                                )
                            ),
                            margin=dict(l=40, r=20, t=50, b=40),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5,
                            ),
                        )
                        st.plotly_chart(fig_bar, width="stretch")
                        st.caption(
                            "※ 파란색 = 수경재배 스마트가든 24개체 평균, "
                            "주황색 = 토양재배 노지 70개체 평균. 개체 수와 측정 "
                            "환경이 다르므로 값의 크기만 단순 비교하기보다 '경향 "
                            "차이'를 보는 용도로 활용하세요."
                        )
                else:
                    st.info(
                        "선택한 측정 항목 중 두 재배 데이터 모두에 있는 열이 "
                        "없어요. 사이드바에서 항목을 다시 선택해 주세요."
                    )
            else:
                st.info(
                    "수경재배 생육 CSV + 토양재배 생육 CSV를 업로드하면 "
                    "생육 지표 비교가 표시됩니다."
                )

            # ---------- 원본 데이터 + 다운로드 ----------
            with st.expander("📄 환경 데이터 원본 + CSV 다운로드"):
                if has_env_hydro:
                    st.subheader("수경재배 환경 데이터 원본")
                    st.dataframe(env_hydro_df, width="stretch")
                    env_csv = env_hydro_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="수경재배 환경 CSV 다운로드 (utf-8-sig)",
                        data=env_csv,
                        file_name=f"{school_name}_환경데이터_수경.csv",
                        mime="text/csv",
                        key="env_hydro_download_tab2",
                    )
                if has_soil_env:
                    if has_env_hydro:
                        st.markdown("---")
                    st.subheader(
                        "토양재배 환경 데이터 (매칭 파일에서 추출)"
                    )
                    st.dataframe(soil_env_df, width="stretch")
                    soil_env_csv = soil_env_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="토양재배 환경 CSV 다운로드 (utf-8-sig)",
                        data=soil_env_csv,
                        file_name=f"{school_name}_환경데이터_토양.csv",
                        mime="text/csv",
                        key="soil_env_download_tab2",
                    )
                if not has_env_hydro and not has_soil_env:
                    st.info("표시할 환경 데이터가 없습니다.")

    except Exception as e:
        st.warning(
            f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}"
        )


# ============================================================
# TAB 3 · 생육 결과 비교 (수경 vs 토양)
# ============================================================
with tab3:
    try:
        if not has_growth_hydro and not has_growth_soil:
            st.info(
                "수경재배 생육 CSV + 토양재배 생육 CSV를 업로드하면 "
                "개체별 생육 차이 분석이 표시됩니다."
            )
        else:
            def get_gw(df):
                if "생중량" in df.columns:
                    return df["생중량"].dropna()
                return pd.Series(dtype=float)

            def get_series(df, col):
                if col in df.columns:
                    return df[col].dropna()
                return pd.Series(dtype=float)

            # ----- 🥇 핵심 결과 카드 -----
            st.subheader("🥇 핵심 결과 카드")

            h_gw = get_gw(growth_hydro_df)
            s_gw = get_gw(growth_soil_df)
            h_leaf = get_series(growth_hydro_df, "잎 수")
            s_leaf = get_series(growth_soil_df, "잎 수")
            h_shoot = get_series(growth_hydro_df, "지상부 길이")
            s_shoot = get_series(growth_soil_df, "지상부 길이")
            h_root = get_series(growth_hydro_df, "지하부 길이")
            s_root = get_series(growth_soil_df, "지하부 길이")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            if len(h_gw) >= 2:
                h_avg_gw = round(h_gw.mean(), 2)
                h_std_gw = round(h_gw.std(), 2)
                h_cv = (
                    round(h_std_gw / h_avg_gw * 100, 1)
                    if h_avg_gw > 0
                    else 0
                )
                kpi1.metric("수경 평균 생중량 (g)", h_avg_gw)
                kpi2.metric("수경 CV (%)", h_cv)
            else:
                kpi1.metric("수경 평균 생중량 (g)", "자료 부족")
                kpi2.metric("수경 CV (%)", "자료 부족")

            if len(s_gw) >= 2:
                s_avg_gw = round(s_gw.mean(), 2)
                s_std_gw = round(s_gw.std(), 2)
                s_cv = (
                    round(s_std_gw / s_avg_gw * 100, 1)
                    if s_avg_gw > 0
                    else 0
                )
                kpi3.metric("토양 평균 생중량 (g)", s_avg_gw)
                kpi4.metric("토양 CV (%)", s_cv)
            else:
                kpi3.metric("토양 평균 생중량 (g)", "자료 부족")
                kpi4.metric("토양 CV (%)", "자료 부족")

            st.caption(
                "CV(변동계수, %) = (표준편차 ÷ 평균) × 100 — 개체 간 생육 "
                "차이가 상대적으로 얼마나 큰지 나타냄. 값이 클수록 개체 간 "
                "편차가 크다는 의미입니다."
            )

            st.markdown("---")

            # 최대·최소 개체 정보
            if len(h_gw) >= 2:
                h_idx_max = h_gw.idxmax()
                h_idx_min = h_gw.idxmin()
                h_max_id = growth_hydro_df.loc[h_idx_max, "개체번호"]
                h_min_id = growth_hydro_df.loc[h_idx_min, "개체번호"]
                st.info(
                    f"🏆 수경재배 생중량 **최대** 개체: #{h_max_id}  "
                    f"({round(h_gw.max(), 2)} g)   |   "
                    f"📉 수경재배 생중량 **최소** 개체: #{h_min_id}  "
                    f"({round(h_gw.min(), 2)} g)"
                )
            if len(s_gw) >= 2:
                s_idx_max = s_gw.idxmax()
                s_idx_min = s_gw.idxmin()
                s_max_id = growth_soil_df.loc[s_idx_max, "개체번호"]
                s_min_id = growth_soil_df.loc[s_idx_min, "개체번호"]
                st.info(
                    f"🏆 토양재배 생중량 **최대** 개체: #{s_max_id}  "
                    f"({round(s_gw.max(), 2)} g)   |   "
                    f"📉 토양재배 생중량 **최소** 개체: #{s_min_id}  "
                    f"({round(s_gw.min(), 2)} g)"
                )

            st.markdown("---")

            # ----- 개체별 생육 비교 (그룹별 평균 막대, 지표 수만큼 행) -----
            st.subheader(
                "개체별 생육 비교 (수경 vs 토양, 지표별 평균 막대그래프)"
            )
            bar_cols = [
                c
                for c in selected_measures
                if (has_growth_hydro and c in growth_hydro_df.columns)
                or (has_growth_soil and c in growth_soil_df.columns)
            ]
            if bar_cols:
                h_means = {}
                s_means = {}
                for col in bar_cols:
                    if (
                        has_growth_hydro
                        and col in growth_hydro_df.columns
                        and len(get_series(growth_hydro_df, col)) >= 2
                    ):
                        h_means[col] = round(
                            get_series(growth_hydro_df, col).mean(), 2
                        )
                    else:
                        h_means[col] = None
                    if (
                        has_growth_soil
                        and col in growth_soil_df.columns
                        and len(get_series(growth_soil_df, col)) >= 2
                    ):
                        s_means[col] = round(
                            get_series(growth_soil_df, col).mean(), 2
                        )
                    else:
                        s_means[col] = None

                n_rows = len(bar_cols)
                fig_ind = make_subplots(
                    rows=n_rows,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=bar_cols,
                    vertical_spacing=0.12,
                )
                for i, col in enumerate(bar_cols):
                    row_n = i + 1
                    is_first = (i == 0)
                    # 수경재배 trace
                    if (
                        has_growth_hydro
                        and col in growth_hydro_df.columns
                    ):
                        h_series = get_series(growth_hydro_df, col)
                        if len(h_series) >= 2:
                            h_labels = [
                                f"수경 #{j+1}"
                                for j in range(len(h_series))
                            ]
                            fig_ind.add_trace(
                                go.Bar(
                                    x=h_labels,
                                    y=h_series.values,
                                    name="수경재배",
                                    marker_color="#4FC3F7",
                                    showlegend=is_first,
                                    xaxis="x" if row_n == 1 else "x",
                                    yaxis="y" if row_n == 1 else f"y{row_n}",
                                ),
                                row=row_n,
                                col=1,
                            )
                    # 토양재배 trace
                    if (
                        has_growth_soil
                        and col in growth_soil_df.columns
                    ):
                        s_series = get_series(growth_soil_df, col)
                        if len(s_series) >= 2:
                            s_labels = [
                                f"토양 #{j+1}"
                                for j in range(len(s_series))
                            ]
                            fig_ind.add_trace(
                                go.Bar(
                                    x=s_labels,
                                    y=s_series.values,
                                    name="토양재배",
                                    marker_color="#FFB74D",
                                    showlegend=is_first,
                                    xaxis="x" if row_n == 1 else "x",
                                    yaxis="y" if row_n == 1 else f"y{row_n}",
                                ),
                                row=row_n,
                                col=1,
                            )
                fig_ind.update_layout(
                    height=380 * n_rows,
                    font=dict(
                        family=(
                            "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        )
                    ),
                    margin=dict(l=70, r=20, t=60, b=40),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.01,
                        xanchor="center",
                        x=0.5,
                    ),
                )
                n_total = len(growth_hydro_df) + len(growth_soil_df)
                fig_ind.update_xaxes(
                    tickangle=-45 if n_total > 40 else 0
                )
                st.plotly_chart(fig_ind, width="stretch")
            else:
                st.info(
                    "선택한 측정 항목 중 생육 데이터에 있는 열이 없어요. "
                    "사이드바에서 항목을 다시 선택해 주세요."
                )

            st.markdown("---")

            # ----- 생중량 분포 (박스플롯) -----
            st.subheader("생중량 분포 (박스플롯 · 수경 vs 토양)")
            h_gw_box = (
                get_gw(growth_hydro_df)
                if len(get_gw(growth_hydro_df)) >= 2
                else pd.Series(dtype=float)
            )
            s_gw_box = (
                get_gw(growth_soil_df)
                if len(get_gw(growth_soil_df)) >= 2
                else pd.Series(dtype=float)
            )
            fig_box = go.Figure()
            if len(h_gw_box) >= 2:
                fig_box.add_trace(
                    go.Box(
                        y=h_gw_box,
                        name=f"수경재배 (n={len(h_gw_box)})",
                        fillcolor="#4FC3F7",
                        marker=dict(color="#4FC3F7"),
                        boxpoints="outliers",
                    )
                )
            if len(s_gw_box) >= 2:
                fig_box.add_trace(
                    go.Box(
                        y=s_gw_box,
                        name=f"토양재배 (n={len(s_gw_box)})",
                        fillcolor="#FFB74D",
                        marker=dict(color="#FFB74D"),
                        boxpoints="outliers",
                    )
                )
            fig_box.update_layout(
                title="생중량 분포 비교 (박스플롯)",
                yaxis_title="생중량 (g)",
                font=dict(
                    family=(
                        "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                    )
                ),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_box, width="stretch")

            st.markdown("---")

            # ----- 상관관계 분석 (산점도 + 추세선) -----
            st.subheader("상관관계 분석 (산점도 + 추세선)")

            def scatter_with_trend(df_label, df, x_col, y_col, color):
                """x_col vs y_col 산점도 + np.polyfit 추세선 Figure 반환"""
                x_s = get_series(df, x_col)
                if y_col == "생중량":
                    y_s = get_gw(df)
                else:
                    y_s = get_series(df, y_col)
                if len(x_s) < 2 or len(y_s) < 2 or len(x_s) != len(y_s):
                    return None
                xy = pd.concat([x_s.reset_index(drop=True),
                                y_s.reset_index(drop=True)], axis=1).dropna()
                if len(xy) < 2:
                    return None
                x = xy.iloc[:, 0].values
                y = xy.iloc[:, 1].values
                r = np.corrcoef(x, y)[0, 1]
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                if r > 0.5:
                    tag = "강한 양의 상관"
                elif r < -0.5:
                    tag = "강한 음의 상관"
                elif abs(r) >= 0.3:
                    tag = "중간 상관"
                else:
                    tag = "약한 관계"
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=dict(size=7, color=color, opacity=0.7),
                        name=f"{df_label} ({len(x)}개체)",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="red", width=2, dash="dot"),
                        name=f"추세선 (r={round(r, 3)})",
                    )
                )
                fig.add_annotation(
                    x=0.98,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text=f"상관계수 r = {round(r, 3)}  ({tag})",
                    showarrow=False,
                    font=dict(size=11, color="black"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="gray",
                    borderwidth=1,
                )
                fig.update_layout(
                    title=f"{x_col} vs {y_col}  |  {df_label}",
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    font=dict(
                        family=(
                            "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        )
                    ),
                    margin=dict(l=40, r=20, t=60, b=40),
                )
                return fig

            # 수경재배 상관계수
            if has_growth_hydro:
                st.markdown("#### 수경재배(스마트팜) 내 상관")
                for x_col in ["잎 수", "지상부 길이"]:
                    if (
                        x_col in growth_hydro_df.columns
                        and "생중량" in growth_hydro_df.columns
                    ):
                        fig_h = scatter_with_trend(
                            "수경재배", growth_hydro_df, x_col, "생중량",
                            "#4FC3F7"
                        )
                        if fig_h is not None:
                            st.plotly_chart(fig_h, width="stretch")
                        else:
                            st.info(
                                f"수경재배 {x_col} vs 생중량 데이터가 "
                                "충분하지 않아 그래프를 건너뛰었어요."
                            )
                    else:
                        missing = [
                            c
                            for c in [x_col, "생중량"]
                            if c not in growth_hydro_df.columns
                        ]
                        st.info(
                            f"{', '.join(missing)} 열이 없어 수경재배의 "
                            "이 상관을 건너뛰었어요."
                        )

            st.markdown("---")

            # 토양재배 상관계수
            if has_growth_soil:
                st.markdown("#### 토양재배(노지) 내 상관")
                for x_col in ["잎 수", "지상부 길이"]:
                    if (
                        x_col in growth_soil_df.columns
                        and "생중량" in growth_soil_df.columns
                    ):
                        fig_s = scatter_with_trend(
                            "토양재배", growth_soil_df, x_col, "생중량",
                            "#FFB74D"
                        )
                        if fig_s is not None:
                            st.plotly_chart(fig_s, width="stretch")
                        else:
                            st.info(
                                f"토양재배 {x_col} vs 생중량 데이터가 "
                                "충분하지 않아 그래프를 건너뛰었어요."
                            )
                    else:
                        missing = [
                            c
                            for c in [x_col, "생중량"]
                            if c not in growth_soil_df.columns
                        ]
                        st.info(
                            f"{', '.join(missing)} 열이 없어 토양재배의 "
                            "이 상관을 건너뛰었어요."
                        )

            if not has_growth_hydro and not has_growth_soil:
                st.info("상관계수를 표시할 생육 데이터가 없습니다.")

            st.markdown("---")

            # ----- 원본 생육 데이터 + 다운로드 -----
            with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                if has_growth_hydro:
                    st.subheader("수경재배 생육 데이터 원본")
                    st.dataframe(growth_hydro_df, width="stretch")
                    gw_h_csv = growth_hydro_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="수경재배 생육 CSV 다운로드 (utf-8-sig)",
                        data=gw_h_csv,
                        file_name=f"{school_name}_생육데이터_수경.csv",
                        mime="text/csv",
                        key="gw_hydro_download_tab3",
                    )
                if has_growth_soil:
                    if has_growth_hydro:
                        st.markdown("---")
                    st.subheader("토양재배 생육 데이터 원본")
                    st.dataframe(growth_soil_df, width="stretch")
                    gw_s_csv = growth_soil_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="토양재배 생육 CSV 다운로드 (utf-8-sig)",
                        data=gw_s_csv,
                        file_name=f"{school_name}_생육데이터_토양.csv",
                        mime="text/csv",
                        key="gw_soil_download_tab3",
                    )
                if not has_growth_hydro and not has_growth_soil:
                    st.info("표시할 생육 데이터가 없습니다.")

    except Exception as e:
        st.warning(
            f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}"
        )
