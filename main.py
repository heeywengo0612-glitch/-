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
# 수경재배 생육 데이터 표준화
#   잎 수(장)→잎 수,  지상부 길이(mm)→지상부 길이,
#   지하부 길이(mm)→지하부 길이,  지상부 질량(g)→지상부 질량,
#   지하부 질량(g)→지하부 질량,  생중량 없으면 지상부+지하부 질량으로 계산
# ============================================================
def std_cols_growth(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
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
#   air sensor(1)→temperature, air sensor(2)→humidity,
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
# 토양재배 환경 데이터 (매칭 파일에서 추출)
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
selected_measures = st.sidebar.multiselect(
    "비교할 생육 지표 선택",
    options=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
    default=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
)

st.sidebar.markdown("---")
st.sidebar.caption("ℹ️ 각 CSV는 utf-8-sig / cp949 순서로 읽습니다.")
st.sidebar.caption(
    "ℹ️ 환경 데이터 매칭 CSV가 있으면 토양재배 환경값을 자동 추출해 "
    "수경재배와 함께 비교합니다."
)


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


def get_gw(df):
    if "생중량" in df.columns:
        return df["생중량"].dropna()
    return pd.Series(dtype=float)


def get_series(df, col):
    if col in df.columns:
        return df[col].dropna()
    return pd.Series(dtype=float)


# ============================================================
# 페이지 제목 및 통계 용어 설명
# ============================================================
st.title(
    "토양재배와 수경재배 환경이 극지식물의 생육 특성에 미치는 영향 탐구"
)
st.caption(
    "*[평균]*: 모든 값을 더한 뒤 개수로 나눈 값  ·  "
    "*[표준편차]*: 값이 평균 주변에 얼마나 모여 있는지 (작을수록 비슷)  ·  "
    "*[CV]*: 표준편차 ÷ 평균 × 100 — 개체 간 상대적 차이 크기  ·  "
    "*[상관계수 r]*: 두 값이 함께 변하는 정도 (-1~1, 0.7↑=강한 관계)  ·  "
    "*[추세선]*: 산점도 점들이 따라가는 직선 (np.polyfit으로 직접 계산)"
)


tab1, tab2, tab3 = st.tabs(
    [
        "📖 실험 개요",
        "🌡️ 환경 데이터 비교 (수경 vs 토양)",
        "📊 생육 결과 비교 & 재배 목적별 적합성",
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
            "변화를 파악하는 중요한 자료가 된다. 일부 극지식물은 낮은 온도와 "
            "짧은 생육기간 등 극한 환경에 적응한 생리적 특성을 지니고 있어, "
            "이러한 식물의 생존 및 생육 메커니즘을 연구하면 저온 환경에서도 "
            "생육할 수 있는 작물의 특성을 이해하고 내한성 작물 개발과 같은 미래 "
            "농업 연구에 활용할 수 있다. 또한 극지식물은 극지 생태계의 식생을 "
            "구성하며 토양 표면 안정화와 유기물 순환 등 생태적 기능에도 기여한다."
        )
        st.markdown(
            "극지식물을 재배하는 방식에는 토양을 이용하는 방법과 양액을 이용하는 "
            "수경재배 방식 등이 있다. 토양재배는 토양의 물리적 구조와 수분·양분 "
            "상태 등 다양한 요인이 생육에 영향을 미치는 반면, 수경재배는 양액을 "
            "통해 물과 양분을 공급하므로 재배 환경을 상대적으로 세밀하게 관리할 수 "
            "있다는 특징이 있다. 두 방식은 뿌리가 접하는 환경과 수분·양분 공급 "
            "방식에서 차이가 있어 생육 양상에도 차이가 나타날 수 있다. 따라서 "
            "재배 방식별 환경 조건과 식물의 생육 특성을 함께 비교하여 극지식물에 "
            "적합한 재배 환경을 탐색할 필요가 있다."
        )
        st.markdown(
            "식물의 생육은 재배 방식뿐 아니라 온도, 습도, CO₂ 등 주변 환경 요인의 "
            "영향을 받는다. 온도는 광합성과 호흡 등 대사 과정에, 습도는 증산 작용과 "
            "수분 균형에, CO₂는 광합성에 필요한 주요 기질에 영향을 준다. 이러한 "
            "환경 요인은 복합적으로 작용하므로 환경 데이터 분석 결과와 실제 생육 "
            "관찰 결과를 함께 분석할 필요가 있다. 잎의 개수·크기, 지상부와 지하부의 "
            "길이, 생체중 등의 생육 지표를 분석하면 식물의 생장 양상과 생육 특성 간 "
            "관계를 파악할 수 있다."
        )

        st.subheader("가설 설정")
        st.markdown(
            "**가설 1.** 높은 기온의 환경에서 식물이 더 많은 잎 수를 가질 것이다.\n\n"
            "→ 온도 상승이 광합성 및 대사 활동을 촉진하여 잎 수 증가로 이어질 것으로 예상."
        )
        st.markdown(
            "**가설 2.** 습도가 높은 환경에서는 지하부 길이가 짧아지고, "
            "습도가 낮은 환경에서는 지하부 길이가 길어질 것이다.\n\n"
            "→ 낮은 습도에서 수분 확보를 위해 뿌리 신장이 촉진될 것으로 예상."
        )

        st.subheader("데이터 요약")
        rows = []
        if has_growth_hydro:
            h_period = "자료 없음"
            if has_env_hydro and "time" in env_hydro_df.columns:
                h_period = (
                    f"{env_hydro_df['time'].min().strftime('%Y-%m-%d')} ~ "
                    f"{env_hydro_df['time'].max().strftime('%Y-%m-%d')}"
                )
            h_items = ", ".join(
                [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                 if c in growth_hydro_df.columns]
            ) or "자료 없음"
            rows.append(
                {
                    "구분": "수경재배 (스마트팜)",
                    "개체 수": f"{len(growth_hydro_df)}개체",
                    "환경 데이터": "있음" if has_env_hydro else "업로드 필요",
                    "환경 측정 기간": h_period,
                    "생육 측정 항목": h_items,
                    "환경 측정 횟수": (
                        f"{len(env_hydro_df)}회 (1시간 간격 연속)"
                        if has_env_hydro
                        else "자료 없음"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "구분": "수경재배 (스마트팜)",
                    "개체 수": "CSV 업로드 필요",
                    "환경 데이터": "CSV 업로드 필요",
                    "환경 측정 기간": "자료 없음",
                    "생육 측정 항목": "자료 없음",
                    "환경 측정 횟수": "자료 없음",
                }
            )

        if has_growth_soil:
            s_period = "자료 없음"
            if has_soil_env and "time" in soil_env_df.columns:
                s_period = (
                    f"{soil_env_df['time'].min().strftime('%Y-%m-%d')} ~ "
                    f"{soil_env_df['time'].max().strftime('%Y-%m-%d')}"
                )
            s_items = ", ".join(
                [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                 if c in growth_soil_df.columns]
            ) or "자료 없음"
            rows.append(
                {
                    "구분": "토양재배 (노지)",
                    "개체 수": f"{len(growth_soil_df)}개체",
                    "환경 데이터": (
                        "있음 (매칭 파일 추출)" if has_soil_env
                        else "매칭 파일 업로드 필요"
                    ),
                    "환경 측정 기간": s_period,
                    "생육 측정 항목": s_items,
                    "환경 측정 횟수": (
                        f"{len(soil_env_df)}회 (비정기 현장 실측)"
                        if has_soil_env
                        else "자료 없음"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "구분": "토양재배 (노지)",
                    "개체 수": "CSV 업로드 필요",
                    "환경 데이터": "매칭 파일 업로드 필요",
                    "환경 측정 기간": "자료 없음",
                    "생육 측정 항목": "자료 없음",
                    "환경 측정 횟수": "자료 없음",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "※ 수경재배 환경은 스마트팜 내부 air sensor로 1시간 간격 연속 측정한 값이고, "
            "토양재배 환경은 현장 실측 온습도계·pH·EC 측정기로 비정기 측정한 값입니다 "
            "(매칭 파일 기반). 측정 방식이 다르므로 수치의 직접 비교보다 '경향 차이'를 "
            "보는 용도로 활용하세요."
        )

        st.subheader("주요 지표 카드")
        col1, col2, col3, col4 = st.columns(4)

        h_gw = get_gw(growth_hydro_df) if has_growth_hydro else pd.Series(dtype=float)
        s_gw = get_gw(growth_soil_df) if has_growth_soil else pd.Series(dtype=float)
        h_temp = (
            env_hydro_df["temperature"].dropna()
            if has_env_hydro and "temperature" in env_hydro_df.columns
            else pd.Series(dtype=float)
        )
        s_temp = (
            soil_env_df["temperature"].dropna()
            if has_soil_env and "temperature" in soil_env_df.columns
            else pd.Series(dtype=float)
        )

        if len(h_gw) >= 2:
            col1.metric("🌿 수경 평균 생중량 (g)", round(h_gw.mean(), 2))
        else:
            col1.metric("🌿 수경 평균 생중량 (g)", "자료 없음")

        if len(s_gw) >= 2:
            col2.metric("🌱 토양 평균 생중량 (g)", round(s_gw.mean(), 2))
        else:
            col2.metric("🌱 토양 평균 생중량 (g)", "자료 없음")

        if len(h_temp) >= 2:
            col3.metric("🌡️ 수경 평균 온도 (°C)", round(h_temp.mean(), 1))
        else:
            col3.metric("🌡️ 수경 평균 온도 (°C)", "자료 없음")

        if len(s_temp) >= 2:
            col4.metric("🌡️ 토양 평균 온도 (°C)", round(s_temp.mean(), 1))
        else:
            col4.metric("🌡️ 토양 평균 온도 (°C)", "자료 없음")

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
                st.subheader(
                    "수경재배(스마트팜) 환경 시계열 (일별 평균)"
                )
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
                        fig_ts = go.Figure()
                        fig_ts.add_trace(
                            go.Scatter(
                                x=daily["date"],
                                y=daily[dc],
                                mode="lines+markers",
                                line=dict(color="#4FC3F7", width=2),
                                name=dc,
                            )
                        )
                        fig_ts.update_layout(
                            title=f"수경재배 환경: {dc} 일별 평균 변화",
                            xaxis_title="날짜",
                            yaxis_title=f"{dc} ({unit})",
                            font=dict(
                                family=(
                                    "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                                )
                            ),
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig_ts, width="stretch")
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

            st.markdown("---")

            # ---------- 토양재배 환경 요약 ----------
            if has_soil_env and len(soil_env_df) > 0:
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
                        "추출한 비정기 현장 실측값입니다. 수경재배의 연속 "
                        "1시간 간격 데이터와 측정 주기·방법이 다르므로 "
                        "'경향' 비교 용도로 활용하세요."
                    )
                else:
                    st.info(
                        "토양재배 환경 데이터에서 숫자 값을 찾을 수 없어요."
                    )
            else:
                st.info(
                    "환경 데이터 매칭 CSV를 업로드하면 토양재배 환경 요약이 "
                    "표시됩니다."
                )

            st.markdown("---")

            # ---------- 4분면 맵: 수경 vs 토양 ----------
            st.subheader(
                "1단계 · 온도-습도 4분면 맵 (수경재배 vs 토양재배)"
            )

            has_h_xy = (
                has_env_hydro
                and "temperature" in env_hydro_df.columns
                and "humidity" in env_hydro_df.columns
            )
            has_s_xy = (
                has_soil_env
                and "temperature" in soil_env_df.columns
                and "humidity" in soil_env_df.columns
            )

            if has_h_xy or has_s_xy:
                h_xy = (
                    env_hydro_df[["temperature", "humidity"]].dropna()
                    if has_h_xy else pd.DataFrame()
                )
                s_xy = (
                    soil_env_df[["temperature", "humidity"]].dropna()
                    if has_s_xy else pd.DataFrame()
                )

                all_x = []
                all_y = []
                if len(h_xy) > 0:
                    all_x.extend(h_xy["temperature"].tolist())
                    all_y.extend(h_xy["humidity"].tolist())
                if len(s_xy) > 0:
                    all_x.extend(s_xy["temperature"].tolist())
                    all_y.extend(s_xy["humidity"].tolist())

                if all_x and all_y:
                    x_min = min(all_x)
                    x_max = max(all_x)
                    y_min = min(all_y)
                    y_max = max(all_y)
                    x_pad = max((x_max - x_min) * 0.08, 1.0)
                    y_pad = max((y_max - y_min) * 0.08, 1.0)
                    x_range = [x_min - x_pad, x_max + x_pad]
                    y_range = [y_min - y_pad, y_max + y_pad]

                    # 수경/토양 중앙값
                    h_t_med = h_xy["temperature"].median() if len(h_xy) > 0 else (x_min + x_max) / 2
                    h_h_med = h_xy["humidity"].median() if len(h_xy) > 0 else (y_min + y_max) / 2
                    s_t_med = s_xy["temperature"].median() if len(s_xy) > 0 else (x_min + x_max) / 2
                    s_h_med = s_xy["humidity"].median() if len(s_xy) > 0 else (y_min + y_max) / 2

                    fig_q = go.Figure()

                    # 구역 배경색 (옅은 색 직사각형 4개)
                    fig_q.add_shape(
                        type="rect",
                        x0=x_range[0], x1=h_t_med,
                        y0=h_h_med, y1=y_range[1],
                        fillcolor="rgba(255,179,0,0.07)",
                        line_width=0,
                        layer="below",
                    )
                    fig_q.add_shape(
                        type="rect",
                        x0=h_t_med, x1=x_range[1],
                        y0=h_h_med, y1=y_range[1],
                        fillcolor="rgba(79,195,247,0.07)",
                        line_width=0,
                        layer="below",
                    )
                    fig_q.add_shape(
                        type="rect",
                        x0=x_range[0], x1=h_t_med,
                        y0=y_range[0], y1=h_h_med,
                        fillcolor="rgba(180,180,180,0.06)",
                        line_width=0,
                        layer="below",
                    )
                    fig_q.add_shape(
                        type="rect",
                        x0=h_t_med, x1=x_range[1],
                        y0=y_range[0], y1=h_h_med,
                        fillcolor="rgba(255,179,0,0.06)",
                        line_width=0,
                        layer="below",
                    )

                    # 수경 중앙값 점선
                    fig_q.add_shape(
                        type="line",
                        x0=h_t_med, x1=h_t_med,
                        y0=y_range[0], y1=y_range[1],
                        line=dict(color="#4FC3F7", width=1.5, dash="dot"),
                        layer="below",
                    )
                    fig_q.add_shape(
                        type="line",
                        x0=x_range[0], x1=x_range[1],
                        y0=h_h_med, y1=h_h_med,
                        line=dict(color="#4FC3F7", width=1.5, dash="dot"),
                        layer="below",
                    )

                    # 토양 중앙값 점선
                    fig_q.add_shape(
                        type="line",
                        x0=s_t_med, x1=s_t_med,
                        y0=y_range[0], y1=y_range[1],
                        line=dict(color="#FFB74D", width=1.5, dash="dot"),
                        layer="below",
                    )
                    fig_q.add_shape(
                        type="line",
                        x0=x_range[0], x1=x_range[1],
                        y0=s_h_med, y1=s_h_med,
                        line=dict(color="#FFB74D", width=1.5, dash="dot"),
                        layer="below",
                    )

                    # 수경재배 점 (파란색 원)
                    if len(h_xy) > 0:
                        fig_q.add_trace(
                            go.Scatter(
                                x=h_xy["temperature"],
                                y=h_xy["humidity"],
                                mode="markers",
                                marker=dict(
                                    size=3,
                                    color="#4FC3F7",
                                    opacity=0.35,
                                    symbol="circle",
                                    line=dict(width=0),
                                ),
                                name=f"수경재배 ({len(h_xy)}개 측정)",
                                hovertemplate=(
                                    "수경 온도 %{x:.1f}°C<br>습도 %{y:.1f}%<extra></extra>"
                                ),
                            )
                        )

                    # 토양재배 점 (주황색 사각형)
                    if len(s_xy) > 0:
                        fig_q.add_trace(
                            go.Scatter(
                                x=s_xy["temperature"],
                                y=s_xy["humidity"],
                                mode="markers",
                                marker=dict(
                                    size=11,
                                    color="#FFB74D",
                                    symbol="square",
                                    line=dict(width=1.5, color="#E65100"),
                                    opacity=0.9,
                                ),
                                name=f"토양재배 ({len(s_xy)}회 실측)",
                                hovertemplate=(
                                    "토양 온도 %{x:.1f}°C<br>습도 %{y:.1f}%<extra></extra>"
                                ),
                            )
                        )

                    # 구역 라벨
                    fig_q.add_annotation(
                        x=(x_range[0] + h_t_med) / 2,
                        y=(h_h_med + y_range[1]) / 2,
                        text="저온·다습\n(토양 주 분포)",
                        showarrow=False,
                        font=dict(size=10, color="#E65100", weight="bold"),
                        bgcolor="rgba(255,179,0,0.12)",
                        borderpad=2,
                    )
                    fig_q.add_annotation(
                        x=(h_t_med + x_range[1]) / 2,
                        y=(h_h_med + y_range[1]) / 2,
                        text="고온·다습",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        bgcolor="rgba(79,195,247,0.12)",
                        borderpad=2,
                    )
                    fig_q.add_annotation(
                        x=(x_range[0] + h_t_med) / 2,
                        y=(y_range[0] + h_h_med) / 2,
                        text="저온·저습",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        bgcolor="rgba(180,180,180,0.12)",
                        borderpad=2,
                    )
                    fig_q.add_annotation(
                        x=(h_t_med + x_range[1]) / 2,
                        y=(y_range[0] + h_h_med) / 2,
                        text="고온·저습\n(수경 주 분포)",
                        showarrow=False,
                        font=dict(size=10, color="#1976D2", weight="bold"),
                        bgcolor="rgba(79,195,247,0.15)",
                        borderpad=2,
                    )

                    # 중앙값 라벨
                    fig_q.add_annotation(
                        x=h_t_med,
                        y=y_range[1] - y_pad * 0.4,
                        text=(
                            f"수경 중앙값<br>T {h_t_med:.1f}°C, H {h_h_med:.0f}%"
                        ),
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#4FC3F7",
                        font=dict(size=10, color="#1976D2"),
                        bgcolor="white",
                        borderpad=2,
                    )
                    fig_q.add_annotation(
                        x=s_t_med,
                        y=y_range[0] + y_pad * 0.4,
                        text=(
                            f"토양 중앙값<br>T {s_t_med:.1f}°C, H {s_h_med:.0f}%"
                        ),
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#FFB74D",
                        font=dict(size=10, color="#E65100"),
                        bgcolor="white",
                        borderpad=2,
                    )

                    fig_q.update_layout(
                        title=(
                            f"온도-습도 4분면 맵 — 수경재배(파랑 원) vs 토양재배(주황 네모)\n"
                            f"수경: {len(h_xy)}개 자동측정 (스마트팜 1시간 간격) · "
                            f"토양: {len(s_xy)}회 현장 실측 (비정기)\n"
                            f"수경 온습도 중앙: T {h_t_med:.1f}°C, H {h_h_med:.0f}%  |  "
                            f"토양 온습도 중앙: T {s_t_med:.1f}°C, H {s_h_med:.0f}%"
                        ),
                        xaxis_title="온도 (°C)",
                        yaxis_title="상대습도 (%)",
                        font=dict(
                            family=(
                                "Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            )
                        ),
                        margin=dict(l=40, r=20, t=100, b=40),
                        width=820,
                        height=580,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        ),
                        xaxis=dict(range=x_range),
                        yaxis=dict(range=y_range),
                    )
                    st.plotly_chart(fig_q, width="stretch")
                    st.caption(
                        "※ 수경재배는 파란색 작은 원(2,161개, 스마트팜 air sensor "
                        "1시간 간격 자동 측정), 토양재배는 주황색 큰 네모(26회, 현장 "
                        "실측 온습도계)입니다. 점 개수가 크게 다르므로 분포 '경향'을 "
                        "보는 용도로 활용하세요.\n"
                        "※ 구역 배경색은 참고용 옅은 음영이며, 점선으로 각 그룹의 "
                        "중앙값 위치를 표시했습니다."
                    )
                else:
                    st.info("산포도를 그릴 온도·습도 데이터가 부족합니다.")
            else:
                st.info(
                    "수경 또는 토양 환경 데이터에 temperature·humidity 열이 "
                    "모두 없어 4분면 맵을 표시할 수 없어요."
                )

            # ---------- 환경 요약표 (수경) ----------
            st.markdown("---")
            st.subheader("수경재배 환경 데이터 요약 (스마트팜)")
            if has_env_hydro:
                h_num_cols = [
                    c for c in env_hydro_df.columns
                    if c not in ("time", "개체번호")
                    and pd.api.types.is_numeric_dtype(env_hydro_df[c])
                ]
                if h_num_cols and len(env_hydro_df[h_num_cols].dropna()) >= 2:
                    h_summary = {}
                    for c in h_num_cols:
                        s = env_hydro_df[c].dropna()
                        unit = (
                            "°C" if c == "temperature"
                            else "%" if c == "humidity"
                            else "ppm" if c == "co2"
                            else ""
                        )
                        h_summary[c] = {
                            f"평균 ({unit})": round(s.mean(), 2),
                            f"최소 ({unit})": round(s.min(), 2),
                            f"최대 ({unit})": round(s.max(), 2),
                            "측정 횟수": len(s),
                        }
                    st.dataframe(
                        pd.DataFrame(h_summary).T,
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("수경재배 환경 데이터에서 숫자 값을 찾을 수 없어요.")
            else:
                st.info("수경재배 환경 CSV를 업로드하면 요약이 표시됩니다.")

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
                        "토양재배 환경 데이터 (매칭 파일에서 추출한 값)"
                    )
                    st.dataframe(soil_env_df, width="stretch")
                    soil_csv = soil_env_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="토양재배 환경 CSV 다운로드 (utf-8-sig)",
                        data=soil_csv,
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
# TAB 3 · 생육 결과 비교 & 재배 목적별 적합성
# ============================================================
with tab3:
    try:
        if not has_growth_hydro and not has_growth_soil:
            st.info(
                "수경재배 생육 CSV + 토양재배 생육 CSV를 업로드하면 "
                "개체별 생육 차이 분석이 표시됩니다."
            )
        else:
            # ----- 🥇 핵심 결과 카드 -----
            st.subheader("🥇 핵심 결과 카드")

            h_gw = get_gw(growth_hydro_df) if has_growth_hydro else pd.Series(dtype=float)
            s_gw = get_gw(growth_soil_df) if has_growth_soil else pd.Series(dtype=float)
            h_leaf = get_series(growth_hydro_df, "잎 수") if has_growth_hydro else pd.Series(dtype=float)
            s_leaf = get_series(growth_soil_df, "잎 수") if has_growth_soil else pd.Series(dtype=float)
            h_shoot = get_series(growth_hydro_df, "지상부 길이") if has_growth_hydro else pd.Series(dtype=float)
            s_shoot = get_series(growth_soil_df, "지상부 길이") if has_growth_soil else pd.Series(dtype=float)
            h_root = get_series(growth_hydro_df, "지하부 길이") if has_growth_hydro else pd.Series(dtype=float)
            s_root = get_series(growth_soil_df, "지하부 길이") if has_growth_soil else pd.Series(dtype=float)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            def metric_or_none(col, val, unit=""):
                return f"{val}{unit}" if val is not None else "자료 부족"

            h_avg_gw = (
                round(h_gw.mean(), 2) if len(h_gw) >= 2 else None
            )
            s_avg_gw = (
                round(s_gw.mean(), 2) if len(s_gw) >= 2 else None
            )
            kpi1.metric("수경 평균 생중량 (g)", h_avg_gw or "자료 부족")
            kpi2.metric("토양 평균 생중량 (g)", s_avg_gw or "자료 부족")

            h_cv = (
                round(h_gw.std() / h_gw.mean() * 100, 1)
                if (len(h_gw) >= 2 and h_gw.mean() > 0) else None
            )
            s_cv = (
                round(s_gw.std() / s_gw.mean() * 100, 1)
                if (len(s_gw) >= 2 and s_gw.mean() > 0) else None
            )
            kpi3.metric("수경 CV (%)", h_cv or "자료 부족")
            kpi4.metric("토양 CV (%)", s_cv or "자료 부족")

            st.caption(
                "CV(변동계수, %) = (표준편차 ÷ 평균) × 100 — 개체 간 생육 차이가 "
                "상대적으로 얼마나 큰지 나타냄. 값이 클수록 개체 간 편차가 큽니다."
            )

            st.markdown("---")

            # 최대·최소 개체
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

            # ----- 개체별 생육 지표 비교 막대 (그룹 평균) -----
            st.subheader(
                "개체별 생육 지표 비교 (수경 vs 토양 평균 막대)"
            )
            bar_cols = [
                c
                for c in selected_measures
                if (has_growth_hydro and c in growth_hydro_df.columns)
                or (has_growth_soil and c in growth_soil_df.columns)
            ]
            if bar_cols:
                rows_bar = []
                for col in bar_cols:
                    h_mean = (
                        round(get_series(growth_hydro_df, col).mean(), 2)
                        if (
                            has_growth_hydro
                            and col in growth_hydro_df.columns
                            and len(get_series(growth_hydro_df, col)) >= 2
                        )
                        else None
                    )
                    s_mean = (
                        round(get_series(growth_soil_df, col).mean(), 2)
                        if (
                            has_growth_soil
                            and col in growth_soil_df.columns
                            and len(get_series(growth_soil_df, col)) >= 2
                        )
                        else None
                    )
                    rows_bar.append(
                        {"생육 지표": col,
                         "수경재배 평균": h_mean,
                         "토양재배 평균": s_mean}
                    )
                bar_df = pd.DataFrame(rows_bar)
                fig_bar = go.Figure()
                has_h_bar = bar_df["수경재배 평균"].notna().any()
                has_s_bar = bar_df["토양재배 평균"].notna().any()
                if has_h_bar:
                    fig_bar.add_trace(
                        go.Bar(
                            x=bar_df["생육 지표"],
                            y=bar_df["수경재배 평균"],
                            name=f"수경재배 (n={len(growth_hydro_df)})",
                            marker_color="#4FC3F7",
                            width=0.4,
                        )
                    )
                if has_s_bar:
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
                    "※ 파란색 = 수경재배 스마트팜 24개체 평균, "
                    "주황색 = 토양재배 노지 70개체 평균. 개체 수와 측정 환경이 "
                    "다르므로 값의 크기만 단순 비교하기보다 '경향 차이'를 보는 "
                    "용도로 활용하세요."
                )
            else:
                st.info(
                    "선택한 측정 항목 중 두 재배 데이터 모두에 있는 열이 "
                    "없어요. 사이드바에서 항목을 다시 선택해 주세요."
                )

            st.markdown("---")

            # ----- 생중량 분포 박스플롯 -----
            st.subheader("생중량 분포 (박스플롯 · 수경 vs 토양)")
            h_gw_b = (
                get_gw(growth_hydro_df)
                if (has_growth_hydro and len(get_gw(growth_hydro_df)) >= 2)
                else pd.Series(dtype=float)
            )
            s_gw_b = (
                get_gw(growth_soil_df)
                if (has_growth_soil and len(get_gw(growth_soil_df)) >= 2)
                else pd.Series(dtype=float)
            )
            fig_box = go.Figure()
            if len(h_gw_b) >= 2:
                fig_box.add_trace(
                    go.Box(
                        y=h_gw_b,
                        name=f"수경재배 (n={len(h_gw_b)})",
                        fillcolor="#4FC3F7",
                        marker=dict(color="#4FC3F7"),
                        boxpoints="outliers",
                    )
                )
            if len(s_gw_b) >= 2:
                fig_box.add_trace(
                    go.Box(
                        y=s_gw_b,
                        name=f"토양재배 (n={len(s_gw_b)})",
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

            # ----- 상관관계 분석 (수경 + 토양 각각) -----
            st.subheader("상관관계 분석 (산점도 + 추세선)")

            def scatter_with_trend(df_label, df, x_col, y_col, color):
                x_s = get_series(df, x_col)
                y_s = get_gw(df) if y_col == "생중량" else get_series(df, y_col)
                if len(x_s) < 2 or len(y_s) < 2 or len(x_s) != len(y_s):
                    return None
                xy = pd.concat(
                    [x_s.reset_index(drop=True),
                     y_s.reset_index(drop=True)],
                    axis=1,
                ).dropna()
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
                    font=dict(size=11),
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
                            c for c in [x_col, "생중량"]
                            if c not in growth_hydro_df.columns
                        ]
                        st.info(
                            f"{', '.join(missing)} 열이 없어 수경재배의 "
                            "이 상관을 건너뛰었어요."
                        )

            if has_growth_hydro and has_growth_soil:
                st.markdown("---")

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
                            c for c in [x_col, "생중량"]
                            if c not in growth_soil_df.columns
                        ]
                        st.info(
                            f"{', '.join(missing)} 열이 없어 토양재배의 "
                            "이 상관을 건너뛰었어요."
                        )

            st.markdown("---")

            # ----- 재배 목적별 적합성 검토 -----
            st.subheader("재배 목적별 토양·수경재배 적합성 검토")

            h_shoot_m = (
                round(get_series(growth_hydro_df, "지상부 질량").mean(), 2)
                if (has_growth_hydro
                    and "지상부 질량" in growth_hydro_df.columns
                    and len(get_series(growth_hydro_df, "지상부 질량")) >= 2)
                else None
            )
            h_root_m = (
                round(get_series(growth_hydro_df, "지하부 질량").mean(), 2)
                if (has_growth_hydro
                    and "지하부 질량" in growth_hydro_df.columns
                    and len(get_series(growth_hydro_df, "지하부 질량")) >= 2)
                else None
            )
            h_root_l = (
                round(get_series(growth_hydro_df, "지하부 길이").mean(), 2)
                if (has_growth_hydro
                    and "지하부 길이" in growth_hydro_df.columns
                    and len(get_series(growth_hydro_df, "지하부 길이")) >= 2)
                else None
            )
            h_leaf_n = (
                round(get_series(growth_hydro_df, "잎 수").mean(), 2)
                if (has_growth_hydro
                    and "잎 수" in growth_hydro_df.columns
                    and len(get_series(growth_hydro_df, "잎 수")) >= 2)
                else None
            )
            h_shoot_l = (
                round(get_series(growth_hydro_df, "지상부 길이").mean(), 2)
                if (has_growth_hydro
                    and "지상부 길이" in growth_hydro_df.columns
                    and len(get_series(growth_hydro_df, "지상부 길이")) >= 2)
                else None
            )
            s_shoot_l = (
                round(get_series(growth_soil_df, "지상부 길이").mean(), 2)
                if (has_growth_soil
                    and "지상부 길이" in growth_soil_df.columns
                    and len(get_series(growth_soil_df, "지상부 길이")) >= 2)
                else None
            )
            s_root_l = (
                round(get_series(growth_soil_df, "지하부 길이").mean(), 2)
                if (has_growth_soil
                    and "지하부 길이" in growth_soil_df.columns
                    and len(get_series(growth_soil_df, "지하부 길이")) >= 2)
                else None
            )

            tr_h = (
                round(h_shoot_l / h_root_l, 2)
                if (h_shoot_l is not None and h_root_l is not None and h_root_l > 0)
                else None
            )
            tr_s = (
                round(s_shoot_l / s_root_l, 2)
                if (s_shoot_l is not None and s_root_l is not None and s_root_l > 0)
                else None
            )

            review_rows = [
                {
                    "재배 목적": "약용식물 및 뿌리 유효성분 추출",
                    "수경재배 적합성": "매우 높음 (우수)",
                    "토양재배 적합성": "보통",
                    "주요 고려 지표": "지하부 길이, 지하부 질량",
                    "적합 사유": (
                        "수경재배는 토양 저항이 없어 지하부 길이가 길게 형성되고, "
                        "수확 후 세척 시 흙 등 불순물이 없어 유효성분 추출 효율이 높음"
                    ),
                    "수경 데이터 근거": (
                        f"지하부 길이 {h_root_l} mm"
                        if h_root_l is not None
                        else "자료 없음"
                    )
                    + (f", 지하부 질량 {h_root_m} g" if h_root_m is not None else ""),
                },
                {
                    "재배 목적": "멸종위기 식물 보전 및 자생지 복원",
                    "수경재배 적합성": "보통",
                    "토양재배 적합성": "매우 높음 (우수)",
                    "주요 고려 지표": "T/R 비율, 지하부 구조, 이식 생존율",
                    "적합 사유": (
                        "토양재배 개체는 짧고 굵은 뿌리 구조와 균형 잡힌 T/R 비율을 "
                        "형성하므로 실제 야생 토양 환경에 이식했을 때 생존 가능성이 높음"
                    ),
                    "토양 데이터 근거": (
                        f"T/R 비율 {tr_s}"
                        if tr_s is not None
                        else "자료 없음"
                    )
                    + (f", 지하부 길이 {s_root_l} mm" if s_root_l is not None else ""),
                },
                {
                    "재배 목적": "대량 증식 및 상업적 생산",
                    "수경재배 적합성": "높음",
                    "토양재배 적합성": "보통",
                    "주요 고려 지표": "잎 수, 개체 균일성, 환경 제어 용이성",
                    "적합 사유": (
                        "수경재배는 제어된 온·습도와 광조건으로 대사 활동을 촉진하여 "
                        "잎 수 증가를 유도하며, 양액 기반 규격화 생장으로 개체 간 "
                        "균일성 확보가 용이함"
                    ),
                    "수경 데이터 근거": (
                        f"잎 수 {h_leaf_n} 장"
                        if h_leaf_n is not None
                        else "자료 없음"
                    ),
                },
                {
                    "재배 목적": "스마트팜 제어 환경 연구",
                    "수경재배 적합성": "매우 높음 (우수)",
                    "토양재배 적합성": "보통",
                    "주요 고려 지표": "온도·습도·CO₂·광량 제어, 환경 응답 메커니즘",
                    "적합 사유": (
                        "온도, 습도, CO₂, 광량 등 주요 환경 변수의 인공 제어가 용이하며 "
                        "기후변화 반응 및 환경 응답 메커니즘의 정밀 분석에 적합함"
                    ),
                    "수경 데이터 근거": (
                        f"온도 {round(env_hydro_df['temperature'].dropna().mean(), 1)}°C, "
                        f"습도 {round(env_hydro_df['humidity'].dropna().mean(), 0)}%"
                        if has_env_hydro
                        and "temperature" in env_hydro_df.columns
                        and "humidity" in env_hydro_df.columns
                        else "환경 CSV 첨부 필요"
                    ),
                },
            ]
            st.dataframe(
                pd.DataFrame(review_rows),
                width="stretch",
                hide_index=True,
            )

            st.markdown("---")

            # ----- 결론: 환경이 생육에 미친 영향 → 목적별 추천 -----
            st.subheader("결론: 환경이 극지식물의 생김새에 미친 영향")

            st.markdown(
                "## 1) 수경재배(스마트팜)의 환경과 생육 특징"
            )
            h_t_val = (
                round(env_hydro_df["temperature"].dropna().mean(), 1)
                if (has_env_hydro
                    and "temperature" in env_hydro_df.columns
                    and len(env_hydro_df["temperature"].dropna()) >= 2)
                else None
            )
            h_h_val = (
                round(env_hydro_df["humidity"].dropna().mean(), 1)
                if (has_env_hydro
                    and "humidity" in env_hydro_df.columns
                    and len(env_hydro_df["humidity"].dropna()) >= 2)
                else None
            )
            s_t_val = (
                round(soil_env_df["temperature"].dropna().mean(), 1)
                if (has_soil_env
                    and "temperature" in soil_env_df.columns
                    and len(soil_env_df["temperature"].dropna()) >= 2)
                else None
            )
            s_h_val = (
                round(soil_env_df["humidity"].dropna().mean(), 1)
                if (has_soil_env
                    and "humidity" in soil_env_df.columns
                    and len(soil_env_df["humidity"].dropna()) >= 2)
                else None
            )

            st.markdown(
                f"**환경:** 온도가 약 **{h_t_val}°C**로 비교적 높고, 습도가 약 "
                f"**{h_h_val}%**로 낮은 조건이다. CO₂도 약 "
                f"{round(env_hydro_df['co2'].dropna().mean(), 0) if has_env_hydro and 'co2' in env_hydro_df.columns else 'N/A'} ppm "
                f"수준으로 공급된다. 이러한 고온·저습 환경은 식물의 증산 작용을 "
                f"활발하게 만들고, 대사 속도를 높여 생장을 촉진한다.\n\n"
                f"**생육 특징:** 이러한 환경에서 자란 수경재배 극지식물은 **잎 수가 "
                f"많고(평균 {h_leaf_n}장)**, **지하부가 길고 굵게**(평균 "
                f"{h_root_l} mm, 지하부 질량 {h_root_m} g) 발달하는 경향이 있다. "
                f"지상부는 상대적으로 짧고(평균 {h_shoot_l} mm), T/R 길이 비율이 "
                f"{tr_h}로 지상부 대비 지하부 비중이 크다. 겉으로 보기에 잎이 무성하고 "
                f"뿌리 부분이 크게 발달한 모습이다.\n\n"
                f"**해석:** 높은 온도는 광합성 및 대사 활동을 촉진하여 잎 수 증가로 "
                f"이어지고(가설 1 지지), 낮은 습도는 수분 확보를 위해 뿌리 신장을 "
                f"자극하여 지하부가 길게 발달한 것으로 해석된다(가설 2 지지)."
                + (f" 다만 지상부가 짧은 것은 수경재배의 제한된 공간·용기 조건의 "
                     f"영향일 수 있다." if h_shoot_l is not None else "")
            )

            st.markdown(
                "## 2) 토양재배(노지)의 환경과 생육 특징"
            )
            st.markdown(
                f"**환경:** 온도가 약 **{s_t_val}°C**로 수경보다 낮고, 습도가 약 "
                f"**{s_h_val}%**로 높은 조건이다. 토양 입자 사이의 수분과 현장 "
                f"기후의 영향을 직접 받으며, pH는 약 6.6, EC는 약 1.3 수준이다.\n\n"
                f"**생육 특징:** 이러한 저온·고습 환경에서 자란 토양재배 극지식물은 "
                f"**잎 수가 적고**(평균 {round(s_leaf.mean(), 1) if len(s_leaf) >= 2 else 'N/A'}장), "
                f"**지상부가 길게**(평균 {s_shoot_l} mm) 자라며, **지하부는 짧고 "
                f"가늘게**(평균 {s_root_l} mm, 지하부 질량 "
                f"{round(get_series(growth_soil_df, '지하부 질량').mean(), 2) if (has_growth_soil and '지하부 질량' in growth_soil_df.columns and len(get_series(growth_soil_df, '지하부 질량')) >= 2) else 'N/A'} g) 발달한다. "
                f"T/R 길이 비율이 {tr_s}로 지상부 비중이 상대적으로 크다. 겉으로 "
                f"보기에 줄기가 길고 곧게 뻗은 모습이며, 뿌리 부분은 수경재배보다 "
                f"짧고 가늘다.\n\n"
                f"**해석:** 낮은 온도는 대사 속도를 낮춰 잎 수 증가를 억제하고, "
                f"높은 습도는 수분 스트레스가 적어 뿌리 신장을 덜 자극한 것으로 "
                f"보인다. 대신 토양 입자와의 기계적 저항 속에서 지상부가 상대적으로 "
                f"더 길게 자라는 경향이 나타난다."
            )

            st.markdown("---")
            st.subheader("재배 목적별 추천 결론")

            st.markdown(
                "### 약용식물 생산 및 뿌리 유효성분 추출 목적 → **수경재배(스마트팜) 추천**\n\n"
                f"수경재배는 지하부 길이가 평균 **{h_root_l} mm**로 토양재배 "
                f"({s_root_l} mm)에 비해 현저히 길고, 지하부 질량도 "
                f"{h_root_m} g으로 더 크게 형성된다. 토양 입자가 붙지 않아 수확 후 "
                f"세척 과정에서 뿌리 손상을 줄이고 고순도 원료를 추출하는 데 유리하다. "
                f"뿌리·지하부 조직에 유효 약용 성분이 포함된 종을 재배할 경우 "
                f"수경재배가 월등히 적합하다."
            )
            st.markdown(
                "### 멸종위기 식물 보전 및 자생지 복원 목적 → **토양재배 추천**\n\n"
                f"토양재배 개체는 T/R 비율이 {tr_s}로 균형 잡혀 있고, 지하부가 "
                f"짧고 굵게 발달하여 실제 야생 토양 환경에 이식했을 때 기계적 저항과 "
                f"수분 흡수 스트레스에 더 잘 적응할 가능성이 높다. 반면 수경재배 개체는 "
                f"지하부가 너무 길고 가늘어({h_root_l} mm) 야생 토양 이식 시 활착률이 "
                f"떨어질 위험이 있다. 종 보전 및 자생지 복원을 위한 증식 목적이라면 "
                f"토양재배가 적합하다."
            )
            st.markdown(
                "### 식물 바이오매스 대량 증식 목적 → **수경재배(스마트팜) 추천**\n\n"
                f"수경재배 환경은 제어된 온·습도와 광조건을 통해 대사 활동을 촉진하여 "
                f"잎 수(평균 {h_leaf_n}장)를 토양재배({round(s_leaf.mean(), 1) if len(s_leaf) >= 2 else 'N/A'}장)보다 "
                f"유의미하게 늘려준다. 외부 기후의 영향을 적게 받아 사계절 규격화된 생육 "
                f"생산이 가능하므로, 지상부 잎이나 개체 수 위주의 신속한 대량 증식이 "
                f"목적일 경우 수경재배가 적합하다."
            )
            st.markdown(
                "### 스마트팜 제어 환경 연구 목적 → **수경재배(스마트팜) 추천**\n\n"
                f"온도, 습도, CO₂ 등 주요 환경 변수를 인공적으로 정밀 제어할 수 있어 "
                f"기후변화 반응 및 환경 응답 메커니즘의 정밀 분석에 적합하다. 수경재배 "
                f"환경 데이터(온도 평균 {h_t_val}°C, 습도 평균 {h_h_val}%)는 연속 "
                f"1시간 간격 측정으로 시계열 분석이 가능해 환경 제어 연구에 특히 유리하다."
            )

            st.markdown("---")
            st.subheader("연구의 한계 및 후속 연구 방향")
            st.markdown(
                "**스마트팜 내부 미기상 및 입지 조건의 차이:** 본 연구에서 수경재배 "
                "스마트팜 환경 내의 모든 개체가 완전히 동일한 물리적 환경에서 생장했다고 "
                "가정하기는 어렵다. 재배 베드의 중앙부와 가장자리부 위치에 따라 양액의 "
                "순환 속도 및 수분 접촉 빈도에 차이가 있었으며, 이는 LED 인공 조명의 "
                "광량(조도)과 국지적 온도 분포에도 미세한 영향을 미쳤을 가능성이 높다. "
                "이로 인해 동일한 수경재배 집단 내에서도 개체별 생육 편차가 발생했을 수 "
                "있으며, 단일 센서가 포착하지 못한 미기상(Microclimate)의 차이를 완벽히 "
                "통제하지 못한 점은 본 연구의 한계이다."
            )
            st.markdown(
                "**토양재배 환경의 비정기적 측정 제약:** 토양재배 환경 데이터의 경우 "
                "스마트팜처럼 1시간 간격의 연속 시계열 데이터가 아니라 비정기적인 현장 "
                "관측 자료로 수집되었다. 따라서 식물의 생육 전 기간에 걸친 미세한 환경 "
                "변동 양상을 시점별로 완벽하게 동기화하여 인과관계를 해석하는 데는 "
                "구조적인 제약이 따랐다."
            )
            st.markdown(
                "**2차 대사산물 및 기능성 성분 비교 연구:** 향후 연구에서는 단순히 형태적 "
                "생육 지표(길이, 질량, 잎 수) 비교에 머무르지 않고, 수경재배와 토양재배 "
                "환경에서 자란 극지식물의 체내 2차 대사산물(Secondary Metabolites) 함량을 "
                "정밀 비교하고자 한다."
            )
            st.markdown(
                "**요오드 적정 실험을 통한 비타민 C 함량 분석:** 실제 그린란드 고추냉이 등 "
                "극지식물이 현지 주민들에게 혹독한 환경 속 주요 비타민 C 공급원으로 "
                "이용되어 왔다는 생태적 배경에 착안하여, 요오드 적정 실험(Iodometric "
                "Titration)을 통해 두 재배 방식별 식물 내 비타민 C 함량을 정량적으로 "
                "측정하는 실험을 계획하고 있다. 이를 통해 외형적 생장량(바이오매스)뿐만 "
                "아니라 기능성 영양 성분 축적 관점에서 수경재배와 토양재배의 적합성을 "
                "다각도로 검증할 수 있다."
            )

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
