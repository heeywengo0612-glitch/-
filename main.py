import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
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
    """파일 바이트를 직접 읽어서 인코딩 순차 시도"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ============================================================
# 열 이름 표준화 (KeyError 방지, 학교별 차이 흡수)
# ============================================================
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        # --- 공통 시간 열 ---
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        # --- 온도 / 습도 ---
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        # --- pH / EC ---
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        # --- CO2 ---
        elif k == "co2":
            rename[c] = "co2"
        elif "air" in k and "sensor" in k and "co2" in k:
            rename[c] = "co2"
        elif "air" in k and "sensor" in k and "1" in k:
            rename[c] = "temperature"
        elif "air" in k and "sensor" in k and "2" in k:
            rename[c] = "humidity"
        # --- 생육 열 ---
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
    # 숫자 아닌 열은 숫자 변환 시도 (글자 열은 그대로 유지)
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
    # 생중량 없으면 지상부+지하부 질량으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    # 시간 열 파싱 (ISO 형식 + '날짜_시각' 형식 모두 처리)
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(
                parts[0], format="%Y%m%d"
            ) + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


# ============================================================
# 사이드바
# ============================================================
st.sidebar.title("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 생육 측정 항목 선택")
growth_cols_default = [
    "잎 수",
    "지상부 길이",
    "지하부 길이",
    "생중량",
]
selected_measures = st.sidebar.multiselect(
    "비교할 항목 선택",
    options=growth_cols_default,
    default=growth_cols_default,
)

st.sidebar.markdown("---")
st.sidebar.caption("ℹ️ 각 CSV는 utf-8-sig / cp949 순서로 읽습니다.")
st.sidebar.caption("ℹ️ 업로드 전에는 탭 내용이 보이지 않습니다.")


# ============================================================
# 데이터 로딩 (@st.cache_data + spinner)
# ============================================================
@st.cache_data
def load_env(file):
    return std_cols(read_csv(file))


@st.cache_data
def load_growth(file):
    return std_cols(read_csv(file))


env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중..."):
        try:
            env_df = load_env(env_file)
        except Exception:
            env_df = pd.DataFrame()

if growth_file is not None:
    with st.spinner("생육 데이터를 읽는 중..."):
        try:
            growth_df = load_growth(growth_file)
        except Exception:
            growth_df = pd.DataFrame()

# 학교명 추출 (파일명 앞부분에서 읽기, 하드코딩 금지)
school_name = "우리 학교"
if env_file is not None:
    fn = os.path.basename(env_file.name)
    school_name = fn.split("_")[0] if "_" in fn else fn.split(".")[0]
elif growth_file is not None:
    fn = os.path.basename(growth_file.name)
    school_name = fn.split("_")[0] if "_" in fn else fn.split(".")[0]


# ============================================================
# 페이지 구성
# ============================================================
st.title(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")
st.markdown(
    "*[평균]*: 모든 개체를 더한 뒤 개체 수로 나눈 값  "
    "*[표준편차]*: 개체들이 평균에서 얼마나 퍼져 있는지 나타내는 값  "
    "*[상관계수]*: 두 값이 함께 변하는 정도를 -1~1 사이로 나타낸 값  "
    "*[CV(변동계수)]* : 표준편차를 평균으로 나눈 값, 단위가 다른 때도 비교 가능",
)

tab1, tab2, tab3 = st.tabs(
    ["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"]
)


# ============================================================
# Tab 1: 실험 개요
# ============================================================
with tab1:
    try:
        st.subheader("연구 배경 및 목적")
        st.markdown(
            f"**{school_name}**에서 키운 극지식물 개체를 서로 비교하여, "
            "개체별로 생육 상태에 어떤 차이가 있는지 살펴봅니다. "
            "평균·표준편차·상관계수 같은 기본 통계를 이용해, "
            "어떤 개체가 잘 자랐는지, 어떤 환경 조건이 생육과 관련 있는지 "
            "직관적으로 확인하는 것이 목적입니다.",
        )

        st.subheader("데이터 요약")
        # 요약표 행 생성 (rows.append로 한 줄씩)
        rows = []
        n_env = len(env_df)
        n_growth = len(growth_df)
        has_env = n_env > 0
        has_growth = n_growth > 0

        rows.append(
            {
                "항목": "총 개체 수",
                "값": str(n_growth) if has_growth else "자료 없음",
            }
        )
        if has_env and "time" in env_df.columns:
            t_min = env_df["time"].min()
            t_max = env_df["time"].max()
            rows.append(
                {
                    "항목": "측정 기간",
                    "값": f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}",
                }
            )
        else:
            rows.append({"항목": "측정 기간", "값": "자료 없음"})

        measure_items = []
        if has_growth:
            for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
                if c in growth_df.columns:
                    measure_items.append(c)
        rows.append(
            {
                "항목": "생육 측정 항목",
                "값": ", ".join(measure_items) if measure_items else "자료 없음",
            }
        )
        for c in ["temperature", "humidity", "ph", "ec", "co2"]:
            if has_env and c in env_df.columns:
                rows.append({"항목": f"환경 측정 '{c}'", "값": "있음"})
        if has_env and not any(
            c in env_df.columns for c in ["temperature", "humidity", "ph", "ec", "co2"]
        ):
            rows.append({"항목": "환경 측정 항목", "값": "인식된 열 없음"})

        if rows:
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, width="stretch", hide_index=True)
        else:
            st.info("표시할 데이터 요약이 없습니다. CSV를 업로드해 주세요.")

        # 주요 지표 카드 4개
        st.subheader("주요 지표")
        col1, col2, col3, col4 = st.columns(4)

        if has_growth and "생중량" in growth_df.columns:
            gw = growth_df["생중량"].dropna()
            if len(gw) >= 2:
                avg_gw = gw.mean()
                col4.metric("평균 생중량 (g)", round(avg_gw, 2))
            else:
                col4.metric("평균 생중량 (g)", "자료 부족")
        else:
            col4.metric("평균 생중량 (g)", "자료 없음")

        if has_env and "temperature" in env_df.columns:
            t = env_df["temperature"].dropna()
            if len(t) >= 2:
                col1.metric("평균 온도 (°C)", round(t.mean(), 1))
            else:
                col1.metric("평균 온도 (°C)", "자료 부족")
        else:
            col1.metric("평균 온도 (°C)", "자료 없음")

        if has_env and "humidity" in env_df.columns:
            h = env_df["humidity"].dropna()
            if len(h) >= 2:
                col2.metric("평균 습도 (%)", round(h.mean(), 1))
            else:
                col2.metric("평균 습도 (%)", "자료 부족")
        else:
            col2.metric("평균 습도 (%)", "자료 없음")

        col3.metric("총 개체 수", n_growth if has_growth else 0)

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# Tab 2: 환경 데이터
# ============================================================
with tab2:
    try:
        if len(env_df) == 0:
            st.info("환경 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            # --- 환경 요약 표 ---
            st.subheader("환경 데이터 요약 (평균 · 최소 · 최대)")
            env_num_cols = [
                c for c in env_df.columns if c not in ("time", "개체번호")
            ]
            env_num_cols = [
                c for c in env_num_cols if pd.api.types.is_numeric_dtype(env_df[c])
            ]
            if env_num_cols:
                env_valid = env_df[env_num_cols].dropna()
                if len(env_valid) >= 2:
                    summary_dict = {}
                    for c in env_num_cols:
                        s = env_valid[c]
                        summary_dict[c] = {
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                        }
                    st.dataframe(
                        pd.DataFrame(summary_dict).T,
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("환경 데이터 값이 충분하지 않아 요약을 표시하지 못했습니다.")
            else:
                st.info("환경 데이터에서 숫자 열을 찾을 수 없어요.")

            # --- 날짜별 시계열 ---
            st.subheader("날짜별 환경 변화 (일별 평균)")
            if "time" in env_df.columns:
                env_daily = (
                    env_df.copy()
                    if "time" in env_df.columns
                    else pd.DataFrame()
                )
                env_daily["date"] = env_daily["time"].dt.date
                date_cols = [
                    c
                    for c in ["temperature", "humidity", "ph", "ec", "co2"]
                    if c in env_daily.columns
                ]
                for dc in date_cols:
                    daily = (
                        env_daily.groupby("date")[dc]
                        .mean()
                        .dropna()
                        .reset_index()
                    )
                    if len(daily) >= 2:
                        unit = ""
                        if dc == "temperature":
                            unit = "°C"
                        elif dc == "humidity":
                            unit = "%"
                        elif dc == "co2":
                            unit = "ppm"
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=daily["date"],
                                y=daily[dc],
                                mode="lines+markers",
                                name=f"{dc} ({unit})",
                                line=dict(width=2),
                            )
                        )
                        fig.update_layout(
                            title=f"{dc} 일별 평균 변화",
                            xaxis_title="날짜",
                            yaxis_title=f"{dc} ({unit})",
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"{dc} 데이터가 충분하지 않아 그래프를 건너뛰었어요.")
                if not date_cols:
                    st.info(
                        "온도·습도·pH·EC·CO2 중 인식된 열이 없어 시계열을 표시하지 못했어요."
                    )
            else:
                st.info("환경 데이터에 'time' 열이 없어 시계열을 표시하지 못했어요.")

            # --- 원본 테이블 + 다운로드 ---
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# Tab 3: 생육 결과
# ============================================================
with tab3:
    try:
        if len(growth_df) == 0:
            st.info("생육 결과 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            # --- 🥇 핵심 결과 카드 ---
            st.subheader("🥇 핵심 결과")
            has_gw = "생중량" in growth_df.columns
            gw_valid = growth_df["생중량"].dropna() if has_gw else pd.Series(dtype=float)

            kpi1, kpi2, kpi3 = st.columns(3)
            if has_gw and len(gw_valid) >= 2:
                avg_gw = gw_valid.mean()
                std_gw = gw_valid.std()
                cv_gw = (std_gw / avg_gw * 100) if avg_gw > 0 else 0
                kpi1.metric("평균 생중량 (g)", round(avg_gw, 2))
                kpi2.metric("표준편차 (g)", round(std_gw, 2))
                kpi3.metric("CV (변동계수, %)", round(cv_gw, 1))
            else:
                kpi1.metric("평균 생중량 (g)", "자료 없음")
                kpi2.metric("표준편차 (g)", "자료 없음")
                kpi3.metric("CV (변동계수, %)", "자료 없음")

            st.markdown("---")

            # 최대·최소 개체
            if has_gw and len(gw_valid) >= 2:
                gw_idxmax = gw_valid.idxmax()
                gw_idxmin = gw_valid.idxmin()
                max_row = growth_df.loc[gw_idxmax]
                min_row = growth_df.loc[gw_idxmin]
                max_id = max_row["개체번호"]
                min_id = min_row["개체번호"]
                max_val = max_row["생중량"]
                min_val = min_row["생중량"]
                st.info(
                    f"🏆 생중량 **최대** 개체: #{max_id}  ({round(max_val, 2)} g)   |   "
                    f"📉 생중량 **최소** 개체: #{min_id}  ({round(min_val, 2)} g)"
                )

            # --- 개체별 생육 비교 (막대 그래프) ---
            st.subheader("개체별 생육 비교 (막대 그래프)")
            bar_cols = [c for c in selected_measures if c in growth_df.columns]
            if bar_cols:
                n_cols = min(len(bar_cols), 2)
                fig = make_subplots(
                    rows=(len(bar_cols) + 1) // 2,
                    cols=n_cols,
                    shared_xaxes=True,
                    subplot_titles=bar_cols,
                )
                for i, col in enumerate(bar_cols):
                    row_n = i // n_cols + 1
                    col_n = i % n_cols + 1
                    valid = growth_df[[ "개체번호", col ]].dropna()
                    if len(valid) >= 2:
                        fig.add_trace(
                            go.Bar(
                                x=valid["개체번호"].astype(str),
                                y=valid[col],
                                name=col,
                                marker_color="#4CAF50",
                            ),
                            row=row_n,
                            col=col_n,
                        )
                fig.update_layout(
                    height=300 * ((len(bar_cols) + 1) // 2),
                    font=dict(
                        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                    ),
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info(
                    "선택한 측정 항목 중 생육 데이터에 있는 열이 없어요. "
                    "사이드바에서 항목을 다시 선택해 주세요."
                )

            # --- 생중량 분포 (박스플롯) ---
            st.markdown("---")
            st.subheader("생중량 분포 (박스플롯)")
            if has_gw and len(gw_valid) >= 2:
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        y=growth_df["생중량"].dropna(),
                        name="생중량",
                        fillcolor="#81C784",
                        boxpoints="outliers",
                        marker=dict(color="#81C784"),
                    )
                )
                avg_val = gw_valid.mean()
                fig.add_hline(
                    y=avg_val,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f"평균 {round(avg_val, 2)} g",
                    annotation_position="top right",
                )
                fig.update_layout(
                    title="개체별 생중량 분포 (빨간 점선 = 평균)",
                    yaxis_title="생중량 (g)",
                    font=dict(
                        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                    ),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("생중량 열이 없어 분포를 표시하지 못했어요.")

            # --- 상관관계 분석 ---
            st.markdown("---")
            st.subheader("상관관계 분석 (산점도 + 추세선)")
            corr_pairs = [
                ("잎 수", "생중량"),
                ("지상부 길이", "생중량"),
            ]
            for x_col, y_col in corr_pairs:
                if x_col in growth_df.columns and y_col in growth_df.columns:
                    df_xy = growth_df[[x_col, y_col]].dropna()
                    if len(df_xy) >= 2:
                        x = df_xy[x_col].values
                        y = df_xy[y_col].values
                        r = np.corrcoef(x, y)[0, 1]
                        # 추세선: np.polyfit
                        slope, intercept = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        y_line = slope * x_line + intercept
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=y,
                                mode="markers",
                                name=f"{x_col} vs {y_col}",
                                marker=dict(size=8, color="#4FC3F7"),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode="lines",
                                name=f"추세선 (r={round(r, 2)})",
                                line=dict(color="red", width=2, dash="dot"),
                            )
                        )
                        fig.update_layout(
                            title=(
                                f"{x_col} vs {y_col}  |  "
                                f"상관계수 r = {round(r, 3)} "
                                f"({'강한 양의 상관' if r > 0.5 else '약한 관계' if abs(r) < 0.3 else '중간 상관'})"
                            ),
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                            margin=dict(l=40, r=20, t=50, b=40),
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(
                            f"{x_col}와 {y_col} 데이터가 충분하지 않아 상관을 표시하지 못했어요."
                        )
                else:
                    missing = [c for c in [x_col, y_col] if c not in growth_df.columns]
                    st.info(f"{', '.join(missing)} 열이 없어 이 상관을 건너뛰었어요.")

            # --- 상위 25% vs 하위 25% ---
            st.markdown("---")
            if has_gw and len(gw_valid) >= 4:
                q75 = gw_valid.quantile(0.75)
                q25 = gw_valid.quantile(0.25)
                top_mask = growth_df["생중량"] >= q75
                bot_mask = growth_df["생중량"] <= q25
                top_df = growth_df[top_mask]
                bot_df = growth_df[bot_mask]
                cmp_rows = []
                cmp_rows.append(
                    {
                        "그룹": "생중량 상위 25%",
                        "개체 수": len(top_df),
                        "평균 생중량 (g)": round(top_df["생중량"].mean(), 2),
                        "평균 잎 수": (
                            round(top_df["잎 수"].mean(), 1)
                            if "잎 수" in top_df.columns
                            else "없음"
                        ),
                        "평균 지상부 길이 (mm)": (
                            round(top_df["지상부 길이"].mean(), 1)
                            if "지상부 길이" in top_df.columns
                            else "없음"
                        ),
                    }
                )
                cmp_rows.append(
                    {
                        "그룹": "생중량 하위 25%",
                        "개체 수": len(bot_df),
                        "평균 생중량 (g)": round(bot_df["생중량"].mean(), 2),
                        "평균 잎 수": (
                            round(bot_df["잎 수"].mean(), 1)
                            if "잎 수" in bot_df.columns
                            else "없음"
                        ),
                        "평균 지상부 길이 (mm)": (
                            round(bot_df["지상부 길이"].mean(), 1)
                            if "지상부 길이" in bot_df.columns
                            else "없음"
                        ),
                    }
                )
                st.subheader("생중량 상위 25% vs 하위 25% 개체 비교")
                st.dataframe(
                    pd.DataFrame(cmp_rows),
                    width="stretch",
                    hide_index=True,
                )

            # --- 원본 테이블 + 다운로드 ---
            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
