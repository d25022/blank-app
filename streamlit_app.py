# streamlit_app.py
# -*- coding: utf-8 -*-
# =========================================================
# 📊 보고서 맞춤 대시보드 (민트+브라운 테마)
# ---------------------------------------------------------
# 구조:
#   1) 🥠 포춘쿠키: 버튼을 눌러 랜덤 실천 카드 보기(6개 이상)
#   2) 🔬 데이터 관측실: 더미 데이터 기반 라인/막대 상호작용(지표/지역/연도 범위)
#   3) 🗂️ 자료실: 출처/참고 링크 모음 (클릭 열람)
#
# 폰트: /fonts/Pretendard-Bold.ttf (있으면 적용, 없으면 생략)
# 표준화: date, value, group
# 전처리: 결측/형변환/중복 제거/미래(로컬 자정 이후) 제거
# 캐싱: @st.cache_data
# 내보내기: 관측실의 전처리된 표 CSV 다운로드 제공
#
# ※ 공개 데이터 실제 호출은 본 앱에서는 하지 않습니다(시연용 더미 데이터).
#    공개 데이터 연결이 필요한 경우: NASA/NOAA/World Bank 등 API를 동일 스키마로 붙이면 됩니다.
# =========================================================

import os
from datetime import datetime
from typing import Tuple, Optional, List
from dateutil import tz

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
from matplotlib import font_manager

APP_TITLE = "📊 내일은 물 위의 학교? — 인터랙티브 보고서 대시보드"
LOCAL_TZ = tz.gettz("Asia/Seoul")

# ---------------------------
# 폰트 적용 시도 (Pretendard)
# ---------------------------
def _try_set_pretendard() -> None:
    try:
        font_path = "/fonts/Pretendard-Bold.ttf"
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            matplotlib.rcParams["font.family"] = "Pretendard"
        st.session_state.setdefault("base_font_family", "Pretendard")
    except Exception:
        st.session_state.setdefault("base_font_family", "sans-serif")

_try_set_pretendard()

# ---------------------------
# 민트+브라운 테마 CSS
# ---------------------------
THEME_MINT = "#2dd4bf"   # 민트
THEME_BROWN = "#8b5e34"  # 브라운
THEME_BG = "#fbf8f5"     # 따뜻한 베이지 배경
THEME_CARD = "#ffffff"   # 카드 배경

_CSS = f"""
<style>
:root {{
  --mint: {THEME_MINT};
  --brown: {THEME_BROWN};
  --bg: {THEME_BG};
  --card: {THEME_CARD};
}}
html, body, .block-container {{ background-color: var(--bg); }}
.block-container {{padding-top: 0.8rem; padding-bottom: 1.0rem;}}
h1, h2, h3, h4 {{ color: var(--brown); margin-bottom: .5rem; }}
hr {{ margin: .6rem 0 .9rem 0; border-color: #e2e8f0; }}
.card {{
  background: var(--card);
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.03);
}}
.badge {{
  display:inline-block; padding:2px 8px; border-radius: 999px;
  background: var(--mint); color: #064e3b; font-weight:700; font-size:.75rem;
}}
.small {{ color:#6b7280; font-size:.9rem; }}
.mini {{ color:#666; font-size:.85rem; }}
.stButton>button, .stDownloadButton>button {{
  border-radius: 12px;
  border: 1px solid var(--brown);
  background: var(--mint);
  color: #064e3b;
  font-weight: 700;
}}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------
# 공통 유틸
# ---------------------------
def truncate_future_rows(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """로컬 자정 이후 데이터 제거"""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        now_local = datetime.now(LOCAL_TZ)
        today_midnight = datetime(now_local.year, now_local.month, now_local.day, tzinfo=LOCAL_TZ)
        cutoff = today_midnight.replace(tzinfo=None)
        out = out[out[date_col] < cutoff]
    return out.dropna().drop_duplicates()

def style_plot(fig: go.Figure, title: str, xlab: str, ylab: str) -> go.Figure:
    base_font = st.session_state.get("base_font_family", "sans-serif")
    fig.update_layout(
        title=title,
        font=dict(family=base_font, size=14, color=THEME_BROWN),
        xaxis_title=xlab,
        yaxis_title=ylab,
        hovermode="x unified",
        margin=dict(l=40, r=18, t=50, b=36),
        paper_bgcolor=THEME_BG,
        plot_bgcolor="#ffffff",
        colorway=[THEME_MINT, THEME_BROWN, "#0ea5e9", "#65a30d"],
        legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1)
    )
    return fig

# ---------------------------
# 더미 데이터 생성기 (지표/지역별)
# ---------------------------
@st.cache_data(show_spinner=False)
def make_dummy_series(kind: str, region: str, year_start: int = 1990, year_end: int = 2025) -> pd.DataFrame:
    """지표(kind)와 지역(region)에 따라 그럴듯한 시계열 더미 생성"""
    rng = pd.date_range(f"{year_start}-01-01", f"{year_end}-12-01", freq="MS")
    base = np.linspace(0, 1, len(rng))
    noise = np.random.default_rng(42).normal(0, 0.05, len(rng))

    # 지표별 스케일/추세
    if kind == "해수면":
        trend = 30 * base    # mm 상승 가정
        seasonal = 2.0 * np.sin(np.linspace(0, 8*np.pi, len(rng)))
        values = 10 + trend + seasonal + noise*5
        unit = "mm"
    elif kind == "해수온":
        trend = 0.6 * base   # ℃ 상승 가정
        seasonal = 0.15 * np.sin(np.linspace(0, 12*np.pi, len(rng)))
        values = 0.2 + trend + seasonal + noise*0.5
        unit = "℃"
    else:  # 폭염일수
        trend = 10 * base
        seasonal = 2.5 * np.sin(np.linspace(0, 6*np.pi, len(rng)))
        values = 2 + trend + seasonal + noise*3
        values = np.clip(values, 0, None)
        unit = "일"

    # 지역 보정
    if region == "대한민국":
        values = values * 1.1 + 1.0
    else:  # 세계 평균
        values = values

    df = pd.DataFrame({
        "date": rng,
        "value": values,
        "group": f"{region}·{kind}({unit})"
    })
    df = truncate_future_rows(df, "date").sort_values("date")
    return df

@st.cache_data(show_spinner=False)
def make_dummy_bar(kind: str, region: str) -> pd.DataFrame:
    """막대용 카테고리 더미(최근 연도 기준)"""
    cats = {
        "해수면": ["침식·방파제 보강", "내륙침수 대비", "연안관리 예산", "주거이동 지원"],
        "해수온": ["어장 변화", "산호 백화", "해양열파", "연안 생태"],
        "폭염일수": ["냉방부하", "야외활동 제한", "열 관련 질환", "전력피크"]
    }
    base = np.array([40, 55, 30, 35], dtype=float)
    if region == "대한민국": base = base * 1.1
    if kind == "해수온": base = base * [0.9, 1.2, 1.3, 1.0]
    if kind == "해수면": base = base * [1.3, 1.1, 1.0, 1.2]
    if kind == "폭염일수": base = base * [1.4, 1.2, 1.3, 1.5]
    return pd.DataFrame({"항목": cats.get(kind, []), "비율(%)": np.round(base, 1)})

def bar_percent(df: pd.DataFrame, horizontal: bool = True, title: str = "") -> go.Figure:
    if horizontal:
        fig = px.bar(df, x="비율(%)", y="항목", orientation="h", text="비율(%)")
    else:
        fig = px.bar(df, x="항목", y="비율(%)", text="비율(%)")
    fig.update_traces(textposition="outside", marker_line_color=THEME_BROWN, marker_line_width=1.2)
    return style_plot(fig, title, "비율(%)" if horizontal else "항목", "항목" if horizontal else "비율(%)")

# ---------------------------
# 상단 보고서 개요(소제목 이모티콘/클릭 요약)
# ---------------------------
def report_overview():
    st.markdown("### 📌 보고서 개요")
    with st.expander("🌊 해수면 상승의 현실"):
        st.markdown("바다는 거대한 열 저장소이며, 온난화로 인해 평균 해수면이 서서히 상승하고 있어요.")
    with st.expander("🧊 뜨거워지는 지구, 녹아내리는 빙하"):
        st.markdown("해수온 상승과 빙하 융해는 해수면 상승의 두 축입니다. 바다는 열팽창으로도 높아져요.")
    with st.expander("💨 온실가스와 기후 경고"):
        st.markdown("온실가스가 많아질수록 에너지가 지구에 머물고, 극한기상 빈도가 늘어납니다.")
    with st.expander("📚 청소년과 미래 세대의 위기"):
        st.markdown("폭염·침수·해충 증가 등은 학습·건강·정서에 영향을 줍니다.")
    with st.expander("🌱 우리가 만들 해답, 우리의 실천"):
        st.markdown("교실 26℃ 유지, 칼환기, 에너지 절약, 데이터 기반 제안으로 변화를 이끌 수 있어요.")

# ---------------------------
# (1) 포춘쿠키 탭
# ---------------------------
FORTUNES = [
    "교실 1°C 낮추기: 오후 블라인드 내리기",
    "쉬는 시간 2분 칼환기: 앞·뒤 창문 활짝!",
    "빈 교실 전원 OFF: 프로젝터·모니터 확인",
    "냉방은 26°C, 선풍기와 병행",
    "물병 챙기기: 열 스트레스 줄이기",
    "그늘길 동선 짜기: 햇빛 강한 시간 피하기",
    "우리 반 에너지 지킴이 지정하기",
    "기후 데이터 한 장 공유: 오늘의 한 그래프",
    "옥상 차열 페인트 제안서 데이터 붙이기",
    "운동장 그늘막 설치 서명받기",
]

def fortune_cookie_tab():
    st.markdown("### 🥠 포춘쿠키 — 오늘의 실천 한 가지")
    st.markdown('<div class="small">버튼을 눌러 오늘 바로 할 수 있는 짧고 구체적인 실천을 받아보세요.</div>', unsafe_allow_html=True)

    if "fortune" not in st.session_state:
        st.session_state["fortune"] = np.random.choice(FORTUNES)

    if st.button("포춘쿠키 열기 🍪"):
        st.session_state["fortune"] = np.random.choice(FORTUNES)

    st.markdown(
        f"""
        <div class="card">
          <span class="badge">오늘의 실천</span>
          <h4 style="margin:.4rem 0 0 0; color:var(--brown)">{st.session_state['fortune']}</h4>
          <p class="mini" style="margin:.4rem 0 0 0;">작은 행동이 모이면 교실이 달라집니다. 💚🤎</p>
        </div>
        """, unsafe_allow_html=True
    )

# ---------------------------
# (2) 데이터 관측실 탭
# ---------------------------
def data_lab_tab():
    st.markdown("### 🔬 데이터 관측실 — 지표·지역·기간을 바꿔보세요")
    st.markdown('<div class="small">※ 시연용 더미 데이터입니다. 실제 연결 시 NOAA/NASA/정부 공개 데이터로 교체하세요.</div>', unsafe_allow_html=True)

    colc1, colc2, colc3 = st.columns([1,1,2])
    with colc1:
        kind = st.selectbox("지표 선택", ["해수면", "해수온", "폭염일수"], index=0)
    with colc2:
        region = st.selectbox("지역 선택", ["대한민국", "세계 평균"], index=0)
    with colc3:
        yr = st.slider("표시 연도 범위", min_value=1980, max_value=2025, value=(1995, 2025))

    # 시계열 생성/필터
    df = make_dummy_series(kind, region, 1980, 2025)
    df = df[(df["date"].dt.year >= yr[0]) & (df["date"].dt.year <= yr[1])].copy()

    # 라인(이동평균 옵션)
    win = st.slider("스무딩(이동평균, 개월)", 1, 24, 12)
    df["MA"] = df.sort_values("date")["value"].rolling(win, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], name="월별", opacity=0.35, line=dict(color=THEME_BROWN)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA"], name=f"{win}개월 평균", line=dict(width=3, color=THEME_MINT)))
    st.plotly_chart(style_plot(fig, f"{df['group'].iloc[0]} — 시계열", "날짜", "값"), use_container_width=True)

    # 한 줄 요약 캡션
    if not df.empty:
        v0, v1 = df["MA"].iloc[0], df["MA"].iloc[-1]
        diff = v1 - v0
        arrow = "⬆️" if diff > 0 else ("⬇️" if diff < 0 else "➡️")
        st.caption(f"요약: {yr[0]}–{yr[1]} 기간 동안 {df['group'].iloc[0]}은(는) {arrow} {diff:+.2f} 변화했습니다.")

    # 막대(카테고리 영향)
    bar_df = make_dummy_bar(kind, region)
    st.plotly_chart(bar_percent(bar_df, horizontal=True, title=f"{region} {kind} 관련 영향도(시연값, %)"),
                    use_container_width=True)

    # CSV 다운로드(전처리된 관측 데이터)
    st.download_button(
        "CSV 다운로드(관측실 데이터)",
        data=df[["date","value","group","MA"]].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"observatory_{kind}_{region}_{yr[0]}_{yr[1]}.csv",
        mime="text/csv"
    )

# ---------------------------
# (3) 자료실 탭
# ---------------------------
SOURCES = [
    ("데이터 가이드", "NOAA(미 해양대기청) 포털", "https://www.noaa.gov/"),
    ("데이터 가이드", "NASA Climate", "https://climate.nasa.gov/"),
    ("정부 보고", "해양수산부: 우리나라 해수면 상승 현황", "https://www.mof.go.kr/doc/ko/selectDoc.do?docSeq=44140"),
    ("참고 아틀라스", "National Atlas(해수면 영향)", "http://nationalatlas.ngii.go.kr/pages/page_3813.php"),
    ("뉴스 사례", "연합뉴스: 베트남 농작물 피해", "http://yna.co.kr/view/AKR20240318090400076"),
    ("실천 아이디어", "기후행동연구소/청소년 실천 기사", "https://climateaction.re.kr/news01/180492"),
]

def library_tab():
    st.markdown("### 🗂️ 자료실 — 출처/참고 링크 모음")
    st.markdown('<div class="small">클릭하면 새 창으로 열립니다. 수업·보고서에 활용하세요.</div>', unsafe_allow_html=True)
    for kind, title, url in SOURCES:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:8px;">
              <span class="badge">{kind}</span>
              <div style="font-size:18px; font-weight:700; margin-top:6px;">🔗 <a href="{url}" target="_blank">{title}</a></div>
              <div class="mini">URL: {url}</div>
            </div>
            """, unsafe_allow_html=True
        )

# ---------------------------
# 상단 보고서 본문(서론/본론/결론 요약)
# ---------------------------
def report_body():
    st.markdown("## 🧭 문제 제기(서론)")
    st.markdown(
        "최근 기후 이상과 함께 해수면 상승이 눈에 띄게 진행되고 있습니다. "
        "바다는 지구의 열을 저장하고 순환시키는 거대한 장치인데, 온도가 올라가면 **열팽창**과 **빙하 융해**가 겹쳐 "
        "해수면이 서서히 높아집니다. 이 변화는 해안 침식, 내륙 침수 위험 증가, 폭염·해충 증가 등으로 학생들의 일상과 "
        "학습 환경에 직접적인 영향을 주고 있습니다."
    )
    st.markdown("---")
    st.markdown("## 🔍 본론 1 — 데이터로 본 해수면 상승의 현실·원인")
    st.markdown(
        "- **국제 추세**: 장기적으로 평균 해수면과 해수온이 상승하는 패턴이 관측됩니다. "
        "이는 해양열파 빈도 증가, 산호 백화 등 생태계 변화를 동반합니다.\n"
        "- **대한민국 사례**: 연안 관리·침식 대응, 내륙 침수 대비, 주거·시설물 보강의 필요성이 커지고 있습니다."
    )
    st.markdown("## 🧑‍🎓 본론 1-2 — 피해 통계와 사례(청소년)")
    st.markdown(
        "가정에서는 **폭염·한파**, **해충 증가**, **침수·곰팡이** 문제가 잦아지며, 학생들은 집중력 저하와 "
        "불안감 증가를 호소합니다. 이는 단순한 날씨 문제가 아니라, **건강·학습권·정서**의 종합 이슈입니다."
    )
    st.markdown("## 🌍 본론 2 — 청소년과 미래에 미치는 영향 / 정책적 대응 필요성")
    st.markdown(
        "- **영향**: 학습 활동 제한, 열 관련 질환 위험, 여가·체육 활동의 제약, 안전한 통학권 위협 등.\n"
        "- **정책적 대응**: 교실 26℃ 유지/환기 지침, 학교·지역의 냉방/그늘 인프라, "
        "취약 계층 지원 강화, 데이터 기반 예산 편성 및 시설 개선이 필요합니다."
    )
    st.markdown("---")
    st.markdown("## ✅ 결론 — 고1 눈높이로 정리한 우리의 선택")
    st.markdown(
        "해수면 상승은 멀리 있는 바다 이야기처럼 보일 수 있지만, 실제로는 교실의 온도, 등하굣길의 안전, "
        "집안의 곰팡이 같은 아주 가까운 문제로 나타납니다. 그래서 우리는 **오늘 할 수 있는 일**부터 시작해야 합니다. "
        "오후 햇빛이 강할 때 블라인드를 내려 교실 온도를 낮추고, 쉬는 시간에는 2분간 칼환기를 해 더운 공기를 내보냅니다. "
        "빈 교실의 전원을 끄는 습관을 들이면 에너지도 아끼고 열도 줄일 수 있어요. 이런 작은 실천을 반 전체가 함께하면 "
        "효과는 더 커집니다.\n\n"
        "동시에 우리는 **데이터로 말하는 힘**을 키워야 합니다. 기온·해수면 그래프를 직접 그려 보고, "
        "우리 학교 상황을 조사해보세요. 숫자와 근거를 붙여 학생회나 학교, 교육청에 **그늘막 설치**나 "
        "**차열 페인트** 같은 구체적인 개선을 요구한다면, 어른들도 더 쉽게 움직일 수 있습니다. "
        "바다가 뜨거워지는 속도를 바로 멈출 수는 없지만, 우리의 교실을 더 안전하고 시원하게 만드는 일은 "
        "지금 당장 시작할 수 있습니다. **작은 변화가 모여 내일의 학교를 바꿉니다. 💚🤎**"
    )

# ---------------------------
# 메인
# ---------------------------
def main():
    st.set_page_config(page_title="보고서 맞춤 대시보드", layout="wide")
    st.title(APP_TITLE)

    # 상단 보고서 개요/본문(클릭형 소제목)
    report_overview()
    report_body()
    st.divider()

    tabs = st.tabs(["🥠 포춘쿠키", "🔬 데이터 관측실", "🗂️ 자료실"])
    with tabs[0]:
        fortune_cookie_tab()
    with tabs[1]:
        data_lab_tab()
    with tabs[2]:
        library_tab()

    st.caption("※ 본 앱의 데이터는 시연용 더미입니다. 실제 분석 시 NOAA/NASA/정부 공개 데이터를 동일 스키마(date, value, group)로 연결하세요.")

if __name__ == "__main__":
    main()
