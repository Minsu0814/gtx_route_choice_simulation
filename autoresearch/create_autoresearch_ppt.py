# -*- coding: utf-8 -*-
"""AutoResearch 소개 PPT 생성 — 프레임워크 설명, 적용 사례, 실험 결과, 활용 방안"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import csv
import json

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / 'pptx' / 'AutoResearch_소개.pptx'
RESULTS_PATH = Path(__file__).resolve().parent / 'results.tsv'
BASELINE_PATH = Path(__file__).resolve().parent / 'baseline.json'

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# 색상
DARK = RGBColor(0x2D, 0x2D, 0x2D)
BLUE = RGBColor(0x1F, 0x77, 0xB4)
RED = RGBColor(0xD6, 0x27, 0x28)
GREEN = RGBColor(0x2C, 0xA0, 0x2C)
GRAY = RGBColor(0x7F, 0x7F, 0x7F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF5, 0xF5, 0xF5)
TITLE_BG = RGBColor(0x1A, 0x47, 0x2A)
BOX_BLUE = RGBColor(0xDB, 0xEB, 0xF7)
BOX_GREEN = RGBColor(0xD5, 0xF0, 0xD5)
BOX_ORANGE = RGBColor(0xFD, 0xE8, 0xD0)
BOX_RED = RGBColor(0xF8, 0xD7, 0xDA)
BOX_GRAY = RGBColor(0xE8, 0xE8, 0xE8)
BOX_PURPLE = RGBColor(0xE8, 0xDA, 0xF0)
BORDER_BLUE = RGBColor(0x1F, 0x77, 0xB4)
BORDER_GREEN = RGBColor(0x2C, 0xA0, 0x2C)
BORDER_ORANGE = RGBColor(0xE6, 0x85, 0x0C)
BORDER_PURPLE = RGBColor(0x8B, 0x5C, 0xF6)


def add_slide(title_text, layout_idx=5):
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    bg = slide.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.9))
    shape.fill.solid(); shape.fill.fore_color.rgb = TITLE_BG
    shape.line.fill.background()
    tf = shape.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = title_text
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.5); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return slide


def add_textbox(slide, left, top, width, height, text, size=14, bold=False, color=DARK, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    p.alignment = alignment
    return tf


def add_multi_text(slide, left, top, width, height, lines, size=13):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame; tf.word_wrap = True
    for i, (text, bold, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text; p.font.size = Pt(size); p.font.bold = bold
        p.font.color.rgb = color; p.space_after = Pt(4)
    return tf


def add_table(slide, left, top, width, height, headers, rows, col_widths=None):
    n_rows = len(rows) + 1; n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths): table.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        cell = table.cell(0, j); cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
        cell.fill.solid(); cell.fill.fore_color.rgb = TITLE_BG
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j); cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10); p.font.color.rgb = DARK; p.alignment = PP_ALIGN.CENTER
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT_BG if i % 2 == 0 else WHITE
    return table


def add_box(slide, left, top, width, height, text, fill_color, border_color=None, text_size=11, bold=False, text_color=DARK):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color; shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    tf = shape.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.1)
    p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(text_size)
    p.font.bold = bold; p.font.color.rgb = text_color; p.alignment = PP_ALIGN.CENTER
    return shape


# ── 데이터 로딩 ──
def load_results():
    rows = []
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    return rows


def load_baseline():
    with open(BASELINE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


results = load_results()
baseline = load_baseline()
kept_results = [r for r in results if r['kept'] == 'Y']
first_result = results[0]


# ============================================================
# Slide 1: 표지
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[5])
bg = slide.background.fill; bg.solid(); bg.fore_color.rgb = TITLE_BG
add_textbox(slide, 1, 1.8, 11, 1.5,
    "AutoResearch", 44, True, WHITE, PP_ALIGN.CENTER)
add_textbox(slide, 1, 3.0, 11, 1.0,
    "AI 에이전트 기반 자율 ML 연구 프레임워크  (Andrej Karpathy)", 22, False, WHITE, PP_ALIGN.CENTER)
add_textbox(slide, 1, 4.5, 11, 0.8,
    "2026.03.31", 20, False, RGBColor(0xCC, 0xCC, 0xCC), PP_ALIGN.CENTER)
add_textbox(slide, 1, 5.2, 11, 0.8,
    "김민수", 18, False, RGBColor(0xCC, 0xCC, 0xCC), PP_ALIGN.CENTER)


# ============================================================
# Slide 2: AutoResearch란? — Karpathy 원본 개념
# ============================================================
slide = add_slide("1. AutoResearch란?  (Andrej Karpathy, 2025)")

# 왼쪽: 핵심 개념
add_textbox(slide, 0.5, 1.2, 6, 0.4, "개념: LLM 에이전트가 LLM 학습을 자율적으로 수행", 15, True, BLUE)
add_multi_text(slide, 0.5, 1.8, 6, 3.5, [
    ("기존 ML 연구 워크플로", True, GRAY),
    ("  사람이 코드 수정 → 학습 → 결과 확인 → 다시 수정 ... (반복)", False, GRAY),
    ("", False, DARK),
    ("AutoResearch 워크플로", True, RED),
    ("  사람은 가이드 문서만 작성 → AI 에이전트가 밤새 자율 실험", False, RED),
    ("", False, DARK),
    ("핵심 아이디어", True, DARK),
    ("  • AI 에이전트에게 실제 학습 코드를 주고 자율적으로 실험하게 함", False, DARK),
    ("  • 코드를 직접 수정 → 학습 → 결과 확인 → 개선/폐기를 반복", False, DARK),
    ("  • 사람의 역할: Python 코딩이 아니라 \"가이드 문서\" 작성으로 전환", False, DARK),
    ("  • 실험당 5분 고정 → 시간당 ~12회, 하룻밤 ~100회 실험 가능", False, DARK),
], size=13)

# 오른쪽: 프레임워크 구조 (원본)
add_textbox(slide, 7, 1.2, 5.5, 0.4, "프레임워크 구성 (3개 파일)", 15, True, BLUE)

add_box(slide, 7.0, 1.8, 5.5, 0.8,
    "program.md  —  가이드 문서 (사람이 작성)\n"
    "AI에게 연구 방향, 규칙, 제약조건을 지시하는 SOP.\n"
    "사람은 코드 대신 이 문서를 \"프로그래밍\"한다.",
    BOX_BLUE, BORDER_BLUE, 11, False)
add_box(slide, 7.0, 2.8, 5.5, 0.8,
    "prepare.py  —  고정 인프라 (수정 불가)\n"
    "데이터 로딩, 평가 함수, 학습 루프 등.\n"
    "실험 환경을 고정하여 공정한 비교를 보장.",
    BOX_GREEN, BORDER_GREEN, 11, False)
add_box(slide, 7.0, 3.8, 5.5, 0.8,
    "train.py  —  탐색 대상 (AI만 수정)\n"
    "모델 아키텍처, 하이퍼파라미터, 학습 전략.\n"
    "AI 에이전트가 코드 수준에서 직접 수정.",
    BOX_ORANGE, BORDER_ORANGE, 11, True)

# 하단: 원본 vs 기존 방법 비교
add_textbox(slide, 0.5, 5.2, 12, 0.4, "기존 자동화 방법과의 차이", 15, True, BLUE)
add_table(slide, 0.5, 5.6, 12, 1.5,
    ['방법', 'Grid/Random Search', 'Bayesian Opt (Optuna)', 'AutoResearch (LLM)'],
    [
        ['탐색 공간', '사전 정의 (고정)', '사전 정의 (고정)', '제한 없음 (코드 자체)'],
        ['탐색 대상', '하이퍼파라미터만', '하이퍼파라미터만', '아키텍처 + HP + 전략 전부'],
        ['지능', '무작위/격자', '통계적 surrogate', 'LLM의 ML 지식 활용'],
    ],
    col_widths=[1.8, 2.8, 3.0, 4.4])


# ============================================================
# Slide 3: 실험 루프 & 파일 흐름
# ============================================================
slide = add_slide("2. 작동 원리: AI 에이전트 실험 루프")

# 상단: 순환 루프 흐름도
add_textbox(slide, 0.5, 1.2, 12, 0.4, "자율 실험 사이클 (5분/회, 시간당 ~12회)", 15, True, BLUE)

add_box(slide, 0.5, 1.8, 2.3, 0.8, "1. 코드 분석\ntrain.py +\nresults.tsv 읽기", BOX_PURPLE, BORDER_PURPLE, 10, True)
add_textbox(slide, 2.8, 2.0, 0.4, 0.3, "->", 18, True, GRAY, PP_ALIGN.CENTER)

add_box(slide, 3.2, 1.8, 2.3, 0.8, "2. 개선안 구상\n이전 실험 이력 +\nML 지식 기반", BOX_PURPLE, BORDER_PURPLE, 10, True)
add_textbox(slide, 5.5, 2.0, 0.4, 0.3, "->", 18, True, GRAY, PP_ALIGN.CENTER)

add_box(slide, 5.9, 1.8, 2.3, 0.8, "3. train.py 수정\n아키텍처/HP/전략\n코드 직접 변경", BOX_ORANGE, BORDER_ORANGE, 10, True)
add_textbox(slide, 8.2, 2.0, 0.4, 0.3, "->", 18, True, GRAY, PP_ALIGN.CENTER)

add_box(slide, 8.6, 1.8, 2.1, 0.8, "4. 학습 실행\npython train.py\n(5분 제한)", BOX_GREEN, BORDER_GREEN, 10, True)
add_textbox(slide, 10.7, 2.0, 0.4, 0.3, "->", 18, True, GRAY, PP_ALIGN.CENTER)

add_box(slide, 11.1, 1.8, 2.0, 0.8, "5. 결과 판정\n개선 → Keep\n미개선 → Discard", BOX_RED, RGBColor(0xC0, 0x39, 0x2B), 10, True)

# 순환 화살표 (텍스트로)
add_box(slide, 4.5, 2.9, 4.5, 0.4,
    "개선 시 baseline 갱신 → 다음 실험으로 순환 반복",
    BOX_GRAY, None, 10, True, GRAY)

# 하단: 파일 흐름 3열 구조
add_textbox(slide, 0.5, 3.6, 12, 0.4, "파일 역할 분담", 15, True, BLUE)

col_w = 3.8
x1, x2, x3 = 0.5, 4.8, 9.1

add_textbox(slide, x1, 4.1, col_w, 0.3, "고정 레이어 (수정 불가)", 13, True, BLUE)
add_box(slide, x1, 4.5, col_w, 0.7,
    "program.md\n규칙, 목표, 제약조건 (사람이 작성)",
    BOX_BLUE, BORDER_BLUE, 10)
add_box(slide, x1, 5.4, col_w, 0.7,
    "prepare.py\n데이터 로딩, 평가 함수, 학습 루프",
    BOX_BLUE, BORDER_BLUE, 10)

add_textbox(slide, x2, 4.1, col_w, 0.3, "탐색 레이어 (AI 수정)", 13, True, RGBColor(0xE6, 0x85, 0x0C))
add_box(slide, x2, 4.5, col_w, 0.7,
    "train.py — 아키텍처\n레이어, 차원, 활성함수, Normalization",
    BOX_ORANGE, BORDER_ORANGE, 10)
add_box(slide, x2, 5.4, col_w, 0.7,
    "train.py — 하이퍼파라미터 / 학습 전략\nLR, Batch, Dropout, Optimizer, Scheduler",
    BOX_ORANGE, BORDER_ORANGE, 10)

add_textbox(slide, x3, 4.1, col_w, 0.3, "자동 기록", 13, True, GREEN)
add_box(slide, x3, 4.5, col_w, 0.7,
    "results.tsv\n모든 실험 자동 기록 (성능, 개선 여부)",
    BOX_GREEN, BORDER_GREEN, 10)
add_box(slide, x3, 5.4, col_w, 0.7,
    "baseline.json + best_model.pt\n최고 성능 자동 갱신 + 모델 저장",
    BOX_GREEN, BORDER_GREEN, 10)

# 하단: 핵심 메시지
add_box(slide, 0.5, 6.4, 12.3, 0.8,
    "핵심: 사람은 규칙(program.md)과 데이터(prepare.py)만 준비하면, "
    "AI 에이전트가 train.py를 반복 수정하며 자율적으로 최적화. 모든 실험은 자동 로깅.",
    BOX_GREEN, BORDER_GREEN, 12, True, DARK)


# ============================================================
# Slide 4: 적용 사례
# ============================================================
slide = add_slide("3. 우리의 적용: 수도권 대중교통 경로선택 최적화")

# 왼쪽: 데이터 설명
add_textbox(slide, 0.5, 1.2, 6, 0.4, "데이터", 16, True, BLUE)
add_multi_text(slide, 0.5, 1.7, 6, 2.5, [
    ("수도권 스마트카드 전수 데이터", True, DARK),
    ("  • 37,000,000 통행 (7일 전수)", False, DARK),
    ("  • 526,968 정류장 OD 쌍", False, DARK),
    ("  • 1,960,427 학습 데이터 행", False, DARK),
    ("", False, DARK),
    ("9개 모형 피처", True, DARK),
    ("  • IVT (차내시간), Access/Egress 도보, 환승 도보", False, DARK),
    ("  • 환승 횟수, 요금, 버스/철도/GTX 여부", False, DARK),
    ("", False, DARK),
    ("2개 컨텍스트 피처", True, DARK),
    ("  • OD 거리, 선택지 수 (choice set size)", False, DARK),
], size=13)

# 오른쪽: 모형 설명
add_textbox(slide, 7, 1.2, 5.5, 0.4, "기본 모형: TasteNet", 16, True, BLUE)
add_multi_text(slide, 7, 1.7, 5.5, 2.5, [
    ("TasteNet 구조", True, DARK),
    ("  • Context z = (OD 거리, 선택지 수)", False, DARK),
    ("  • Taste Network: z -> beta(z)  (개인별 계수 생성)", False, DARK),
    ("  • 효용함수: V = X * beta(z)", False, DARK),
    ("  • 선택확률: softmax(V) with masking", False, DARK),
    ("", False, DARK),
    ("기본 성능 (TasteNet baseline)", True, RED),
    (f"  • test rho_sq = {float(first_result['rho_sq']):.4f}", False, DARK),
    (f"  • Top-1 정확도 = {float(first_result['fpr_top1'])*100:.1f}%", False, DARK),
    (f"  • Top-3 정확도 = {float(first_result['fpr_top3'])*100:.1f}%", False, DARK),
], size=13)

# 하단: 최적화 목표
add_box(slide, 0.5, 5.0, 12.3, 1.8,
    "최적화 설정\n\n"
    f"• 최적화 목표: test rho_sq (McFadden's pseudo R-squared) — 단일 지표, 높을수록 좋음\n"
    f"• 기본 모형: TasteNet (rho_sq = {float(first_result['rho_sq']):.4f})\n"
    "• 시간 제약: 실험당 최대 300초 (5분)\n"
    "• AI 에이전트에게 train.py만 수정 허용, 나머지 모든 파일 수정 금지\n"
    "• 제약: 피처 9개 + 컨텍스트 2개 고정, forward(X, z, mask) 시그니처 유지, MAX_ALTS=5",
    BOX_ORANGE, BORDER_ORANGE, 12, False, DARK)


# ============================================================
# Slide 5: 실험 결과 - 성능 변화 추이
# ============================================================
slide = add_slide("4. 실험 결과: 성능 개선 이력")

# kept=Y인 실험만 테이블로
table_rows = []
for r in kept_results:
    table_rows.append([
        r['experiment'].replace('_', ' '),
        r['model'],
        f"{float(r['rho_sq']):.4f}",
        f"{float(r['fpr_top1'])*100:.1f}%",
        r['notes'],
    ])

add_table(slide, 0.3, 1.1, 12.7, 5.0,
    ['실험명', '모델', 'rho_sq', 'Top-1', '비고'],
    table_rows,
    col_widths=[3.0, 2.5, 1.5, 1.2, 4.5])

total = len(results)
n_kept = len(kept_results)
add_textbox(slide, 0.5, 6.3, 12, 0.5,
    f"총 {total}회 실험 중 {n_kept}회 개선 ({n_kept/total*100:.0f}% 성공률)  |  "
    f"소요 시간: ~3시간 (00:56 ~ 03:47)  |  "
    f"최종 rho_sq: {float(baseline['rho_sq']):.4f}",
    14, True, BLUE, PP_ALIGN.LEFT)


# ============================================================
# Slide 6: 아키텍처 진화 과정
# ============================================================
slide = add_slide("5. AI가 발견한 아키텍처 진화 과정")

# 핵심 마일스톤 5개
milestones = [
    ("TasteNet\n(baseline)", "0.5425", "기본 TasteNet\n2-layer taste net\nβ(z) * X", BOX_BLUE),
    ("MultiHead\n(exp03)", "0.5618", "피처 그룹별\n별도 taste net\n+interaction terms", BOX_GREEN),
    ("UtilityNet\n(exp09-10)", "0.5730", "효용 함수에\n비선형 네트워크\n추가 (BigUtilNet)", BOX_ORANGE),
    ("ResUtilNet\n(exp13)", "0.5757", "Residual 연결\nSkip connection\n깊은 네트워크 안정화", BOX_RED),
    ("SingleRes\n(exp20)", "0.5764", "단일 ResNet\n4 블록, 256 dim\nFeature expansion", RGBColor(0xD5, 0xC8, 0xF0)),
]

y_top = 1.3
box_w = 2.2
box_h = 1.8
gap = 0.35

for i, (name, rho, desc, color) in enumerate(milestones):
    x = 0.5 + i * (box_w + gap)
    # 모델명 박스
    border = BORDER_BLUE if i == 0 else BORDER_GREEN if i == 1 else BORDER_ORANGE if i == 2 else RGBColor(0xC0, 0x39, 0x2B) if i == 3 else BORDER_PURPLE
    add_box(slide, x, y_top, box_w, 0.7, name, color, border, 11, True)
    # rho_sq 값
    add_textbox(slide, x, y_top + 0.8, box_w, 0.4, f"rho_sq = {rho}", 14, True, BLUE, PP_ALIGN.CENTER)
    # 설명
    add_box(slide, x, y_top + 1.3, box_w, 1.2, desc, LIGHT_BG, None, 10, False)
    # 화살표 (마지막 제외)
    if i < len(milestones) - 1:
        add_textbox(slide, x + box_w, y_top + 0.25, gap, 0.3, "->", 16, True, GRAY, PP_ALIGN.CENTER)

# 개선폭 시각화
add_textbox(slide, 0.5, 4.6, 12, 0.4, "단계별 개선폭", 15, True, BLUE)

improvements = [
    ("baseline -> MultiHead", "+0.0193", "가장 큰 점프 (MultiHead + interaction terms)"),
    ("MultiHead -> UtilityNet", "+0.0112", "비선형 효용 함수 네트워크 도입"),
    ("UtilityNet -> ResUtilNet", "+0.0027", "Residual connection으로 안정화"),
    ("ResUtilNet -> SingleRes", "+0.0006", "단일 ResNet 구조 (미미한 개선)"),
]

for i, (stage, delta, note) in enumerate(improvements):
    y = 5.1 + i * 0.35
    add_textbox(slide, 0.5, y, 3.5, 0.3, stage, 11, False, DARK)
    add_textbox(slide, 4.2, y, 1.2, 0.3, delta, 11, True, GREEN if float(delta) > 0.005 else GRAY)
    add_textbox(slide, 5.5, y, 7, 0.3, note, 11, False, GRAY)

# 정체 구간
add_box(slide, 8.5, 4.5, 4.3, 0.5,
    "exp20 이후 11회 연속 실패 -> 수렴 판단",
    BOX_RED, RGBColor(0xC0, 0x39, 0x2B), 11, True, RED)


# ============================================================
# Slide 7: 최종 결과 요약
# ============================================================
slide = add_slide("6. 결과 요약")

# Before/After 비교 테이블
add_textbox(slide, 0.5, 1.2, 6, 0.4, "Before vs After", 16, True, BLUE)
add_table(slide, 0.5, 1.7, 6, 2.0,
    ['지표', 'Before (TasteNet)', 'After (SingleRes)', '변화'],
    [
        ['rho_sq', f"{float(first_result['rho_sq']):.4f}", f"{float(baseline['rho_sq']):.4f}",
         f"+{float(baseline['rho_sq']) - float(first_result['rho_sq']):.4f}"],
        ['Top-1 정확도', f"{float(first_result['fpr_top1'])*100:.1f}%", f"{float(baseline['fpr_top1'])*100:.1f}%",
         f"+{(float(baseline['fpr_top1']) - float(first_result['fpr_top1']))*100:.1f}%p"],
        ['Top-3 정확도', f"{float(first_result['fpr_top3'])*100:.1f}%", f"{float(baseline['fpr_top3'])*100:.1f}%",
         f"+{(float(baseline['fpr_top3']) - float(first_result['fpr_top3']))*100:.1f}%p"],
        ['RMSE', f"{float(first_result['rmse']):.4f}", f"{float(baseline['rmse']):.4f}",
         f"{float(baseline['rmse']) - float(first_result['rmse']):.4f}"],
    ],
    col_widths=[1.5, 2.0, 2.0, 1.0])

# 오른쪽: 핵심 성과 박스
add_textbox(slide, 7, 1.2, 5.5, 0.4, "핵심 성과", 16, True, BLUE)

rho_improve = (float(baseline['rho_sq']) - float(first_result['rho_sq'])) / float(first_result['rho_sq']) * 100

add_box(slide, 7.0, 1.7, 5.5, 0.7,
    f"rho_sq 향상: +{rho_improve:.1f}%\n({float(first_result['rho_sq']):.4f} -> {float(baseline['rho_sq']):.4f})",
    BOX_GREEN, BORDER_GREEN, 13, True, DARK)
add_box(slide, 7.0, 2.6, 5.5, 0.6,
    "소요 시간: 약 3시간\n00:56 ~ 03:47 (2026-03-30)",
    BOX_BLUE, BORDER_BLUE, 13, True, DARK)
add_box(slide, 7.0, 3.4, 5.5, 0.6,
    f"실험 횟수: {len(results)}회 (성공 {len(kept_results)}회, {len(kept_results)/len(results)*100:.0f}%)",
    BOX_ORANGE, BORDER_ORANGE, 13, True, DARK)

# 하단: 최적 모델 상세
add_textbox(slide, 0.5, 4.5, 12, 0.4, "최적 모델: TasteNet-SingleRes", 16, True, BLUE)
add_multi_text(slide, 0.5, 5.0, 12, 2.0, [
    ("아키텍처", True, DARK),
    ("  • 입력: 9개 피처 + 6개 제곱항 + 8개 교차항 = 23개 확장 피처", False, DARK),
    ("  • 컨텍스트 확장: z + 대안별 평균/표준편차 → 풍부한 OD 정보 활용", False, DARK),
    ("  • ResBlock 4개 (LayerNorm → Linear → GELU → Dropout), 256 hidden dim", False, DARK),
    ("  • 총 파라미터 수: ~530K", False, DARK),
    ("", False, DARK),
    ("학습 설정: lr=1e-3, batch=2048, patience=25, AdamW", False, GRAY),
], size=12)


# ============================================================
# Slide 8: 활용 방안 및 확장성
# ============================================================
slide = add_slide("7. 활용 방안 및 향후 적용")

# 3개 활용 분야
add_textbox(slide, 0.5, 1.2, 12, 0.4, "AutoResearch 적용 가능 분야", 16, True, BLUE)

add_box(slide, 0.5, 1.8, 3.8, 2.0,
    "ML/DL 하이퍼파라미터 최적화\n\n"
    "• 명확한 평가지표가 있는 모든 ML 문제\n"
    "• Grid search/Random search 대비\n  훨씬 지능적인 탐색\n"
    "• 아키텍처와 하이퍼파라미터 동시 최적화",
    BOX_BLUE, BORDER_BLUE, 10, False)

add_box(slide, 4.8, 1.8, 3.8, 2.0,
    "자동 아키텍처 탐색 (NAS)\n\n"
    "• 기존 NAS 대비 유연한 탐색 공간\n"
    "• AI가 코드 수준에서 직접 수정\n  → 사전 정의된 탐색 공간 불필요\n"
    "• 실험 이력 기반 지능적 방향 설정",
    BOX_GREEN, BORDER_GREEN, 10, False)

add_box(slide, 9.1, 1.8, 3.8, 2.0,
    "다양한 도메인 적용\n\n"
    "• 경로선택, 수요 예측, 교통 시뮬레이션\n"
    "• 이미지 분류, NLP, 추천 시스템\n"
    "• prepare.py와 program.md만 교체하면\n  어떤 도메인이든 적용 가능",
    BOX_ORANGE, BORDER_ORANGE, 10, False)

# 적용 방법 흐름
add_textbox(slide, 0.5, 4.2, 12, 0.4, "새로운 문제에 적용하는 방법", 16, True, BLUE)

add_box(slide, 0.5, 4.8, 2.8, 1.0,
    "Step 1\n\nprepare.py 작성\n(데이터 로딩 + 평가 함수)",
    BOX_BLUE, BORDER_BLUE, 11, True)
add_textbox(slide, 3.3, 5.1, 0.5, 0.3, "->", 20, True, GRAY, PP_ALIGN.CENTER)

add_box(slide, 3.8, 4.8, 2.8, 1.0,
    "Step 2\n\nprogram.md 작성\n(목표, 규칙, 제약조건 정의)",
    BOX_GREEN, BORDER_GREEN, 11, True)
add_textbox(slide, 6.6, 5.1, 0.5, 0.3, "->", 20, True, GRAY, PP_ALIGN.CENTER)

add_box(slide, 7.1, 4.8, 2.8, 1.0,
    "Step 3\n\ntrain.py 초기 버전 작성\n(기본 모델 정의)",
    BOX_ORANGE, BORDER_ORANGE, 11, True)
add_textbox(slide, 9.9, 5.1, 0.5, 0.3, "->", 20, True, GRAY, PP_ALIGN.CENTER)

add_box(slide, 10.4, 4.8, 2.5, 1.0,
    "Step 4\n\nAI 에이전트 실행\n(자동 최적화 시작)",
    BOX_PURPLE, BORDER_PURPLE, 11, True)

# 하단: 핵심 메시지
add_box(slide, 0.5, 6.2, 12.3, 1.0,
    "AutoResearch의 핵심 가치: 연구자는 \"무엇을 최적화할 것인가\"에 집중하고,\n"
    "\"어떻게 최적화할 것인가\"는 AI 에이전트에게 위임. 실험 설계-실행-평가의 전 과정을 자동화.",
    BOX_GREEN, BORDER_GREEN, 13, True, DARK)


# ============================================================
# 저장
# ============================================================
prs.save(str(OUT_PATH))
print(f'PPT saved: {OUT_PATH}')
print(f'  Slides: {len(prs.slides)}')
print(f'  Results: {len(results)} experiments ({len(kept_results)} kept)')
print(f'  Best: {baseline["model_name"]} (rho_sq={baseline["rho_sq"]:.4f})')
