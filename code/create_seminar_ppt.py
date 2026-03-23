"""세미나 PPT 생성: 대중교통 경로 유사도 매칭 방법론 개선
기존 PPT 템플릿(헤더 그룹 + 체크아이콘 + 소제목 패턴) 활용
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from lxml import etree
import os

# ── Style Constants ──
FONT_BOLD = "에스코어 드림 6 Bold"
FONT_MEDIUM = "에스코어 드림 5 Medium"
FONT_REGULAR = "에스코어 드림 4 Regular"
FONT_THIN = "에스코어 드림 1 Thin"
FONT_NOTO = "Noto Sans KR"

COLOR_HEADER = RGBColor(0x37, 0x41, 0x51)
COLOR_BLUE = RGBColor(0x13, 0x53, 0x8A)
COLOR_RED = RGBColor(0xFF, 0x00, 0x00)
COLOR_BODY = RGBColor(0x37, 0x41, 0x51)
COLOR_TABLE_TEXT = RGBColor(0x4B, 0x55, 0x63)
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
COLOR_BG_BLUE = RGBColor(0x1A, 0x3C, 0x6E)
COLOR_ACCENT_GREEN = RGBColor(0x00, 0xB0, 0x50)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# Check icon extracted from existing PPT
CHECK_ICON_PATH = os.path.expanduser('~/.claude/skills/make-ppt/check_icon.png')
ICON_SIZE = Inches(0.32)

# ── Header group XML (blue bar + section text + title) ──
HEADER_GROUP_XML = '''<p:grpSp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
         xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
         xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:nvGrpSpPr>
    <p:cNvPr id="{id1}" name="HeaderGroup"/>
    <p:cNvGrpSpPr/>
    <p:nvPr/>
  </p:nvGrpSpPr>
  <p:grpSpPr>
    <a:xfrm>
      <a:off x="600499" y="-106626"/>
      <a:ext cx="10059881" cy="975104"/>
      <a:chOff x="600499" y="-106626"/>
      <a:chExt cx="10059881" cy="975104"/>
    </a:xfrm>
  </p:grpSpPr>
  <p:sp>
    <p:nvSpPr><p:cNvPr id="{id2}" name="SectionText"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
    <p:spPr>
      <a:xfrm><a:off x="947712" y="210679"/><a:ext cx="3587608" cy="350289"/></a:xfrm>
      <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    </p:spPr>
    <p:txBody>
      <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="t"><a:spAutoFit/></a:bodyPr>
      <a:lstStyle/>
      <a:p>
        <a:pPr><a:lnSpc><a:spcPts val="3143"/></a:lnSpc><a:spcBef><a:spcPct val="0"/></a:spcBef></a:pPr>
        <a:r>
          <a:rPr lang="ko-KR" altLang="en-US" sz="1600" spc="-112" dirty="0">
            <a:solidFill><a:srgbClr val="374151"/></a:solidFill>
            <a:latin typeface="\uc5d0\uc2a4\ucf54\uc5b4 \ub4dc\ub9bc 4 Regular" panose="020B0503030302020204" pitchFamily="34" charset="-127"/>
            <a:ea typeface="\uc5d0\uc2a4\ucf54\uc5b4 \ub4dc\ub9bc 4 Regular" panose="020B0503030302020204" pitchFamily="34" charset="-127"/>
          </a:rPr>
          <a:t>{section_text}</a:t>
        </a:r>
      </a:p>
    </p:txBody>
  </p:sp>
  <p:sp>
    <p:nvSpPr><p:cNvPr id="{id3}" name="BlueBar"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
    <p:spPr>
      <a:xfrm flipH="1"><a:off x="600499" y="-106626"/><a:ext cx="272955" cy="926910"/></a:xfrm>
      <a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom>
      <a:solidFill><a:srgbClr val="13538A"/></a:solidFill>
      <a:ln><a:noFill/></a:ln>
    </p:spPr>
    <p:txBody>
      <a:bodyPr rtlCol="0" anchor="ctr"/><a:lstStyle/>
      <a:p><a:pPr algn="ctr"/><a:endParaRPr lang="ko-KR" altLang="en-US" dirty="0"/></a:p>
    </p:txBody>
  </p:sp>
  <p:sp>
    <p:nvSpPr><p:cNvPr id="{id4}" name="TitleText"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
    <p:spPr>
      <a:xfrm><a:off x="947712" y="495491"/><a:ext cx="9712668" cy="372987"/></a:xfrm>
      <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    </p:spPr>
    <p:txBody>
      <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="t"><a:spAutoFit/></a:bodyPr>
      <a:lstStyle/>
      <a:p>
        <a:pPr><a:lnSpc><a:spcPts val="3143"/></a:lnSpc><a:spcBef><a:spcPct val="0"/></a:spcBef></a:pPr>
        <a:r>
          <a:rPr lang="ko-KR" altLang="en-US" sz="2245" spc="-112" dirty="0">
            <a:solidFill><a:srgbClr val="374151"/></a:solidFill>
            <a:latin typeface="\uc5d0\uc2a4\ucf54\uc5b4 \ub4dc\ub9bc 6 Bold" panose="020B0703030302020204" pitchFamily="34" charset="-127"/>
            <a:ea typeface="\uc5d0\uc2a4\ucf54\uc5b4 \ub4dc\ub9bc 6 Bold" panose="020B0703030302020204" pitchFamily="34" charset="-127"/>
          </a:rPr>
          <a:t>{title_text}</a:t>
        </a:r>
      </a:p>
    </p:txBody>
  </p:sp>
</p:grpSp>'''


_next_id = 100

def get_ids():
    global _next_id
    ids = (_next_id, _next_id+1, _next_id+2, _next_id+3)
    _next_id += 4
    return ids


def add_header_group(slide, section_text, title_text):
    id1, id2, id3, id4 = get_ids()
    xml = HEADER_GROUP_XML.format(id1=id1, id2=id2, id3=id3, id4=id4,
                                   section_text=section_text, title_text=title_text)
    slide._element.spTree.append(etree.fromstring(xml.encode('utf-8')))


def set_run(run, text, font_name=FONT_REGULAR, size=Pt(14), color=COLOR_BODY, bold=False):
    run.text = text
    run.font.name = font_name
    run.font.size = size
    run.font.color.rgb = color
    run.font.bold = bold


def add_paragraph(tf, text, font_name=FONT_REGULAR, size=Pt(14), color=COLOR_BODY,
                  bold=False, alignment=PP_ALIGN.LEFT, space_before=Pt(4), space_after=Pt(2)):
    p = tf.add_paragraph()
    p.alignment = alignment
    p.space_before = space_before
    p.space_after = space_after
    run = p.add_run()
    set_run(run, text, font_name, size, color, bold)
    return p


def add_mixed_paragraph(tf, parts, alignment=PP_ALIGN.LEFT, space_before=Pt(4), space_after=Pt(2)):
    p = tf.add_paragraph()
    p.alignment = alignment
    p.space_before = space_before
    p.space_after = space_after
    for text, font, size, color, bold in parts:
        run = p.add_run()
        set_run(run, text, font, size, color, bold)
    return p


def add_content_section(slide, y_offset, subtitle, sub_subtitle=None, body_lines=None):
    """Add a content section with check icon + subtitle + optional sub-subtitle + body.

    Pattern from existing PPT:
    - Check icon at (0.80in, y_offset)
    - Subtitle at (1.22in, y_offset) - 에스코어 드림 5 Medium, Bold, #374151
    - Sub-subtitle at (1.22in, y_offset + 0.41in) - 에스코어 드림 5 Medium, 15pt, Bold
    - Body text at (1.22in, y_offset + 0.79in) - 에스코어 드림 4 Regular, 14pt

    Returns the y position after this section for stacking.
    """
    y = y_offset

    # Check icon
    slide.shapes.add_picture(CHECK_ICON_PATH, Inches(0.80), y, ICON_SIZE, ICON_SIZE)

    # Subtitle (next to icon)
    tx_sub = slide.shapes.add_textbox(Inches(1.22), y, Inches(10.31), Inches(0.32))
    tf_sub = tx_sub.text_frame
    tf_sub.word_wrap = True
    p = tf_sub.paragraphs[0]
    run = p.add_run()
    set_run(run, subtitle, FONT_MEDIUM, Pt(15), COLOR_HEADER, True)

    y_next = y + Inches(0.41)

    # Sub-subtitle (optional)
    if sub_subtitle:
        tx_ssub = slide.shapes.add_textbox(Inches(1.22), y_next, Inches(10.31), Inches(0.32))
        tf_ssub = tx_ssub.text_frame
        tf_ssub.word_wrap = True
        p2 = tf_ssub.paragraphs[0]
        run2 = p2.add_run()
        set_run(run2, sub_subtitle, FONT_MEDIUM, Pt(15), COLOR_HEADER, True)
        y_next += Inches(0.38)

    # Body text
    if body_lines:
        body_height = max(Inches(1.0), Inches(len(body_lines) * 0.32))
        tx_body = slide.shapes.add_textbox(Inches(1.22), y_next, Inches(11.44), body_height)
        tf_body = tx_body.text_frame
        tf_body.word_wrap = True

        for i, line in enumerate(body_lines):
            if i == 0:
                p = tf_body.paragraphs[0]
            else:
                p = tf_body.add_paragraph()
            p.space_before = Pt(3)
            p.space_after = Pt(3)

            # Handle mixed formatting: tuples for styled text
            if isinstance(line, list):
                for text, font, size, color, bold in line:
                    run = p.add_run()
                    set_run(run, text, font, size, color, bold)
            else:
                run = p.add_run()
                set_run(run, line, FONT_REGULAR, Pt(14), COLOR_BODY)

        y_next += body_height

    return y_next + Inches(0.15)


def add_table(slide, data, left, top, width, row_height=Inches(0.35), col_widths=None,
              header_color=COLOR_LIGHT_GRAY):
    rows, cols = len(data), len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, left, top, width,
                                          Inches(row_height.inches * rows))
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w

    for r, row_data in enumerate(data):
        for c, cell_text in enumerate(row_data):
            cell = table.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            is_header = (r == 0)
            run = p.add_run()
            set_run(run, str(cell_text),
                    FONT_MEDIUM if is_header else FONT_REGULAR, Pt(11),
                    COLOR_WHITE if (is_header and header_color == COLOR_BLUE)
                    else (COLOR_HEADER if is_header else COLOR_TABLE_TEXT),
                    bold=is_header)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_color if is_header else COLOR_WHITE
    return table_shape


# ═══════════════════════════════════════════════════
# SLIDES
# ═══════════════════════════════════════════════════

def make_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[2])  # Blank
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG_BLUE
    bg.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10), Inches(1.5))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "대중교통 경로 유사도 매칭\n방법론 개선", FONT_BOLD, Pt(30), COLOR_WHITE, True)

    tx2 = slide.shapes.add_textbox(Inches(1.5), Inches(3.8), Inches(10), Inches(0.8))
    tf2 = tx2.text_frame; tf2.word_wrap = True
    p2 = tf2.paragraphs[0]; p2.alignment = PP_ALIGN.CENTER
    set_run(p2.add_run(), "Transit Route Similarity Matching Methodology Improvement",
            FONT_THIN, Pt(12), RGBColor(0xCC, 0xCC, 0xCC))

    tx3 = slide.shapes.add_textbox(Inches(1.5), Inches(5.2), Inches(10), Inches(1.0))
    tf3 = tx3.text_frame; tf3.word_wrap = True
    p3 = tf3.paragraphs[0]; p3.alignment = PP_ALIGN.CENTER
    set_run(p3.add_run(), "김민수\n2026.03", FONT_NOTO, Pt(14), COLOR_WHITE)


def make_contents_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[2])  # Blank
    tx = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(5), Inches(1.0))
    set_run(tx.text_frame.paragraphs[0].add_run(), "Contents", FONT_MEDIUM, Pt(36), COLOR_BLUE, True)

    items = [
        ("01", "연구 배경 — 멀티모달 시뮬레이션과 경로선택"),
        ("02", "문제 인식"),
        ("03", "선행연구 — 경로선택 모형, 유사도 지표, 스마트카드 매칭"),
        ("04", "기존 유사도 방법론 (5-level)"),
        ("05", "기존 방법론의 문제점"),
        ("06", "개선된 방법론 (3-level + Sequence Gate)"),
        ("07", "가중치 최적화 (2-Stage Grid Search)"),
        ("08", "MNL 모형 결과 비교"),
        ("09", "향후 계획"),
    ]
    for i, (num, title) in enumerate(items):
        y = Inches(1.8) + Inches(i * 0.6)
        tx_n = slide.shapes.add_textbox(Inches(1.2), y, Inches(0.8), Inches(0.5))
        set_run(tx_n.text_frame.paragraphs[0].add_run(), num, FONT_MEDIUM, Pt(15), COLOR_BLUE, True)
        tx_t = slide.shapes.add_textbox(Inches(2.2), y, Inches(8), Inches(0.5))
        set_run(tx_t.text_frame.paragraphs[0].add_run(), title, FONT_REGULAR, Pt(15), COLOR_BODY)


def make_slide_bg_multimodal(prs):
    """연구 배경 — 멀티모달 시뮬레이션의 필요성"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 1장 연구 배경", "1.1 멀티모달 시뮬레이션의 필요성")

    add_content_section(slide, Inches(1.2),
        subtitle="수도권 대중교통 환경 변화",
        body_lines=[
            "• GTX-A/B/C, 신안산선 등 광역급행철도 도입 → 수도권 교통 체계 대변혁",
            "• 버스·도시철도·GTX가 결합된 멀티모달 통행 증가",
            "• 기존 단일수단 분석으로는 복합 통행 패턴 파악 한계",
        ])

    add_content_section(slide, Inches(3.2),
        subtitle="Agent-Based 시뮬레이션 접근",
        body_lines=[
            "• 전통적 4단계 수요모형: 존(Zone) 단위 집계 → 개인 행태 반영 불가",
            "• Agent-Based Model: 개인 단위 통행 의사결정 모사",
            [('• 핵심 요구: 각 에이전트의 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('경로선택 행태를 현실적으로 반영', FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
            [('→ 실제 통행자의 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('경로선택 파라미터(β) 추정', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             ('이 핵심 과제', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    # Info box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8),
                                 Inches(11.5), Inches(1.2))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
    box.line.color.rgb = RGBColor(0x93, 0xB5, 0xE1)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "연구 목표", FONT_MEDIUM, Pt(14), COLOR_BLUE, True)
    add_paragraph(tf, "GTFS+OSM 기반 멀티모달 네트워크 구축 → OTP 경로탐색 → 스마트카드 매칭 → MNL 파라미터 추정 → 시뮬레이션 반영",
                  FONT_REGULAR, Pt(13), COLOR_BODY)


def make_slide_bg_pipeline(prs):
    """연구 흐름 — 경로선택 파라미터 추정 파이프라인"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 1장 연구 배경", "1.2 경로선택 파라미터 추정 파이프라인")

    add_content_section(slide, Inches(1.2),
        subtitle="데이터 기반 접근",
        body_lines=[
            "• 스마트카드(AFC) 데이터: 실제 통행자의 승하차 정류장·시각 기록 (약 3,700만 통행/일)",
            "• GTFS + OSM: 수도권 전체 멀티모달 대중교통 네트워크 구축",
            "• OTP(OpenTripPlanner): RAPTOR 알고리즘 기반 경로탐색 엔진으로 대안경로 생성",
        ])

    add_content_section(slide, Inches(3.2),
        subtitle="파이프라인 구조",
        body_lines=[
            [('① 스마트카드 통행 집계', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (' → OD·수단·환승 패턴 추출', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('② OTP 대안경로 탐색', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (' → OD별 다수의 멀티모달 경로 생성', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('③ 유사도 매칭', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (' → 관측 경로 ↔ 대안경로 매칭, 선택확률 부여', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('④ MNL 경로선택모형 추정', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (' → β 계수 도출 (시간·비용·환승 가치)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('⑤ 시뮬레이션 파라미터 반영', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (' → DTUMOS 등 시뮬레이터에 β 매핑', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    # Highlight box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8),
                                 Inches(11.5), Inches(1.2))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFD, 0xE8, 0xE8)
    box.line.color.rgb = RGBColor(0xF5, 0xA0, 0xA0)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "핵심 과제", FONT_MEDIUM, Pt(14), COLOR_RED, True)
    add_paragraph(tf, "③번 유사도 매칭의 정확도가 전체 파이프라인 성능을 좌우 → 본 발표의 주제",
                  FONT_REGULAR, Pt(13), COLOR_BODY)


def make_slide_bg_matching(prs):
    """OTP 경로탐색과 스마트카드 매칭"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 1장 연구 배경", "1.3 OTP 경로탐색과 스마트카드 매칭")

    add_content_section(slide, Inches(1.2),
        subtitle="OTP/RAPTOR 경로탐색",
        body_lines=[
            "• OD별 다수의 대안경로 생성 (시간대·환승 조합별 Pareto-optimal 경로)",
            "• 각 대안의 소요시간, 환승횟수, 도보거리, 요금 등 속성 추출",
            "• 평균 3.7개 대안/OD → 총 약 200만 대안경로",
        ])

    add_content_section(slide, Inches(3.2),
        subtitle="스마트카드 매칭 문제",
        body_lines=[
            [('• 스마트카드: 승차/하차 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('정류장·시각만 기록', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (' (중간 경로 미관측)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• OTP 대안경로: 완전한 경로 정보 보유 (leg별 정류장·노선·시각)",
            [('• ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('과제: ', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             ('불완전한 스마트카드 정보로부터 실제 이용경로를 추정', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('→ ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('유사도(Similarity) 기반 매칭', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             ('으로 해결', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    # Comparison table
    data = [
        ["", "스마트카드 (관측)", "OTP 대안경로 (계획)"],
        ["출발 정류장", "○ 기록됨", "○ 생성됨"],
        ["도착 정류장", "○ 기록됨", "○ 생성됨"],
        ["중간 정류장", "✕ 미관측", "○ 전체 경로"],
        ["이용 노선", "○ 부분적", "○ 전체 leg"],
        ["환승 정보", "△ 추론 필요", "○ 명시적"],
    ]
    add_table(slide, data, Inches(0.8), Inches(5.3), Inches(7.0),
              row_height=Inches(0.32),
              col_widths=[Inches(1.8), Inches(2.6), Inches(2.6)])


def make_slide_bg_importance(prs):
    """유사도 매칭의 역할과 중요성"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 1장 연구 배경", "1.4 유사도 매칭의 역할과 중요성")

    add_content_section(slide, Inches(1.2),
        subtitle="왜 유사도 매칭이 핵심인가",
        body_lines=[
            "• 매칭 정확도 → 선택확률(choice probability) 품질 → MNL 계수 신뢰도",
            [('• 잘못된 매칭 = 잘못된 선택확률 = ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('편향된 파라미터', FONT_MEDIUM, Pt(14), COLOR_RED, True)],
            "• 시뮬레이션에 왜곡된 β 투입 → 비현실적 통행 배분",
        ])

    add_content_section(slide, Inches(3.2),
        subtitle="기존 방법론의 한계",
        body_lines=[
            "• 5-level composite: mode(0.10) + route(0.15) + seq(0.15) + time(0.10) + spatial(0.50)",
            [('• Spatial(Hausdorff)이 유사도의 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('50%를 차지', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (' → 정류장 좌표 기반 → 도보시간과 상관', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• 노선명(route) 일치의 중요성이 과소평가됨 (15%만 반영)",
        ])

    # Conclusion box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.3),
                                 Inches(11.5), Inches(1.5))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    box.line.color.rgb = RGBColor(0x66, 0xBB, 0x6A)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "본 발표의 목적", FONT_MEDIUM, Pt(14), RGBColor(0x2E, 0x7D, 0x32), True)
    add_paragraph(tf, "① Spatial 유사도 제거로 내생성 해소  ② Route 중심 3-level 방법론으로 개선  ③ 체계적 Grid Search로 최적 가중치 도출",
                  FONT_REGULAR, Pt(13), COLOR_BODY)
    add_paragraph(tf, "→ MNL 파라미터 신뢰도 향상 → 시뮬레이션 경로선택 현실성 확보",
                  FONT_MEDIUM, Pt(13), COLOR_BLUE, True)


def make_slide_background(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 2장 문제 인식", "2.1 연구 배경 및 문제 인식")

    add_content_section(slide, Inches(1.2),
        subtitle="교수님 지적사항",
        body_lines=[
            [('• "유사도 구하는 방식에 문제점이 많다"', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('• "이대로 논문 내면 공격받을 여지가 많다"', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('→ 유사도 선행연구 조사 + 알고리즘 수정 방안 정리 필요', FONT_MEDIUM, Pt(14), COLOR_RED, True)],
        ])

    add_content_section(slide, Inches(3.0),
        subtitle="연구 배경",
        sub_subtitle="유사도 산출이 MNL 모형 추정에 미치는 영향",
        body_lines=[
            "• 수도권 스마트카드 데이터 → OTP 대안경로 매칭 → MNL 경로선택모형 추정",
            "• 핵심 과정: 스마트카드 관측 경로와 OTP 계획 경로 간 유사도 산출",
            "• 유사도 점수 → 선택확률(choice_prob) → 종속변수로 사용",
            [('• 기존 방식: 공간적 유사도(Hausdorff) 50% 비중 → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('내생성 유발', FONT_MEDIUM, Pt(14), COLOR_RED, True)],
        ])


def make_slide_prior_research_1(prs):
    """3.1 경로선택모형 선행연구 — PSL/C-Logit 등 모형 구조"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 3장 선행연구", "3.1 대중교통 경로선택 모형")

    add_content_section(slide, Inches(1.2),
        subtitle="경로선택 모형의 핵심 문제: 중첩 경로 (Overlapping Paths)",
        body_lines=[
            "• MNL의 IIA 가정: 대안 간 독립 → 공유 구간이 많은 경로들의 선택확률 과대추정",
            "• 대중교통: 같은 환승역·노선 공유 빈번 → 중첩 보정이 필수",
        ])

    data = [
        ["연구", "모형", "유사도/중첩 지표", "데이터", "주요 성과"],
        ["Ben-Akiva &\nBierlaire (1999)", "Path Size\nLogit (PSL)", "링크 길이 기반\nPath Size 보정항", "도로 네트워크\n(이론적 제안)", "PSL 최초 제안\nMNL+보정항으로 중첩 해소"],
        ["Hoogendoorn-\nLanser et al.\n(2005)", "PSL\n(대중교통)", "Leg-count 기반\n중첩도 측정", "네덜란드\nRP 설문조사", "Leg-count > 거리/시간\n대중교통에 적합한 중첩 정의"],
        ["Bovy et al.\n(2008)", "Path Size\nCorrection", "링크 기반\nPS 보정항", "이론적 도출\n(도로망)", "Nested Logit에서\nPSC 이론적 유도"],
        ["Tan et al.\n(2015)", "Freq-aware\nPSL", "공간적 중첩 +\n배차빈도 상관", "런던\nOyster Card", "병행 노선 빈도 반영\nρ² 0.14→0.18 향상"],
        ["Dixit et al.\n(2021)", "Transfer-node\nPSC", "환승역 공유 기반\n중첩도", "암스테르담\nOV-chipkaart", "환승역 기반 보정이\n최고 적합도 (ρ²=0.33)"],
    ]
    add_table(slide, data, Inches(0.8), Inches(3.0), Inches(11.8),
              row_height=Inches(0.65),
              col_widths=[Inches(1.8), Inches(1.5), Inches(2.2), Inches(2.0), Inches(4.3)],
              header_color=COLOR_BLUE)


def make_slide_prior_research_2(prs):
    """3.2 경로 유사도 측정 방법"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 3장 선행연구", "3.2 경로 유사도 측정 방법")

    add_content_section(slide, Inches(1.2),
        subtitle="경로/궤적 유사도 측정 기법 비교",
        sub_subtitle="어떤 연구에서 어떤 지표를 사용했는가?",
        body_lines=[])

    data = [
        ["유사도 지표", "수식/원리", "사용 연구", "적용 맥락", "한계"],
        ["Jaccard Index", "|A∩B| / |A∪B|\n(집합 교집합 비율)", "본 연구\n(route, mode)", "정류장/노선 집합\n일치도 비교", "순서 무시\n→ LCS로 보완"],
        ["LCS Ratio", "최장 공통 부분수열\n/ max(|A|,|B|)", "본 연구\n(sequence)", "정류장 방문 순서\n보존 비교", "삽입/삭제 비용\n동일 가정"],
        ["Edit Distance\n(Levenshtein)", "삽입/삭제/치환\n최소 연산 횟수", "Tiam-Lee &\nHenriques (2022)", "정류장 시퀀스\n차이 정량화", "정규화 필요\n계산 O(nm)"],
        ["Hausdorff\nDistance", "max(min(d(a,b)))\n최대-최소 점간 거리", "본 연구 (기존)\nHassan et al.", "경로 기하형상\n공간적 유사도", "이상치 민감\n내생성 유발 가능"],
        ["Fréchet\nDistance", "단조 정렬 조건 하\n최대 편차", "Bai et al.\n(2023)", "GPS 궤적\n연속 경로 비교", "계산 비용 높음\nO(nm)"],
        ["DTW (Dynamic\nTime Warping)", "시간축 왜곡 허용\n최적 정렬 거리", "Yuan & Li\n(2021)", "시계열 궤적\nGPS 매칭", "대중교통 정류장\n데이터에 부적합"],
    ]
    add_table(slide, data, Inches(0.8), Inches(2.5), Inches(11.8),
              row_height=Inches(0.55),
              col_widths=[Inches(1.6), Inches(2.0), Inches(1.8), Inches(2.0), Inches(4.4)],
              header_color=COLOR_BLUE)


def make_slide_prior_research_3(prs):
    """3.3 스마트카드 기반 경로 식별 선행연구"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 3장 선행연구", "3.3 스마트카드 기반 경로 매칭/식별")

    add_content_section(slide, Inches(1.2),
        subtitle="스마트카드(AFC) 데이터로 경로를 어떻게 식별하는가?",
        body_lines=[
            "• 핵심 난제: 스마트카드는 승/하차 정보만 기록 → 실제 이용 경로(중간 구간)는 미관측",
            "• 각 연구마다 경로 식별·매칭에 서로 다른 유사도 지표와 방법론 사용",
        ])

    data = [
        ["연구", "도시/데이터", "경로 식별 방법", "유사도 지표", "주요 성과"],
        ["Hassan et al.\n(2021)", "브리즈번\nGoCard AFC", "다기준 최단경로 생성\n+ 반복 경로 제거", "소요시간 비율\n+ 환승 패턴 일치", "관측 경로 70.5% 재현\nMNL ρ²=0.29"],
        ["Tiam-Lee &\nHenriques\n(2022)", "리스본\nAFC (버스)", "시간표 없이 AFC만\n→ 경로 추론", "Edit Distance\n(정류장 시퀀스)", "시간표 불필요\n정확도 82.3%"],
        ["Dixit et al.\n(2023)", "암스테르담\nOV-chipkaart", "GTFS 기반 대안경로\n+ PSC 보정", "환승역 공유 기반\nPath Size", "다수단 PSC 최초 적용\nρ²=0.33"],
        ["Hussain et al.\n(2021)", "(리뷰 논문)\n17개국 사례", "환승감지 규칙:\n시간<1hr, 거리<500m", "시공간 근접성\n(threshold 기반)", "도착지 추론 60~75%\n환승감지 정확도 정리"],
        ["Yap et al.\n(2025)", "네덜란드\nOV-chipkaart", "Neural Network\nvs PSL", "학습 기반\n(비선형 효용)", "NN이 PSL 능가\n비관측 이질성 반영"],
    ]
    add_table(slide, data, Inches(0.8), Inches(3.0), Inches(11.8),
              row_height=Inches(0.65),
              col_widths=[Inches(1.8), Inches(1.8), Inches(2.2), Inches(2.0), Inches(4.0)],
              header_color=COLOR_BLUE)


def make_slide_prior_research_4(prs):
    """3.4 선행연구 종합 비교 및 본 연구 포지셔닝"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 3장 선행연구", "3.4 선행연구 종합 및 본 연구 위치")

    add_content_section(slide, Inches(1.2),
        subtitle="선행연구 Gap 분석",
        body_lines=[
            [('① 경로 중첩 보정 (PSL 계열)', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (': 링크/환승역 기반 중첩도 → 대안경로 생성 전제 (Choice Set 필요)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('② 유사도 지표', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (': Jaccard, LCS, Hausdorff 등 개별 지표 연구 → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('복합 유사도의 최적 가중치 연구 부재', FONT_MEDIUM, Pt(14), COLOR_RED, True)],
            [('③ 스마트카드 경로 식별', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (': 단일 지표(시간/거리) 매칭 → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('다차원 유사도 기반 확률적 매칭 미시도', FONT_MEDIUM, Pt(14), COLOR_RED, True)],
        ])

    add_content_section(slide, Inches(3.5),
        subtitle="본 연구의 차별점",
        body_lines=[])

    data = [
        ["구분", "기존 선행연구", "본 연구"],
        ["경로 매칭 방식", "단일 지표 (시간 비율 or 최단경로)", "복합 유사도 (Mode + Route + Seq)"],
        ["유사도 가중치", "임의 설정 or 미사용", "2-Stage Grid Search 최적화 (462개 시나리오)"],
        ["공간적 유사도", "Hausdorff 등 공간 지표 사용", "제거 (내생성 유발 → MNL 계수 왜곡)"],
        ["선택확률 부여", "최단경로 = 선택 (이진)", "유사도 기반 연속 확률 (softmax)"],
        ["데이터 규모", "수천~수만 통행", "53만 OD, 160만 대안, 3,700만 통행"],
        ["검증", "ρ² 또는 정확도 단일 지표", "ρ² + FPR-1 + FPR-3 + 수단분담률 + β비율"],
    ]
    add_table(slide, data, Inches(0.8), Inches(4.2), Inches(11.8),
              row_height=Inches(0.38),
              col_widths=[Inches(2.0), Inches(4.4), Inches(5.4)],
              header_color=COLOR_BLUE)


def make_slide_old_method(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 기존 방법론", "4.1 기존 유사도 방법론 (5-Level Composite)")

    add_content_section(slide, Inches(1.2),
        subtitle="기존 Composite 산출식",
        sub_subtitle="S = 0.10×Mode + 0.15×Route + 0.15×Sequence + 0.10×Time + 0.50×Spatial",
        body_lines=[])

    data = [
        ["Component", "Weight", "Metric", "설명"],
        ["Mode", "0.10", "mode_jaccard", "교통수단 집합 (bus/train/GTX)"],
        ["Route", "0.15", "route_jaccard", "노선명 집합 (2호선, 5530번 등)"],
        ["Sequence", "0.15", "seq_lcs", "정류장 순서 (LCS ratio)"],
        ["Time", "0.10", "time_ratio", "소요시간 비율"],
        ["Spatial", "0.50", "avg_hausdorff", "정류장 좌표 Hausdorff 거리"],
    ]
    add_table(slide, data, Inches(0.8), Inches(2.5), Inches(5.5),
              row_height=Inches(0.38),
              col_widths=[Inches(1.3), Inches(0.8), Inches(1.5), Inches(1.9)])

    add_content_section(slide, Inches(5.2),
        subtitle="매칭 조건",
        body_lines=[
            "• Threshold: composite ≥ 0.6  |  Sequence Gate: 없음  |  최고 점수 → chosen = 1",
            [('• Spatial이 유사도의 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('50%를 결정', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (' → 정류장 좌표 간 Hausdorff 거리 기반', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])


def make_slide_metric_mode_route(prs):
    """4.2 Mode & Route 유사도 산출 방식"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 기존 방법론", "4.2 Mode · Route 유사도 산출")

    # --- Mode ---
    add_content_section(slide, Inches(1.2),
        subtitle="Mode Similarity (mode_jaccard) — 교통수단 집합 비교",
        body_lines=[
            "• SC 통행의 수단 집합과 OTP 대안의 수단 집합을 Jaccard Index로 비교",
            [('• Jaccard = |SC ∩ OTP| / |SC ∪ OTP|', FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
            "• 수단 분류: bus, train(지하철/철도), gtx → 7개 카테고리 (bus_only, train_only, ...)",
        ])

    data_mode = [
        ["", "SC 수단", "OTP 수단", "교집합", "합집합", "Jaccard"],
        ["예시 1", "{bus, train}", "{bus, train}", "2", "2", "1.00"],
        ["예시 2", "{bus}", "{bus, train}", "1", "2", "0.50"],
        ["예시 3", "{train}", "{gtx}", "0", "2", "0.00"],
    ]
    add_table(slide, data_mode, Inches(0.8), Inches(3.1), Inches(8.5),
              row_height=Inches(0.32),
              col_widths=[Inches(1.0), Inches(1.5), Inches(1.5), Inches(1.2), Inches(1.2), Inches(2.1)])

    # --- Route ---
    add_content_section(slide, Inches(4.6),
        subtitle="Route Similarity (route_jaccard) — 노선명 집합 비교 + GTFS 보정",
        body_lines=[
            [('• 노선명 정규화 후 Jaccard 비교', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('• 정규화 규칙:', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (' OTP "서울2호선" → "2호선"  |  SC "5531번(군포동행정복지센터방면)" → "5531"', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('• GTFS route_short_name 기반으로 OTP/SC 노선명을 통일하여 매칭 정확도 확보', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    data_route = [
        ["", "SC 노선 (정규화 후)", "OTP 노선 (정규화 후)", "Jaccard"],
        ["예시 1", "{2호선, 5531}", "{2호선, 5531}", "1.00"],
        ["예시 2", "{2호선, 5531}", "{2호선, 5530}", "1/3 = 0.33"],
        ["예시 3", "{7호선}", "{2호선}", "0.00"],
    ]
    add_table(slide, data_route, Inches(0.8), Inches(6.0), Inches(9.5),
              row_height=Inches(0.30),
              col_widths=[Inches(1.0), Inches(2.8), Inches(2.8), Inches(2.9)])


def make_slide_metric_sequence(prs):
    """4.3 Sequence 유사도 산출 방식 — GTFS 보정 상세"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 기존 방법론", "4.3 Sequence 유사도 산출 (GTFS 보정)")

    add_content_section(slide, Inches(1.2),
        subtitle="문제: OTP/SC 정류장 정보의 차이",
        body_lines=[
            "• OTP: leg별 출발/도착 정류장만 기록 (중간 정류장 없음)",
            "• SC(스마트카드): 승차/하차 정류장만 기록 (환승 시 중간 정류장 없음)",
            [('→ 양쪽 모두 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('중간 경유 정류장이 누락', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (' → 직접 비교 시 정보 손실', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    add_content_section(slide, Inches(3.0),
        subtitle="해결: GTFS 기반 정류장 시퀀스 확장 (expand_route)",
        body_lines=[
            [('• GTFS stop_times.txt에서 노선별 전체 정류장 순서를 조회', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('• OTP leg의 from→to 구간을 GTFS로 확장하여 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('중간 정류장 복원', FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
            [('• SC도 동일하게 노선+승하차 정류장으로 GTFS 확장', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• 정류장명 fuzzy 매칭: OTP '군포1동행정복지센터.군포역' ↔ SC '군포역' → 부분일치 허용",
        ])

    # 예시 테이블
    data_seq = [
        ["단계", "OTP (2호선 leg)", "SC (2호선 승하차)"],
        ["원본", "from=강남, to=삼성 (2개)", "승차=강남, 하차=삼성 (2개)"],
        ["GTFS 확장", "[강남, 역삼, 선릉, 삼성] (4개)", "[강남, 역삼, 선릉, 삼성] (4개)"],
        ["LCS 비교", "LCS = 4 / max(4,4) = 1.00", "완전 일치 ✓"],
    ]
    add_table(slide, data_seq, Inches(0.8), Inches(5.2), Inches(11.5),
              row_height=Inches(0.35),
              col_widths=[Inches(1.5), Inches(5.0), Inches(5.0)])

    # LCS 설명 박스
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.7),
                                 Inches(11.5), Inches(0.7))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
    box.line.color.rgb = RGBColor(0x93, 0xB5, 0xE1)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(),
            "seq_lcs = LCS(OTP확장, SC확장) / max(|OTP확장|, |SC확장|)  →  순서 보존하면서 공통 정류장 비율 측정",
            FONT_MEDIUM, Pt(12), COLOR_HEADER, True)


def make_slide_metric_composite(prs):
    """4.4 Composite 산출 및 choice_prob 변환"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 기존 방법론", "4.4 Composite 산출 → choice_prob 변환")

    add_content_section(slide, Inches(1.2),
        subtitle="Step 1: 개별 통행 매칭 — SC 1건 × OTP K개 대안",
        body_lines=[
            [('• Composite(i) = w_mode × mode_jaccard + w_route × route_jaccard + w_seq × seq_lcs',
              FONT_MEDIUM, Pt(13), COLOR_BLUE, True)],
            "• 각 SC 통행에 대해 K개 OTP 대안의 composite score 산출",
            [('• Gate: seq_lcs < 0.3 → 제외 (우회/왕복 필터)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('• Threshold: composite < 0.5 → 제외 (저품질 매칭)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• 최고 점수 대안 → 해당 SC 통행의 매칭 경로 (n_matched +1)",
        ])

    add_content_section(slide, Inches(3.7),
        subtitle="Step 2: OD 집계 → choice_prob",
        body_lines=[
            "• 동일 OD(출발-도착 정류장 쌍)의 SC 통행을 대안별로 집계",
            [('• choice_prob(i) = n_matched(i) / n_total', FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             ('  (대안 i에 매칭된 통행 비율)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• Σ choice_prob = 1.0 → MNL의 종속변수(y)로 사용",
        ])

    # 예시 테이블
    data_cp = [
        ["OD: 강남→종로 (n_total=100)", "대안 1\n(2호선→1호선)", "대안 2\n(3호선 직통)", "대안 3\n(버스 7016)"],
        ["composite score", "0.95", "0.82", "0.61"],
        ["n_matched", "60건", "30건", "10건"],
        ["choice_prob", "0.60", "0.30", "0.10"],
    ]
    add_table(slide, data_cp, Inches(0.8), Inches(5.5), Inches(11.5),
              row_height=Inches(0.35),
              col_widths=[Inches(3.5), Inches(2.6), Inches(2.6), Inches(2.8)])

    # 하단 요약
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.9),
                                 Inches(11.5), Inches(0.5))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    box.line.color.rgb = RGBColor(0x66, 0xBB, 0x6A)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(),
            "SC 37M trips → OD 530K × 대안 평균 3.7개 = 1.97M rows → 80/20 층화 Split → MNL β 추정",
            FONT_MEDIUM, Pt(12), COLOR_HEADER, True)


def make_slide_matching_pipeline(prs):
    """6.3 매칭 파이프라인 — choice_prob 산출 과정"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 6장 방법론 개선", "6.3 매칭 파이프라인: SC → choice_prob → MNL")

    add_content_section(slide, Inches(1.2),
        subtitle="Step 1: 개별 통행 매칭 (37M trips × K alternatives)",
        body_lines=[
            "• 스마트카드 1건 → OTP 대안 K개 (평균 3.7개/OD) 각각과 유사도 비교",
            "• Composite = 0.02×Mode + 0.90×Route + 0.08×Sequence 산출",
            [('• Gate: sim_sequence < 0.3이면 해당 대안 ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('제외', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (' (우회/왕복 필터링)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• Threshold: composite ≥ 0.5인 대안만 유효 매칭 후보",
        ])

    add_content_section(slide, Inches(3.3),
        subtitle="Step 2: OD 집계 → choice_prob 산출",
        body_lines=[
            "• 동일 OD의 SC 통행을 대안별로 집계: n_matched(대안 i) = 해당 대안에 매칭된 통행 수",
            [('• choice_prob(i) = n_matched(i) / n_total', FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
            "• Σ choice_prob = 1.0 (OD 내 확률 분포)",
            "• 530,689 OD → 1,973,353 대안 rows (training set)",
        ])

    add_content_section(slide, Inches(5.2),
        subtitle="Step 3: MNL 추정",
        body_lines=[
            [('• y = choice_prob (연속 확률)', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('  X = IVT, walk, transfers, dist, mode ASC 등 10개 피처', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('• 가중치: n_total (통행 수가 많은 OD에 높은 가중)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [('• L-BFGS-B 최적화, β≤0 제약 (시간/비용/거리/환승)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    # Flow diagram as text box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5),
                                 Inches(11.5), Inches(0.6))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    box.line.color.rgb = RGBColor(0x66, 0xBB, 0x6A)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(),
            "SC 통행 → 유사도 산출 → Gate+Threshold 필터 → OD 집계(choice_prob) → 80/20 층화 Split → MNL β 추정",
            FONT_MEDIUM, Pt(12), COLOR_HEADER, True)


def make_slide_problems(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 5장 문제점 분석", "5.1 기존 방법론의 내생성 (Endogeneity)")

    add_content_section(slide, Inches(1.2),
        subtitle="상관관계 분석",
        sub_subtitle="Spatial 유사도가 도보시간과 상관 → 종속변수에 설명변수 정보 유입",
        body_lines=[
            [('• choice_prob ↔ sim_composite:  ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('r = +0.76', FONT_MEDIUM, Pt(14), COLOR_RED, True)],
            "• sim_composite ↔ access_time:  r = -0.37",
            "• sim_composite ↔ egress_time:  r = -0.37",
            "• choice_prob ↔ in_vehicle_time: r = +0.02 (거의 무관)",
        ])

    add_content_section(slide, Inches(3.5),
        subtitle="인과 경로 및 β 계수 왜곡",
        body_lines=[
            "도보시간 짧음 → 정류장이 OTP 경로에 가까움 → Spatial↑ → Composite↑ → choice_prob↑",
            [('→ β_walk 과대추정: access/IVT = ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('165배', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (', egress/IVT = ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('155배', FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (' (이론적 기대: 2~5배)', FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    # Warning box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.3),
                                 Inches(11.5), Inches(1.5))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFD, 0xE8, 0xE8)
    box.line.color.rgb = RGBColor(0xF5, 0xA0, 0xA0)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "문제의 본질", FONT_MEDIUM, Pt(14), COLOR_HEADER, True)
    add_paragraph(tf, "종속변수(y=choice_prob)에 설명변수(X=walk time) 정보가 매칭 메커니즘(Spatial similarity)을 통해 이미 반영됨",
                  FONT_REGULAR, Pt(13), COLOR_BODY)
    add_paragraph(tf, "→ MNL 계수의 행태적 해석 불가능 → Spatial 제거 필요", FONT_MEDIUM, Pt(13), COLOR_RED, True)


def make_slide_new_method(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 6장 방법론 개선", "6.1 개선된 방법론 (3-Level + Sequence Gate)")

    add_content_section(slide, Inches(1.2),
        subtitle="개선된 Composite 산출식",
        sub_subtitle="S = 0.02×Mode + 0.90×Route + 0.08×Sequence  |  Gate: seq_lcs ≥ 0.3",
        body_lines=[])

    data = [
        ["", "기존 (5-Level)", "개선 (3-Level)", "변경 사유"],
        ["Mode", "0.10", "0.02", "Route에 이미 내포"],
        ["Route", "0.15", "0.90 ★", "경로 구분의 핵심"],
        ["Sequence", "0.15", "0.08", "경유지 감별 + 순서"],
        ["Time", "0.10", "제거 (진단용)", "변별력 부족"],
        ["Spatial", "0.50", "제거 (진단용)", "내생성 유발원"],
        ["Threshold", "0.6", "0.5", "OD 수 확보"],
        ["Seq Gate", "없음", "≥ 0.3", "우회 경로 필터링"],
    ]
    add_table(slide, data, Inches(0.8), Inches(2.5), Inches(6.5),
              row_height=Inches(0.38),
              col_widths=[Inches(1.3), Inches(1.5), Inches(1.5), Inches(2.2)])

    # Right side key changes
    add_content_section(slide, Inches(2.5),
        subtitle="",  # skip subtitle, manual positioning
        body_lines=[])

    tx = slide.shapes.add_textbox(Inches(7.8), Inches(2.5), Inches(4.8), Inches(4.5))
    tf = tx.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "핵심 변경 사항", FONT_MEDIUM, Pt(14), COLOR_HEADER, True)

    add_mixed_paragraph(tf, [
        ("① Spatial 제거: ", FONT_MEDIUM, Pt(12), COLOR_BLUE, True),
        ("도보시간 상관 차단 → 내생성 해소", FONT_REGULAR, Pt(12), COLOR_BODY, False),
    ], space_before=Pt(10))
    add_mixed_paragraph(tf, [
        ("② Route 90%: ", FONT_MEDIUM, Pt(12), COLOR_BLUE, True),
        ("한국 대중교통 특성상 노선명 = 정류장 집합", FONT_REGULAR, Pt(12), COLOR_BODY, False),
    ], space_before=Pt(8))
    add_mixed_paragraph(tf, [
        ("③ Seq Gate ≥0.3: ", FONT_MEDIUM, Pt(12), COLOR_BLUE, True),
        ("우회/왕복 1,064건 제거 (정밀도 100%)", FONT_REGULAR, Pt(12), COLOR_BODY, False),
    ], space_before=Pt(8))
    add_mixed_paragraph(tf, [
        ("④ 진단 보존: ", FONT_MEDIUM, Pt(12), COLOR_BLUE, True),
        ("Time, Spatial은 진단용으로 저장", FONT_REGULAR, Pt(12), COLOR_BODY, False),
    ], space_before=Pt(8))


def make_slide_sequence_gate(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 6장 방법론 개선", "6.2 Sequence Gate: 우회 경로 필터링")

    add_content_section(slide, Inches(1.2),
        subtitle="문제 상황: 노선명이 같지만 방향/경유지가 다른 경우",
        sub_subtitle="예시: SC 강남→종로→강남(왕복) vs OTP 강남→종로(편도)",
        body_lines=[
            [('route_jaccard = 1.0 (같은 노선!) → 매칭됨   ', FONT_REGULAR, Pt(14), COLOR_RED, False)],
            [('seq_lcs = 0.5 (다른 방향!) → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('Gate가 감지 ✓', FONT_MEDIUM, Pt(14), COLOR_ACCENT_GREEN, True)],
            "• sim_sequence ≥ 0.3 이상만 유효 매칭으로 인정",
        ])

    add_content_section(slide, Inches(3.5),
        subtitle="Sequence Gate Sweep 결과",
        body_lines=[])

    data = [
        ["Seq Gate", "ODs", "test ρ²", "test FPR"],
        ["0.0 (없음)", "460,902", "0.5205", "77.5%"],
        ["0.1", "454,514", "0.5203", "77.5%"],
        ["0.2", "444,132", "0.5197", "77.4%"],
        ["0.3 ★", "436,841", "0.5208", "77.5%"],
        ["0.4", "432,473", "0.5233", "77.7%"],
        ["0.5", "425,921", "0.5251", "77.9%"],
    ]
    add_table(slide, data, Inches(0.8), Inches(4.3), Inches(6.0),
              row_height=Inches(0.35),
              col_widths=[Inches(1.2), Inches(1.3), Inches(1.3), Inches(2.2)])

    # Conclusion
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(4.3),
                                 Inches(5.0), Inches(1.5))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    box.line.color.rgb = RGBColor(0x66, 0xBB, 0x6A)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "결론", FONT_MEDIUM, Pt(13), RGBColor(0x2E, 0x7D, 0x32), True)
    add_paragraph(tf, "• 전 구간 ρ²/FPR 거의 동일 (spread < 0.005)", FONT_REGULAR, Pt(12), COLOR_BODY)
    add_paragraph(tf, "• 0.3 선택: 필터링 목적 달성 + OD 손실 0.2%", FONT_REGULAR, Pt(12), COLOR_BODY)


def make_slide_grid_search(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 7장 가중치 최적화", "7.1 2-Stage Grid Search")

    add_content_section(slide, Inches(1.2),
        subtitle="2단계 Grid Search",
        sub_subtitle="mode = 1 - route - sequence (제약 조건), 각 시나리오별 MNL 추정",
        body_lines=[])

    data_stage = [
        ["단계", "범위", "Step", "시나리오 수", "결과"],
        ["Coarse", "route [0,1] × seq [0,1]", "0.05", "231", "ρ² 0.459~0.506"],
        ["Fine", "route [0.80,1.00] × seq [0.00,0.20]", "0.01", "231", "ρ² 0.496~0.522"],
    ]
    add_table(slide, data_stage, Inches(0.8), Inches(2.6), Inches(11.5),
              row_height=Inches(0.38),
              col_widths=[Inches(1.2), Inches(3.5), Inches(1.0), Inches(1.5), Inches(4.3)])

    add_content_section(slide, Inches(3.8),
        subtitle="Fine Grid 결과 분석",
        body_lines=[
            "• 클러스터 A: route ≥ 0.96 → 최고 ρ² 0.497이나 seq 비중 부족 (방어 불가)",
            [('• 클러스터 B: route 0.84~0.91 → ρ² 유사 (0.005 차이), ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('균형 잡힌 가중치', FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
        ])

    add_content_section(slide, Inches(5.2),
        subtitle="최종 선택",
        body_lines=[])

    data_final = [
        ["Route", "Seq", "Mode", "ρ²", "FPR", "ODs"],
        ["0.90", "0.08", "0.02", "0.520", "77.5%", "437K"],
    ]
    add_table(slide, data_final, Inches(0.8), Inches(5.9), Inches(7.0),
              row_height=Inches(0.38),
              col_widths=[Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(2.0)],
              header_color=COLOR_BLUE)


def make_slide_mnl_comparison(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 8장 결과 비교", "8.1 MNL 모형 결과 비교")

    add_content_section(slide, Inches(1.2),
        subtitle="이전 vs 현재 MNL 성능",
        sub_subtitle="유사도 방법론 변경에 따른 모형 성능 비교",
        body_lines=[])

    data = [
        ["지표", "이전 (5-level\n0.20/0.40/0.40)", "현재 (3-level\n0.02/0.90/0.08)", "변화"],
        ["Train ρ²", "0.4746", "0.4925", "+0.018 ↑"],
        ["Test ρ²", "0.4724", "0.4870", "+0.015 ↑"],
        ["FPR-1 (1순위)", "70.3%", "70.9%", "+0.6%p ↑"],
        ["FPR-3 (TOP-3)", "96.2%", "96.1%", "-0.1%p"],
        ["ODs (train)", "478,345", "424,551", "-53,794"],
    ]
    add_table(slide, data, Inches(0.8), Inches(2.6), Inches(11.5),
              row_height=Inches(0.42),
              col_widths=[Inches(2.5), Inches(3.2), Inches(3.2), Inches(2.6)])

    add_content_section(slide, Inches(5.4),
        subtitle="β 계수 비율 개선 (walk / IVT)",
        body_lines=[
            [('• access_walk / IVT: 165배 → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('119배 (개선)', FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
            [('• egress_walk / IVT: 155배 → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('73배 (크게 개선)', FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
            [('• transfer_walk / IVT: 25배 → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('2.3배 (거의 해소)', FONT_MEDIUM, Pt(14), COLOR_ACCENT_GREEN, True)],
        ])


def make_slide_mnl_parameters(prs):
    """8.2 MNL β 계수 및 수단분담률 재현"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 8장 결과 비교", "8.2 MNL β 계수 추정 결과")

    add_content_section(slide, Inches(1.2),
        subtitle="β 계수 (3-level, route=0.90, seq=0.08, mode=0.02)  |  fare 제거 (10개 피처)",
        sub_subtitle="Train: 424,549 ODs / 1,578,855 rows  |  Test: 106,140 ODs / 394,498 rows",
        body_lines=[])

    # β coefficient table (fare 제거, gtx_only OD 2개 test 이동)
    data_beta = [
        ["Feature", "β", "Std.Err", "t-stat", "부호"],
        ["IVT (min)", "-0.00704", "0.000106", "-66.2", "✓ (−)"],
        ["Wait (min)", "0.000 ★", "—", "—", "bound"],
        ["Access walk (min)", "-0.8349", "0.000491", "-1699", "✓ (−)"],
        ["Egress walk (min)", "-0.5122", "0.000326", "-1570", "✓ (−)"],
        ["Transfer walk (min)", "-0.0163", "0.000506", "-32.1", "✓ (−)"],
        ["Total dist (km)", "-0.0433", "0.000348", "-125", "✓ (−)"],
        ["Transfers", "-3.6575", "0.001874", "-1952", "✓ (−)"],
        ["Has bus (ASC)", "-2.2278", "0.001815", "-1228", "✓ (−)"],
        ["Has train (ASC)", "+2.9852", "0.002749", "+1086", "✓ (+)"],
        ["Has GTX (ASC)", "+0.1060", "0.025368", "+4.18", "✓ (+)"],
    ]
    add_table(slide, data_beta, Inches(0.3), Inches(2.5), Inches(7.8),
              row_height=Inches(0.30),
              col_widths=[Inches(2.2), Inches(1.2), Inches(1.2), Inches(1.2), Inches(2.0)],
              header_color=COLOR_BLUE)

    # Mode share reproduction table (right side)
    tx_title = slide.shapes.add_textbox(Inches(8.5), Inches(2.5), Inches(4.5), Inches(0.35))
    tf_title = tx_title.text_frame
    set_run(tf_title.paragraphs[0].add_run(), "수단분담률 재현", FONT_MEDIUM, Pt(14), COLOR_HEADER, True)

    data_mode = [
        ["Category", "SC 실측\n(37M)", "Train y\n(29.6M)", "MNL 예측\n(7.4M)"],
        ["train_only", "48.84%", "48.69%", "49.39%"],
        ["bus_only", "42.91%", "43.01%", "42.54%"],
        ["bus+train", "8.15%", "8.18%", "7.95%"],
        ["train+gtx", "0.072%", "0.072%", "0.091%"],
        ["gtx_only", "0.028%", "0.036%", "0.019%"],
        ["bus+trn+gtx", "0.006%", "0.006%", "0.006%"],
        ["bus+gtx", "0.002%", "0.002%", "0.002%"],
    ]
    add_table(slide, data_mode, Inches(8.3), Inches(3.0), Inches(4.7),
              row_height=Inches(0.30),
              col_widths=[Inches(1.3), Inches(1.0), Inches(1.0), Inches(1.4)],
              header_color=COLOR_BLUE)

    # Key interpretation box (right side, below mode share)
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.5), Inches(4.8),
                                 Inches(4.2), Inches(2.0))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
    box.line.color.rgb = RGBColor(0x93, 0xB5, 0xE1)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "해석", FONT_MEDIUM, Pt(13), COLOR_BLUE, True)
    add_paragraph(tf, "• 부호 검증: ALL CORRECT", FONT_REGULAR, Pt(11), COLOR_BODY)
    add_paragraph(tf, "• Train/Test ρ² 비율 0.989 → 과적합 없음", FONT_REGULAR, Pt(11), COLOR_BODY)
    add_paragraph(tf, "• GTX 분담률 0.1% 미만 (10.5K trips)", FONT_REGULAR, Pt(11), COLOR_BODY)
    add_paragraph(tf, "• fare/wait = 0 (bound) → 이슈 분석 별도", FONT_REGULAR, Pt(11), COLOR_BODY)

    # Bottom note
    tx_note = slide.shapes.add_textbox(Inches(0.3), Inches(6.6), Inches(8.0), Inches(0.5))
    tf_note = tx_note.text_frame; tf_note.word_wrap = True
    add_mixed_paragraph(tf_note, [
        ("★ ", FONT_MEDIUM, Pt(11), COLOR_RED, True),
        ("bound: 비음 제약(β≤0) 하에서 0에 고정. wait는 다른 변수에 흡수. fare는 통합요금제로 제거.", FONT_REGULAR, Pt(11), COLOR_TABLE_TEXT, False),
    ], space_before=Pt(0))


def make_slide_remaining_issues(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 8장 결과 비교", "8.2 남은 이슈 분석")

    add_content_section(slide, Inches(1.2),
        subtitle="fare β = 0 → 제거 예정",
        body_lines=[
            "• 60.8% OD: 대안 간 요금 차이 = 0원 (수도권 통합요금제)",
            "• 81.7% OD: 요금 차이 ≤ 50원 → 변별력 없음",
        ])

    add_content_section(slide, Inches(2.7),
        subtitle="wait_time β = 0 → 유지",
        body_lines=[
            "• 분산 있음 (87.9% OD에서 std > 0), 부호 방향 정상",
            "• 다른 변수에 설명력 흡수 → 데이터 개선 시 활성화 가능",
        ])

    add_content_section(slide, Inches(4.2),
        subtitle="walk 과대 / IVT 과소 → 논문 한계 기술",
        body_lines=[
            "• 가중치 변경으로 상당히 완화되었으나 access/egress 여전히 과대",
            "• MNL의 구조적 한계 (선형 효용) → 딥러닝 기반 모형으로 개선 예정",
        ])

    add_content_section(slide, Inches(5.6),
        subtitle="GTX 과추정 → 크게 개선",
        body_lines=[
            [('• has_gtx ASC: +0.469 → ', FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ('+0.106 (78% 감소)', FONT_MEDIUM, Pt(14), COLOR_ACCENT_GREEN, True)],
            "• GTX 포함 경로 1.00%, 평균 choice_prob 0.030",
        ])


def make_slide_future(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 9장 향후 계획", "9.1 향후 연구 계획")

    add_content_section(slide, Inches(1.2),
        subtitle="단기 과제",
        body_lines=[
            "• fare 변수 제거 후 MNL 재추정",
            "• 논문 (main-kr.tex) 수정: 가중치, 성능지표, 방법론 설명 업데이트",
            "• walk/IVT 내생성 한계점 논문에 명시",
        ])

    add_content_section(slide, Inches(3.0),
        subtitle="중기 과제: 딥러닝 기반 경로선택 모형",
        sub_subtitle="Yap, Cats & Clouaire (2025) - Neural network이 PSL보다 예측력 우수",
        body_lines=[
            "• MNL의 선형 효용 한계 극복 → 비선형 효용 + 비관측 이질성 반영",
            "• 날씨, 인구통계 등 맥락 변수 통합 가능",
        ])

    add_content_section(slide, Inches(4.8),
        subtitle="장기 비전",
        body_lines=[
            "• DTUMOS 시뮬레이션 파라미터 매핑 (β → simulation params)",
            "• 실시간 스마트카드 데이터 기반 모형 업데이트 파이프라인",
        ])


def make_slide_thank_you(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[2])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG_BLUE
    bg.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(1.5), Inches(2.5), Inches(10), Inches(2.0))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "감사합니다", FONT_BOLD, Pt(36), COLOR_WHITE, True)
    add_paragraph(tf, "Thank You", FONT_THIN, Pt(18), RGBColor(0xCC, 0xCC, 0xCC),
                  alignment=PP_ALIGN.CENTER, space_before=Pt(12))


def main():
    template_path = 'd:/Folder/Research/6.route_choice_simulation/pptx/Multi_modal_Simulation.pptx'
    prs = Presentation(template_path)

    # Delete all existing slides
    while len(prs.slides) > 0:
        rId = prs.slides._sldIdLst[0].get(
            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        if rId is None:
            rId = prs.slides._sldIdLst[0].attrib.get('r:id')
        prs.part.drop_rel(rId)
        prs.slides._sldIdLst.remove(prs.slides._sldIdLst[0])

    make_title_slide(prs)             # 1
    make_contents_slide(prs)          # 2
    make_slide_bg_multimodal(prs)     # 3  NEW
    make_slide_bg_pipeline(prs)       # 4  NEW
    make_slide_bg_matching(prs)       # 5  NEW
    make_slide_bg_importance(prs)     # 6  NEW
    make_slide_background(prs)        # 7  (기존: 문제 인식)
    make_slide_prior_research_1(prs)  # 8
    make_slide_prior_research_2(prs)  # 9
    make_slide_prior_research_3(prs)  # 10
    make_slide_prior_research_4(prs)  # 11 NEW
    make_slide_old_method(prs)          # 12
    make_slide_metric_mode_route(prs)   # 13 NEW: Mode/Route 산출
    make_slide_metric_sequence(prs)     # 14 NEW: Sequence + GTFS 보정
    make_slide_metric_composite(prs)    # 15 NEW: Composite → choice_prob
    make_slide_problems(prs)            # 16
    make_slide_new_method(prs)          # 17
    make_slide_sequence_gate(prs)       # 18
    make_slide_matching_pipeline(prs)   # 19 매칭 파이프라인
    make_slide_grid_search(prs)       # 15
    make_slide_mnl_comparison(prs)    # 16
    make_slide_mnl_parameters(prs)    # 17  NEW: β 계수 + 수단분담률
    make_slide_remaining_issues(prs)  # 18
    make_slide_future(prs)            # 18
    make_slide_thank_you(prs)         # 19

    out_path = "d:/Folder/Research/6.route_choice_simulation/pptx/lab-20260317i.pptx"
    prs.save(out_path)
    print(f"PPT saved: {out_path}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
