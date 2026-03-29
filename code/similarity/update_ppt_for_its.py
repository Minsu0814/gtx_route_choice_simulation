"""ITS 발표용 PPT 보강: 기존 20260324-lab.pptx에 새 슬라이드 추가
- 스마트카드 전처리 파이프라인
- OTP 대안경로 생성 과정
- 전체 파이프라인 개요도
- 매칭 결과 통계
- 이슈 분석 보강
- 향후 계획 보강
- lab 진행상황 슬라이드 제거
- 중복 슬라이드(p15/16) 정리
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from lxml import etree
import os
import copy

# ── Style Constants (same as create_seminar_ppt.py) ──
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

CHECK_ICON_PATH = 'C:/Research/6.route_choice_simulation/pptx/check_icon.png'
ICON_SIZE = Inches(0.32)

# ── Header group XML ──
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


_next_id = 500  # High to avoid conflicts with existing slides

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
    y = y_offset
    slide.shapes.add_picture(CHECK_ICON_PATH, Inches(0.80), y, ICON_SIZE, ICON_SIZE)
    tx_sub = slide.shapes.add_textbox(Inches(1.22), y, Inches(10.31), Inches(0.32))
    tf_sub = tx_sub.text_frame
    tf_sub.word_wrap = True
    p = tf_sub.paragraphs[0]
    run = p.add_run()
    set_run(run, subtitle, FONT_MEDIUM, Pt(15), COLOR_HEADER, True)
    y_next = y + Inches(0.41)

    if sub_subtitle:
        tx_ssub = slide.shapes.add_textbox(Inches(1.22), y_next, Inches(10.31), Inches(0.32))
        tf_ssub = tx_ssub.text_frame
        tf_ssub.word_wrap = True
        p2 = tf_ssub.paragraphs[0]
        run2 = p2.add_run()
        set_run(run2, sub_subtitle, FONT_MEDIUM, Pt(15), COLOR_HEADER, True)
        y_next += Inches(0.38)

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


def add_flow_box(slide, left, top, width, height, text, fill_color, border_color, text_color=COLOR_WHITE):
    """Add a rounded rectangle with centered text."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = fill_color
    box.line.color.rgb = border_color
    box.line.width = Pt(1.5)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, text, FONT_MEDIUM, Pt(11), text_color, True)
    tf.paragraphs[0].space_before = Pt(0)
    tf.paragraphs[0].space_after = Pt(0)
    return box


def add_arrow_right(slide, left, top, width=Inches(0.4), height=Inches(0.15)):
    """Add a right-pointing arrow."""
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = COLOR_BLUE
    arrow.line.fill.background()
    return arrow


# ═══════════════════════════════════════════════════
# NEW SLIDES
# ═══════════════════════════════════════════════════

def make_slide_pipeline_overview(prs):
    """전체 파이프라인 개요도"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 1장 연구배경", "1. 전체 파이프라인 개요")

    add_content_section(slide, Inches(1.2),
        subtitle="스마트카드 기반 경로선택 모형 추정 파이프라인",
        body_lines=[
            "• 스마트카드 빅데이터로부터 경로선택 파라미터를 추정하는 End-to-End 프레임워크",
        ])

    # Flow diagram - 6 boxes with arrows
    box_w = Inches(1.65)
    box_h = Inches(0.85)
    arrow_w = Inches(0.35)
    y_center = Inches(2.8)
    start_x = Inches(0.5)
    gap = Inches(0.15)

    steps = [
        ("① 스마트카드\n전처리", RGBColor(0x2E, 0x7D, 0x32), RGBColor(0x1B, 0x5E, 0x20)),
        ("② OD 추출\n환승 복원", RGBColor(0x2E, 0x7D, 0x32), RGBColor(0x1B, 0x5E, 0x20)),
        ("③ OTP\n대안경로 생성", RGBColor(0x15, 0x65, 0xC0), RGBColor(0x0D, 0x47, 0xA1)),
        ("④ 유사도\n매칭", RGBColor(0xE6, 0x51, 0x00), RGBColor(0xBF, 0x36, 0x0C)),
        ("⑤ MNL/DL\n모형 추정", RGBColor(0x6A, 0x1B, 0x9A), RGBColor(0x4A, 0x14, 0x8C)),
        ("⑥ 시뮬레이션\n파라미터 반영", RGBColor(0x37, 0x41, 0x51), RGBColor(0x26, 0x32, 0x38)),
    ]

    for i, (text, fill, border) in enumerate(steps):
        x = start_x + i * (box_w + arrow_w + gap)
        add_flow_box(slide, x, y_center, box_w, box_h, text, fill, border)
        if i < len(steps) - 1:
            ax = x + box_w + Pt(4)
            add_arrow_right(slide, ax, y_center + box_h / 2 - Inches(0.08))

    # Detail boxes below
    details = [
        ("스마트카드 (AFC)", "• 약 3,700만 통행/7일\n• 승하차 정류장/시각 추출\n• BFS 도시철도 환승 복원\n• 노선명 정규화 (GTFS 매핑)"),
        ("OTP 경로탐색", "• GTFS + OSM 네트워크\n• OD당 최대 10개 대안경로\n• 시간대별 탐색 (첨두/비첨두)\n• 71.8만 OD → 272만 대안"),
        ("유사도 매칭", "• 3-level 복합유사도\n  (Mode + Route + Sequence)\n• Composite ≥ 0.5\n• Seq Gate ≥ 0.3\n• OD별 choice_prob 산출"),
    ]

    for i, (title, body) in enumerate(details):
        x = Inches(0.5) + i * Inches(4.3)
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(4.2),
                                     Inches(4.0), Inches(2.5))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xF5, 0xF5, 0xF5)
        box.line.color.rgb = RGBColor(0xBD, 0xBD, 0xBD)
        tf = box.text_frame
        tf.word_wrap = True
        set_run(tf.paragraphs[0].add_run(), title, FONT_MEDIUM, Pt(13), COLOR_BLUE, True)
        for line in body.split('\n'):
            add_paragraph(tf, line, FONT_REGULAR, Pt(11), COLOR_TABLE_TEXT,
                         space_before=Pt(2), space_after=Pt(1))
    return slide


def make_slide_smartcard_preprocessing(prs):
    """스마트카드 전처리 파이프라인"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 1장 연구배경", "2. 스마트카드 데이터 전처리")

    add_content_section(slide, Inches(1.2),
        subtitle="원시 데이터 구조",
        body_lines=[
            "• 수도권 교통카드 (T-money / 캐시비): 승차태그 + 하차태그",
            [("• 기록 정보: ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("승차 정류장, 승차 시각, 하차 정류장, 하차 시각, 노선번호, 수단코드", FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
            "• 환승 시 각 leg별 별도 레코드 생성 → 하나의 통행으로 연결 필요",
        ])

    add_content_section(slide, Inches(3.0),
        subtitle="전처리 단계",
        body_lines=[
            [("Step 1. 통행 연결: ", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             ("환승 시간 규칙(30분 이내) + 공간 근접성으로 leg → trip 결합", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [("Step 2. 도시철도 환승 복원: ", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             ("BFS 기반 그래프 탐색으로 지하철 내 환승 경로 복원", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            [("Step 3. 노선명 정규화: ", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             ("스마트카드 노선명 ↔ GTFS route_short_name 매핑", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "  예) SC '서울2호선(성수지선)' → '2호선',  SC '5531번(군포행정복지센터방면)' → '5531'",
            [("Step 4. OD 집계: ", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             ("출발-도착 정류장 쌍 기준으로 통행 집계 (일 13회 이상 OD만 사용)", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    # Summary table
    data = [
        ["처리 단계", "입력", "출력", "비고"],
        ["원시 데이터", "교통카드 태그", "5,200만 레코드", "7일 (평일5+주말2)"],
        ["통행 연결", "개별 leg", "3,700만 통행", "환승 규칙 적용"],
        ["OD 집계", "통행 레코드", "71.8만 OD쌍", "일 13회 이상 필터"],
        ["노선 정규화", "SC 노선명", "GTFS 노선명", "fuzzy matching"],
    ]
    add_table(slide, data, Inches(0.8), Inches(5.5), Inches(11.0),
              row_height=Inches(0.32),
              col_widths=[Inches(2.0), Inches(2.5), Inches(2.5), Inches(4.0)])
    return slide


def make_slide_otp_generation(prs):
    """OTP 대안경로 생성 과정"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 1장 연구배경", "3. OTP 대안경로 생성")

    add_content_section(slide, Inches(1.2),
        subtitle="OpenTripPlanner (OTP) 기반 경로탐색",
        body_lines=[
            "• GTFS (수도권 전체 버스·도시철도·GTX) + OSM (도보 네트워크) 기반",
            "• RAPTOR 알고리즘: 라운드별 최적 경로 탐색 → Pareto-optimal 대안경로 생성",
            "• 각 OD쌍에 대해 시간대별 최대 10개 대안경로 탐색",
        ])

    add_content_section(slide, Inches(3.0),
        subtitle="탐색 파라미터",
        body_lines=[])

    data_params = [
        ["파라미터", "값", "설명"],
        ["탐색 시간대", "07:00~09:00\n17:00~19:00", "AM/PM 첨두 시간대"],
        ["최대 환승", "3회", "현실적 환승 제한"],
        ["최대 도보", "1,500m", "접근/이탈 도보 제한"],
        ["최대 대안수", "10개/OD", "itinerary 수 제한"],
        ["수단 조합", "Bus + Rail + GTX", "멀티모달 허용"],
    ]
    add_table(slide, data_params, Inches(0.8), Inches(3.7), Inches(5.5),
              row_height=Inches(0.35),
              col_widths=[Inches(1.5), Inches(1.5), Inches(2.5)])

    # Right side - output stats
    add_content_section(slide, Inches(3.0),
        subtitle="",
        body_lines=[])

    tx = slide.shapes.add_textbox(Inches(7.0), Inches(3.7), Inches(5.5), Inches(3.0))
    tf = tx.text_frame
    tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "생성 결과", FONT_MEDIUM, Pt(15), COLOR_HEADER, True)

    stats = [
        ("탐색 OD", "717,896쌍", "(일 13회 이상)"),
        ("생성 대안경로", "2,720,686개", "(평균 3.8개/OD)"),
        ("대안당 정보", "leg별 노선, 정류장,\n시각, 소요시간, 도보거리", ""),
    ]
    for label, value, note in stats:
        add_mixed_paragraph(tf, [
            (f"• {label}: ", FONT_REGULAR, Pt(13), COLOR_BODY, False),
            (value, FONT_MEDIUM, Pt(13), COLOR_BLUE, True),
            (f" {note}" if note else "", FONT_REGULAR, Pt(12), COLOR_TABLE_TEXT, False),
        ], space_before=Pt(8))

    # Info box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(5.5),
                                 Inches(5.5), Inches(1.2))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
    box.line.color.rgb = RGBColor(0x93, 0xB5, 0xE1)
    tf2 = box.text_frame; tf2.word_wrap = True
    set_run(tf2.paragraphs[0].add_run(), "경로 속성 추출", FONT_MEDIUM, Pt(12), COLOR_BLUE, True)
    add_paragraph(tf2, "각 대안경로에서 IVT, 대기시간, 접근/이탈/환승 도보시간,\n총거리, 환승횟수, 요금, 이용 노선, 수단 조합 추출",
                  FONT_REGULAR, Pt(11), COLOR_TABLE_TEXT, space_before=Pt(4))
    return slide


def make_slide_matching_statistics(prs):
    """매칭 결과 통계"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 3장 연구방법론", "3.4 매칭 결과 통계")

    add_content_section(slide, Inches(1.2),
        subtitle="유사도 매칭 결과 요약",
        body_lines=[
            "• 총 52,137,077건의 스마트카드 통행에 대해 대안경로 매칭 수행",
            "• OD 단위 집계 후 choice_prob(선택확률) 산출",
        ])

    # Left: Matching statistics table
    data_match = [
        ["항목", "값", "비율"],
        ["전체 SC 통행", "52,137,077건", "100%"],
        ["매칭 성공 통행", "37,019,247건", "71.0%"],
        ["매칭 실패 통행", "15,117,830건", "29.0%"],
        ["전체 OD쌍", "717,896쌍", "100%"],
        ["매칭 성공 OD", "530,689쌍", "73.9%"],
        ["최종 학습 OD", "424,551쌍", "59.1%"],
        ["최종 검증 OD", "106,138쌍", "14.8%"],
    ]
    add_table(slide, data_match, Inches(0.8), Inches(2.8), Inches(5.5),
              row_height=Inches(0.35),
              col_widths=[Inches(2.0), Inches(2.0), Inches(1.5)])

    # Right: Failure reasons
    tx = slide.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.5), Inches(0.35))
    tf = tx.text_frame
    set_run(tf.paragraphs[0].add_run(), "매칭 실패 사유 분석", FONT_MEDIUM, Pt(15), COLOR_HEADER, True)
    slide.shapes.add_picture(CHECK_ICON_PATH, Inches(6.58), Inches(2.5), ICON_SIZE, ICON_SIZE)

    data_fail = [
        ["실패 사유", "OD 수", "비율"],
        ["Composite < 0.5\n(저품질 매칭)", "156,327", "83.5%"],
        ["Seq Gate < 0.3\n(우회/왕복 경로)", "24,061", "12.9%"],
        ["대안경로 미생성\n(OTP 탐색 실패)", "6,819", "3.6%"],
    ]
    add_table(slide, data_fail, Inches(7.0), Inches(3.0), Inches(5.0),
              row_height=Inches(0.45),
              col_widths=[Inches(2.5), Inches(1.2), Inches(1.3)])

    # Bottom interpretation box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8),
                                 Inches(11.5), Inches(1.2))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    box.line.color.rgb = RGBColor(0x66, 0xBB, 0x6A)
    tf2 = box.text_frame; tf2.word_wrap = True
    set_run(tf2.paragraphs[0].add_run(), "매칭 품질 평가", FONT_MEDIUM, Pt(13), RGBColor(0x2E, 0x7D, 0x32), True)
    add_paragraph(tf2, "• 71% 통행 매칭 성공 → 대규모 데이터(3,700만 건)로 통계적 안정성 확보",
                  FONT_REGULAR, Pt(12), COLOR_BODY, space_before=Pt(4))
    add_paragraph(tf2, "• 매칭 실패의 84%는 저품질(Composite < 0.5) → 보수적 임계값으로 매칭 정밀도 우선",
                  FONT_REGULAR, Pt(12), COLOR_BODY, space_before=Pt(2))
    return slide


def make_slide_issues_enhanced(prs):
    """이슈 분석 보강"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 6장 결론 및 시사점", "11. 이슈 분석")

    add_content_section(slide, Inches(1.2),
        subtitle="wait_time β = 0 → 제거",
        body_lines=[
            "• 스마트카드 특성 상 정류장 출발 시간이 버스 시간과 맞아서 반영하기에 어려움이 존재",
            "• OTP의 대기시간은 시간표 기반 → 실제 대기와 괴리",
        ])

    add_content_section(slide, Inches(2.8),
        subtitle="fare β = 0 → 제거",
        body_lines=[
            [("• 60.8% OD: 대안 간 요금 차이 = 0원", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             (" (수도권 통합요금제)", FONT_MEDIUM, Pt(14), COLOR_RED, True)],
            "• 81.7% OD: 요금 차이 ≤ 50원 → 요금이 경로선택에 영향을 주지 못하는 구조",
        ])

    add_content_section(slide, Inches(4.2),
        subtitle="도보시간 과대추정 (walk/IVT 비율)",
        body_lines=[
            "• 이론적 기대값: access walk / IVT = 2~5배",
            [("• 현재 추정값: access walk / IVT = ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("119배", FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (" (유사도 매칭의 구조적 내생성 잔존)", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• 도보시간이 짧은 경로 → 유사도 높게 평가 → choice_prob 높음 → β 과대",
            [("• 완화 방안: ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("외부 VOT(시간가치) 적용 또는 딥러닝 모형으로 비선형 효용 반영", FONT_MEDIUM, Pt(14), COLOR_BLUE, True)],
        ])
    return slide


def make_slide_future_enhanced(prs):
    """향후 계획 보강"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 6장 결론 및 시사점", "12. 향후 연구 계획")

    add_content_section(slide, Inches(1.2),
        subtitle="시간대별 모형",
        body_lines=[
            "• 출근/퇴근/비첨두 분리 추정. 시간대별로 베타가 다를 수 있음",
        ])

    add_content_section(slide, Inches(2.3),
        subtitle="Nested Logit / Mixed Logit",
        body_lines=[
            "• MNL의 IIA 가정 완화 → 유사 경로 간 상관관계 반영",
            "• 버스전용 / 철도전용 / 혼합수단 그룹 간 nest 구조 도입",
        ])

    add_content_section(slide, Inches(3.6),
        subtitle="딥러닝 기반 경로선택 모형",
        body_lines=[
            "• TasteNet, ResLogit, DNN, ASU-DNN 등 5개 모형 비교 완료",
            [("• ASU-DNN 최고 성능: ρ² ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("0.6856", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (", Top-1 ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("75.99%", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (" (MNL 대비 +29% ρ² 향상)", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• 비선형 효용 + Alternative-Specific 구조로 walk/IVT 내생성 완화 기대",
        ])

    add_content_section(slide, Inches(5.2),
        subtitle="시뮬레이션 파라미터 매핑",
        body_lines=[
            "• 추정된 β → DTUMOS 시뮬레이터 경로선택 파라미터 반영",
            "• GTX-B/C 등 신규 노선 개통 시 수요 전환 시나리오 분석",
        ])
    return slide


def make_slide_dl_prior_research(prs):
    """선행연구: 딥러닝 기반 경로선택 모형"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 모형 추정", "선행연구: 딥러닝 기반 경로선택 모형")

    add_content_section(slide, Inches(1.2),
        subtitle="MNL의 한계와 딥러닝 접근",
        body_lines=[
            "• MNL: 선형 효용함수(V = Xβ) 가정 → 비선형 선호, 개인별 이질성 반영 불가",
            [("• 최근 연구 흐름: ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("이산선택모형 + 딥러닝 결합", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (" → 해석력 유지 + 예측력 향상", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])

    data = [
        ["모형", "제안 연구", "핵심 아이디어", "구조", "특징"],
        ["TasteNet", "Sifringer et al.\n(2020)", "맥락에 따라 β가 변하는\n이질적 선호 학습", "context → β(z)\nV = X · β(z)", "해석 가능\n맥락 의존적 β"],
        ["ResLogit", "Wong &\nFarooq (2021)", "MNL 선형 효용 위에\n비선형 잔차 학습", "V = Xβ_MNL\n+ DNN(X)", "MNL 확장\n잔차 비중 분석"],
        ["L-MNL", "Sifringer et al.\n(2020)", "DNN으로 잠재변수 생성\n→ MNL 피처 확장", "Z = DNN(X)\nV = [X,Z]β", "잠재변수 해석\nMNL 구조 유지"],
        ["ASU-DNN", "Han et al.\n(2022)", "대안별 독립 신경망으로\n대안 특이적 효용 학습", "V_j = DNN_j(X)\n(대안별 별도)", "대안별 이질성\n최고 성능"],
        ["DNN", "—", "순수 신경망\n비선형 효용 학습", "X → DNN\n→ utility", "블랙박스\n예측력 극대화"],
    ]
    add_table(slide, data, Inches(0.3), Inches(2.8), Inches(12.5),
              row_height=Inches(0.60),
              col_widths=[Inches(1.3), Inches(1.5), Inches(3.0), Inches(2.5), Inches(4.2)],
              header_color=COLOR_BLUE)

    # Key insight box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.0),
                                 Inches(11.5), Inches(0.8))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
    box.line.color.rgb = RGBColor(0x93, 0xB5, 0xE1)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "본 연구", FONT_MEDIUM, Pt(12), COLOR_BLUE, True)
    add_paragraph(tf, "위 5개 모형을 동일한 데이터·피처·평가지표로 공정 비교 → MNL 대비 어느 구조가 대중교통 경로선택에 적합한지 검증",
                  FONT_REGULAR, Pt(12), COLOR_BODY, space_before=Pt(2))
    return slide


def make_slide_dl_architecture(prs):
    """딥러닝 모형 구조 비교"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 모형 추정", "딥러닝 모형 구조 비교")

    # Left: Architecture diagrams as text-based flow
    models = [
        ("TasteNet", "맥락 의존 β",
         RGBColor(0x1B, 0x5E, 0x20),
         "Context (OD거리, 대안수)\n→ taste_net (32)\n→ β(z) [9개]\n→ V = X · β(z)"),
        ("ResLogit", "MNL + 비선형 잔차",
         RGBColor(0x0D, 0x47, 0xA1),
         "V_linear = X · β_MNL\nV_residual = DNN(X)\n→ V = V_linear + V_residual\n(β_MNL은 MNL 추정값 초기화)"),
        ("ASU-DNN", "대안별 독립 신경망",
         RGBColor(0xE6, 0x51, 0x00),
         "대안 1: X → DNN₁ → V₁\n대안 2: X → DNN₂ → V₂\n...\n대안 K: X → DNNₖ → Vₖ\n(각 대안별 별도 가중치)"),
    ]

    for i, (name, desc, color, arch) in enumerate(models):
        x = Inches(0.5) + i * Inches(4.2)
        # Title box
        title_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                           x, Inches(1.3), Inches(3.8), Inches(0.5))
        title_box.fill.solid()
        title_box.fill.fore_color.rgb = color
        title_box.line.fill.background()
        tf = title_box.text_frame
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        set_run(p.add_run(), f"{name} — {desc}", FONT_MEDIUM, Pt(13), COLOR_WHITE, True)

        # Architecture box
        arch_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                          x, Inches(1.9), Inches(3.8), Inches(1.6))
        arch_box.fill.solid()
        arch_box.fill.fore_color.rgb = RGBColor(0xF5, 0xF5, 0xF5)
        arch_box.line.color.rgb = color
        tf2 = arch_box.text_frame; tf2.word_wrap = True
        for j, line in enumerate(arch.split('\n')):
            if j == 0:
                set_run(tf2.paragraphs[0].add_run(), line, FONT_REGULAR, Pt(11), COLOR_BODY)
            else:
                add_paragraph(tf2, line, FONT_REGULAR, Pt(11), COLOR_BODY,
                            space_before=Pt(2), space_after=Pt(1))

    # Common design table
    add_content_section(slide, Inches(3.8),
        subtitle="공통 설계 (공정 비교를 위한 통일)",
        body_lines=[])

    data_common = [
        ["항목", "설정"],
        ["입력 피처", "MNL과 동일 9개 (IVT, Access/Egress walk, Transfer walk, Transfers, Fare, Has bus/train/GTX)"],
        ["정규화", "StandardScaler (학습셋 기준 fitting)"],
        ["손실함수", "Weighted Cross-Entropy (OD별 통행수 가중)"],
        ["최적화", "Adam (lr=1e-3) + ReduceLROnPlateau + Early Stopping (patience=15)"],
        ["데이터 분할", "OD 기준 층화분할 80/20 (수단 카테고리 기준 strata)"],
        ["최대 대안수", "5개 (부족분 패딩 + 마스크 처리)"],
    ]
    add_table(slide, data_common, Inches(0.5), Inches(4.5), Inches(12.0),
              row_height=Inches(0.30),
              col_widths=[Inches(2.0), Inches(10.0)])
    return slide


def make_slide_dl_results(prs):
    """딥러닝 모형 성능 비교 + 최종 선택"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 모형 추정", "딥러닝 모형 성능 비교 및 최종 선택")

    add_content_section(slide, Inches(1.2),
        subtitle="전체 모형 성능 비교",
        sub_subtitle="Test set: 105,395 ODs / 394,498 rows",
        body_lines=[])

    data = [
        ["Model", "Test ρ²", "Top-1", "Top-3", "RMSE", "파라미터 수", "MNL 대비 ρ² 향상"],
        ["MNL (K3)", "0.5304", "72.50%", "96.48%", "0.2451", "9", "— (baseline)"],
        ["TasteNet", "0.5424", "73.86%", "96.94%", "0.2417", "459", "+2.3%p"],
        ["ResLogit", "0.5544", "73.53%", "96.85%", "0.2355", "428", "+4.5%p"],
        ["L-MNL", "0.5553", "74.22%", "97.07%", "0.2347", "440", "+4.7%p"],
        ["DNN", "0.5578", "75.14%", "97.29%", "0.2335", "2,881", "+5.2%p"],
        ["ASU-DNN ★", "0.6856", "75.99%", "97.52%", "0.2243", "550", "+29.3%p"],
    ]
    add_table(slide, data, Inches(0.3), Inches(2.5), Inches(12.5),
              row_height=Inches(0.38),
              col_widths=[Inches(1.5), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.5), Inches(4.3)],
              header_color=COLOR_BLUE)

    # Key findings
    add_content_section(slide, Inches(5.3),
        subtitle="핵심 발견",
        body_lines=[
            [("• ASU-DNN이 모든 지표에서 최고 성능: ρ² ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("0.686", FONT_MEDIUM, Pt(14), COLOR_RED, True),
             (" (MNL 대비 +29.3%p 향상)", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
            "• 대안별 독립 신경망 → 대중교통의 대안 특이적 효과(노선 고유 특성) 포착에 유리",
            [("• 해석 가능 모형(TasteNet, ResLogit, L-MNL)도 MNL 대비 ", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             ("2~5%p 향상", FONT_MEDIUM, Pt(14), COLOR_BLUE, True),
             (" → 비선형 효용의 존재 확인", FONT_REGULAR, Pt(14), COLOR_BODY, False)],
        ])
    return slide


def make_slide_final_model_selection(prs):
    """최종 모형 선택 및 차별점 종합"""
    IMG_DIR = 'C:/Research/6.route_choice_simulation/data/training_set'
    IMG_HEATMAP = 'C:/Research/6.route_choice_simulation/images/fig3_2d_grid_heatmap.png'

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 5장 결론", "최종 모형 선택 및 연구 차별점")

    add_content_section(slide, Inches(1.2),
        subtitle="최종 선택: ASU-DNN (Alternative-Specific Utility DNN)",
        body_lines=[
            [("• 예측 성능: ρ² = 0.686, Top-1 = 76.0%", FONT_REGULAR, Pt(14), COLOR_BODY, False),
             (" — 6개 모형 중 1위", FONT_MEDIUM, Pt(14), COLOR_RED, True)],
            "• 대안별 독립 신경망으로 노선/수단 조합의 고유 특성을 각각 학습",
            "• 파라미터 550개로 DNN(2,881개)보다 효율적이면서 성능 우수",
        ])

    # Left: 2D Grid Heatmap image
    if os.path.exists(IMG_HEATMAP):
        slide.shapes.add_picture(IMG_HEATMAP, Inches(0.3), Inches(3.3),
                                 Inches(6.0), Inches(2.5))
        tx_cap = slide.shapes.add_textbox(Inches(0.3), Inches(5.9), Inches(6.0), Inches(0.3))
        set_run(tx_cap.text_frame.paragraphs[0].add_run(),
                "유사도 가중치 2-Stage Grid Search 결과 (462 시나리오)",
                FONT_REGULAR, Pt(10), COLOR_TABLE_TEXT)

    # Right: SHAP comparison image
    shap_img = os.path.join(IMG_DIR, 'shap_comparison.png')
    if os.path.exists(shap_img):
        slide.shapes.add_picture(shap_img, Inches(6.6), Inches(3.3),
                                 Inches(6.0), Inches(2.5))
        tx_cap2 = slide.shapes.add_textbox(Inches(6.6), Inches(5.9), Inches(6.0), Inches(0.3))
        set_run(tx_cap2.text_frame.paragraphs[0].add_run(),
                "SHAP Feature Importance 비교 (5개 딥러닝 모형)",
                FONT_REGULAR, Pt(10), COLOR_TABLE_TEXT)

    # Bottom: Contribution summary
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), Inches(6.3),
                                 Inches(12.3), Inches(0.9))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    box.line.color.rgb = RGBColor(0x66, 0xBB, 0x6A)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(), "연구 기여", FONT_MEDIUM, Pt(12), RGBColor(0x2E, 0x7D, 0x32), True)
    add_paragraph(tf,
        "① 3-level 복합유사도 + Grid Search 최적화  ② 스마트카드 3,700만건 기반 대규모 검증  "
        "③ MNL~딥러닝 6개 모형 체계적 비교  ④ ASU-DNN으로 ρ² 0.686 달성 (기존 MNL 대비 +29%)",
        FONT_REGULAR, Pt(11), COLOR_BODY, space_before=Pt(3))
    return slide


def make_slide_dl_differentiation(prs):
    """본 연구와 선행연구 비교 (딥러닝 관점)"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_header_group(slide, "제 4장 모형 추정", "본 연구 vs 선행연구 비교")

    add_content_section(slide, Inches(1.2),
        subtitle="기존 딥러닝 경로선택 연구와의 차이점",
        body_lines=[])

    data = [
        ["구분", "기존 선행연구", "본 연구"],
        ["적용 수단", "도로(자동차) 위주\n또는 단일 수단", "버스+도시철도+GTX\n멀티모달 대중교통"],
        ["데이터 규모", "수천~수만 통행\n(설문 RP/SP 기반)", "3,700만 통행, 53만 OD\n(스마트카드 빅데이터)"],
        ["모형 비교", "단일 모형 제안\n(vs MNL만 비교)", "5개 딥러닝 + MNL 포함\n6개 모형 공정 비교"],
        ["선택확률 생성", "관측 이진 선택\n(chosen = 0/1)", "유사도 기반 연속 확률\n(choice_prob softmax)"],
        ["유사도 방법론", "해당 없음\n(관측 경로 직접 사용)", "3-level 복합유사도\nGrid Search 최적화"],
        ["해석 도구", "β 계수만 보고", "SHAP 기반 변수 중요도\n모형 간 일관성 검증"],
        ["검증 지표", "ρ² 단일 지표", "ρ² + FPR-1/3 + RMSE\n+ 수단분담률 재현"],
    ]
    add_table(slide, data, Inches(0.3), Inches(1.9), Inches(12.5),
              row_height=Inches(0.52),
              col_widths=[Inches(2.0), Inches(4.5), Inches(6.0)],
              header_color=COLOR_BLUE)

    # Highlight
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.3), Inches(6.1),
                                 Inches(12.5), Inches(0.7))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFD, 0xE8, 0xE8)
    box.line.color.rgb = RGBColor(0xF5, 0xA0, 0xA0)
    tf = box.text_frame; tf.word_wrap = True
    set_run(tf.paragraphs[0].add_run(),
        "핵심: 설문조사 없이 스마트카드 빅데이터만으로 유사도 매칭 → 경로선택 모형 추정 → "
        "딥러닝 확장까지 완결된 End-to-End 파이프라인을 최초로 제시",
        FONT_MEDIUM, Pt(12), COLOR_RED, True)
    return slide


# ═══════════════════════════════════════════════════
# Slide manipulation utilities
# ═══════════════════════════════════════════════════

def delete_slide(prs, index):
    """Delete a slide at the given 0-based index."""
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    el = slides[index]
    rId = el.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
    if rId is None:
        rId = el.attrib.get('r:id')
    prs.part.drop_rel(rId)
    xml_slides.remove(el)


def reorder_slides(prs, new_order):
    """Reorder slides by providing a list of current 0-based indices in desired order."""
    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)

    # Remove all
    for el in slides:
        xml_slides.remove(el)

    # Re-add in desired order
    for idx in new_order:
        xml_slides.append(slides[idx])


def update_slide_numbers(prs):
    """Update slide number placeholders."""
    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if hasattr(shape, 'text') and shape.name and 'Slide Number' in shape.name:
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        for run in p.runs:
                            run.text = str(i + 1)


def main():
    src = 'C:/Research/6.route_choice_simulation/pptx/20260324-lab.pptx'
    dst = 'C:/Research/6.route_choice_simulation/pptx/20260326-its.pptx'

    prs = Presentation(src)
    n_original = len(prs.slides)
    print(f"Original slides: {n_original}")

    # Original 25 slides (0-based):
    # 0: 표지
    # 1: 현재 진행상황 (lab용) → DELETE
    # 2: Contents → REBUILD
    # 3: 연구배경 - 경로매칭 목적
    # 4: 선행연구 - 유사도 측정
    # 5: 선행연구 - 차별성
    # 6: 유사도 지표
    # 7: Mode/Route 유사도
    # 8: Sequence 유사도 (GTFS)
    # 9: Sequence 유사도 시각화
    # 10: Composite → choice_prob
    # 11: MNL 모델 정의
    # 12: 3-Level + Seq Gate
    # 13: Seq Gate 우회경로
    # 14: 가중치 최적화
    # 15: Fare 수식 변경 → DELETE (중복)
    # 16: MNL β 계수 (모델비교)
    # 17: SHAP 1
    # 18: SHAP 2
    # 19: SHAP 3
    # 20: 모형 성능 비교
    # 21: 이슈 분석 (빈약) → DELETE (교체)
    # 22: 향후 계획 (빈약) → DELETE (교체)
    # 23: 감사합니다
    # 24: MNL β 상세

    # === Step 1: Add new slides (appended to end) ===
    make_slide_pipeline_overview(prs)       # 25: 전체 파이프라인
    make_slide_smartcard_preprocessing(prs) # 26: 스마트카드 전처리
    make_slide_otp_generation(prs)          # 27: OTP 대안경로
    make_slide_matching_statistics(prs)     # 28: 매칭 결과 통계
    make_slide_issues_enhanced(prs)         # 29: 이슈 분석 (보강)
    make_slide_future_enhanced(prs)         # 30: 향후 계획 (보강)
    make_slide_dl_prior_research(prs)       # 31: DL 선행연구
    make_slide_dl_architecture(prs)         # 32: DL 구조 비교
    make_slide_dl_results(prs)              # 33: DL 성능 비교 + 최종선택
    make_slide_dl_differentiation(prs)      # 34: 본 연구 vs 선행연구
    make_slide_final_model_selection(prs)   # 35: 최종 모형 + 이미지

    print(f"Total slides before reorder: {len(prs.slides)}")

    # === Step 2: Define desired order (using original indices) ===
    # Slides to DELETE: 1 (lab), 15 (중복), 21 (old이슈), 22 (old향후)
    desired_order = [
        0,   # 표지
        2,   # Contents (will rebuild)
        3,   # 연구배경 - 경로매칭 목적
        25,  # NEW: 전체 파이프라인
        26,  # NEW: 스마트카드 전처리
        27,  # NEW: OTP 대안경로
        4,   # 선행연구 - 유사도 측정
        5,   # 선행연구 - 차별성
        # --- 매칭 방법론 블록 ---
        6,   # 유사도 지표
        7,   # Mode/Route 유사도
        8,   # Sequence 유사도 (GTFS)
        9,   # Sequence 유사도 시각화
        10,  # Composite → choice_prob
        12,  # 3-Level + Seq Gate
        13,  # Seq Gate 우회경로
        14,  # 가중치 최적화
        28,  # NEW: 매칭 결과 통계
        # --- 모형 추정 블록 ---
        11,  # MNL 모델 정의
        24,  # MNL β 상세
        31,  # NEW: DL 선행연구
        32,  # NEW: DL 구조 비교
        16,  # MNL β 계수 (모델비교) - MNL vs DL 비교 테이블
        33,  # NEW: DL 성능 비교 + 최종선택
        34,  # NEW: 본 연구 vs 선행연구 비교
        17,  # SHAP 1
        18,  # SHAP 2
        19,  # SHAP 3
        20,  # 모형 성능 비교
        35,  # NEW: 최종 모형 선택 + 이미지
        29,  # NEW: 이슈 분석 (보강)
        30,  # NEW: 향후 계획 (보강)
        23,  # 감사합니다
    ]

    reorder_slides(prs, desired_order)
    print(f"After reorder: {len(prs.slides)} slides")

    # === Step 3: Update Contents slide (now at index 1) ===
    contents_slide = prs.slides[1]
    shapes_to_delete = []
    for shape in contents_slide.shapes:
        if shape.name and 'Slide Number' not in shape.name:
            shapes_to_delete.append(shape)
    for shape in shapes_to_delete:
        sp = shape._element
        sp.getparent().remove(sp)

    tx = contents_slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(5), Inches(1.0))
    set_run(tx.text_frame.paragraphs[0].add_run(), "Contents", FONT_MEDIUM, Pt(36), COLOR_BLUE, True)

    items = [
        ("01", "연구 배경 - 경로매칭 및 경로선택모형 추정"),
        ("02", "데이터 - 파이프라인 / 스마트카드 전처리 / OTP 경로생성"),
        ("03", "선행연구 - 유사도 지표, 스마트카드 매칭"),
        ("04", "매칭 방법론 - 유사도 지표 / 3-Level + Sequence Gate"),
        ("05", "매칭 방법론 - 가중치 최적화 / 매칭 결과 통계"),
        ("06", "모형 추정 - MNL β 계수 추정"),
        ("07", "모형 추정 - 딥러닝 선행연구 / 모형 구조 / 성능 비교"),
        ("08", "SHAP 분석 / 최종 모형 선택"),
        ("09", "이슈 분석 및 향후 계획"),
    ]
    for i, (num, title) in enumerate(items):
        y = Inches(1.8) + Inches(i * 0.52)
        tx_n = contents_slide.shapes.add_textbox(Inches(1.2), y, Inches(0.8), Inches(0.5))
        set_run(tx_n.text_frame.paragraphs[0].add_run(), num, FONT_MEDIUM, Pt(15), COLOR_BLUE, True)
        tx_t = contents_slide.shapes.add_textbox(Inches(2.2), y, Inches(9), Inches(0.5))
        set_run(tx_t.text_frame.paragraphs[0].add_run(), title, FONT_REGULAR, Pt(15), COLOR_BODY)

    # === Step 4: Update slide numbers ===
    update_slide_numbers(prs)

    # === Save ===
    prs.save(dst)
    print(f"\nSaved: {dst}")
    print(f"Total slides: {len(prs.slides)}")

    # Print final slide order
    print("\n=== Final Slide Order ===")
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, 'text') and shape.text and len(shape.text) > 3:
                clean = shape.text.replace('\n', ' ')[:80]
                if 'Slide Number' not in (shape.name or ''):
                    texts.append(clean)
        title = texts[0] if texts else "(no text)"
        try:
            print(f"  {i+1:2d}. {title}")
        except UnicodeEncodeError:
            print(f"  {i+1:2d}. (encoding issue - slide exists)")


if __name__ == "__main__":
    main()
