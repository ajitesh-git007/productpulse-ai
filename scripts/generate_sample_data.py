"""Generate industrial synthetic retail data for ProductPulse AI.

The dataset is synthetic but production-shaped. It is designed for a senior
data engineering bootcamp where learners need enough signal volume to test:

- Bronze ingestion from mixed CSV, JSONL, and PDF files.
- Multi-day analytics from orders, returns, reviews, tickets, and inventory.
- RAG retrieval from meaningful unstructured product, policy, QA, and support
  documents.
- Agent behavior across SQL-only metrics, RAG evidence, and memory.

Run from the project root:
    python3 scripts/generate_sample_data.py
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import textwrap
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PDF_DIR = RAW_DIR / "knowledge_pdfs"

START_DATE = date(2026, 8, 14)
EVENT_DATES = [START_DATE + timedelta(days=offset) for offset in range(7)]
MARKETS = ["US", "IN", "UK", "DE", "FR", "JP", "AU"]
CHANNELS = ["web", "mobile_app", "store", "marketplace", "partner_retail"]
WAREHOUSES = ["WH-EAST", "WH-WEST", "WH-EU", "WH-APAC"]
CUSTOMER_SEGMENTS = [
    "performance_runner",
    "casual_runner",
    "football_parent",
    "club_buyer",
    "gym_training",
    "student",
    "commuter",
    "team_sports_manager",
]


PRODUCTS = [
    ("RUN-ULTRA-01", "Adizero Ultra Sprint", "Running Shoes", "Unisex", "2026-08-01", 159.0, "race fit", "narrow_toe_box", 122),
    ("RUN-COMFORT-02", "Cloudfoam Tempo", "Running Shoes", "Women", "2026-07-20", 92.0, "comfort fit", "warm_upper", 82),
    ("TRN-HOOD-03", "Train Essentials Hoodie", "Training Apparel", "Men", "2026-07-12", 68.0, "regular fit", "color_fade", 76),
    ("SOC-CLEAT-04", "Predator Edge League FG", "Football Boots", "Unisex", "2026-08-05", 119.0, "locked heel", "stud_crack", 108),
    ("BAG-DUFFEL-05", "Tiro League Duffel", "Accessories", "Unisex", "2026-06-29", 45.0, "one size", "strap_slip", 58),
    ("RUN-TRAIL-06", "Terrex Ridge Flow", "Trail Running Shoes", "Men", "2026-08-10", 132.0, "protective trail fit", "mud_grip", 72),
    ("YGA-TIGHT-07", "Yoga Studio 7/8 Tight", "Training Apparel", "Women", "2026-07-28", 74.0, "compression fit", "waistband_roll", 70),
    ("CAP-RUN-08", "AeroReady Running Cap", "Accessories", "Unisex", "2026-08-03", 28.0, "adjustable", "sweat_stain", 64),
    ("RUN-ROAD-09", "Supernova Rise Road", "Running Shoes", "Men", "2026-07-18", 118.0, "neutral road fit", "outsole_squeak", 86),
    ("APP-TEE-10", "Own The Run Tee", "Training Apparel", "Women", "2026-07-02", 36.0, "regular fit", "seam_twist", 74),
    ("BAG-BACK-11", "Urban Utility Backpack", "Accessories", "Unisex", "2026-06-15", 88.0, "one size", "zipper_snag", 52),
    ("SOC-BALL-12", "UCL Training Ball", "Football Equipment", "Unisex", "2026-07-30", 32.0, "size 5", "air_retention", 66),
    ("TRN-SHORT-13", "Designed 4 Training Short", "Training Apparel", "Men", "2026-07-08", 42.0, "athletic fit", "liner_chafe", 62),
    ("RUN-SOCK-14", "Performance Cushion Sock 3 Pack", "Accessories", "Unisex", "2026-07-11", 18.0, "compression cuff", "elastic_wear", 92),
    ("OUT-JKT-15", "Rain.RDY City Jacket", "Outerwear", "Women", "2026-08-08", 148.0, "regular fit", "zipper_leak", 48),
    ("TRN-BRA-16", "PowerImpact Training Bra", "Training Apparel", "Women", "2026-07-24", 54.0, "high support", "band_tight", 60),
    ("KID-SHOE-17", "Duramo Kids Trainer", "Kids Footwear", "Kids", "2026-07-16", 52.0, "kids regular", "velcro_wear", 88),
    ("RUN-SANDAL-18", "Adilette Comfort Slide", "Slides", "Unisex", "2026-06-20", 35.0, "relaxed fit", "strap_rub", 84),
    ("GYM-MAT-19", "Training Grip Mat", "Training Equipment", "Unisex", "2026-07-19", 48.0, "one size", "surface_peel", 46),
    ("SWM-SHORT-20", "3-Stripes Swim Short", "Swim Apparel", "Men", "2026-07-01", 40.0, "regular fit", "chlorine_fade", 50),
    ("OUT-FLEECE-21", "Terrex Grid Fleece", "Outdoor Apparel", "Men", "2026-08-06", 96.0, "layering fit", "pilling", 44),
    ("SOC-GLOVE-22", "Predator Training Glove", "Football Equipment", "Unisex", "2026-07-26", 38.0, "snug palm", "grip_wear", 56),
    ("RUN-VEST-23", "Marathon Hydration Vest", "Running Accessories", "Unisex", "2026-08-04", 110.0, "adjustable", "bottle_bounce", 42),
    ("TRN-SHOE-24", "Dropset Strength Trainer", "Training Shoes", "Men", "2026-07-22", 128.0, "stable lifting fit", "heel_slip", 68),
]


ISSUES = {
    "narrow_toe_box": {
        "negative": [
            ("toe box feels narrow", "My toes went numb after a five mile tempo run, and the forefoot pressure was worse than my usual racing shoes.", 1),
            ("need half size exchange", "I normally wear this size, but this model feels short and tight across the metatarsals.", 2),
            ("wide foot warning missing", "The product page did not make the race fit obvious enough for wide-foot runners.", 2),
        ],
        "positive": [
            ("fast race-day feel", "The rocker feels quick, turnover is smooth, and the outsole grips well during intervals.", 5),
            ("secure lockdown", "The upper holds the foot securely for speed sessions with no heel movement.", 4),
        ],
        "ticket_reason": "return",
        "return_reasons": ["toe_box_narrow", "size_too_small", "exchange_half_size_up"],
    },
    "stud_crack": {
        "negative": [
            ("front stud cracked", "A front stud cracked after two firm-ground matches and the player could not finish practice.", 1),
            ("heel collar rubbing", "The heel collar caused blisters during the first two practices even with football socks.", 2),
            ("warranty review needed", "Support asked for surface photos because the outsole damage happened unusually early.", 2),
        ],
        "positive": [
            ("excellent ball touch", "Touch, traction, and lockdown are strong on firm natural grass.", 5),
            ("stable during sprints", "The boot feels planted when accelerating and cutting.", 4),
        ],
        "ticket_reason": "warranty",
        "return_reasons": ["stud_crack", "heel_blister", "warranty_review"],
    },
    "color_fade": {
        "negative": [
            ("black color faded", "Cold wash still left visible fading around cuffs and seam panels.", 2),
            ("team order color mismatch", "Several hoodies in the team order looked washed out after one care-label wash.", 1),
            ("photo review requested", "Support asked for wash temperature and photos of cuff fading.", 2),
        ],
        "positive": [
            ("soft fleece", "The fleece feels premium and comfortable for travel days.", 5),
            ("good gym layer", "Fit and warmth are exactly right for warm-up sessions.", 4),
        ],
        "ticket_reason": "quality",
        "return_reasons": ["color_fade", "quality_expectation", "team_order_quality"],
    },
    "warm_upper": {
        "negative": [
            ("upper runs warm", "The mesh felt warm during treadmill sessions and afternoon walking.", 3),
            ("less breathable than expected", "Cushioning is comfortable, but airflow is limited in humid weather.", 3),
        ],
        "positive": [
            ("comfortable daily trainer", "Soft cushioning works well for long shifts and easy runs.", 5),
            ("great value", "The comfort level is strong for the price point.", 4),
        ],
        "ticket_reason": "product_advice",
        "return_reasons": ["comfort_preference", "breathability"],
    },
    "mud_grip": {
        "negative": [
            ("mud traction weak", "The outsole packed with wet mud faster than expected on clay trail sections.", 2),
            ("dries slowly", "The upper held water after creek crossings and felt heavy.", 3),
        ],
        "positive": [
            ("rock protection is strong", "Underfoot protection feels stable on rocky climbs.", 5),
            ("secure on dry trail", "Grip is confident on dry dirt and gravel.", 4),
        ],
        "ticket_reason": "product_advice",
        "return_reasons": ["mud_grip", "water_retention"],
    },
    "waistband_roll": {
        "negative": [
            ("waistband rolls", "The waistband rolled during squats and studio transitions.", 2),
            ("compression too high", "Fabric quality is good, but the waist pressure is too high for yoga.", 3),
        ],
        "positive": [
            ("fabric feels premium", "The fabric is smooth, supportive, and squat proof.", 5),
            ("phone pocket works", "The side pocket keeps my phone secure during class.", 4),
        ],
        "ticket_reason": "return",
        "return_reasons": ["waistband_roll", "fit_preference"],
    },
    "generic_quality": {
        "negative": [
            ("quality concern", "The product works, but one detail did not match the expected premium finish.", 3),
            ("needs clearer guidance", "The product page should explain the material or fit more clearly.", 3),
        ],
        "positive": [
            ("meets expectation", "The product performs as expected and arrived in good condition.", 4),
            ("would buy again", "Quality, fit, and delivery were consistent with the product description.", 5),
        ],
        "ticket_reason": "product_advice",
        "return_reasons": ["changed_mind", "fit_preference"],
    },
}


THEME_ALIAS = {
    "strap_slip": "generic_quality",
    "sweat_stain": "generic_quality",
    "outsole_squeak": "generic_quality",
    "seam_twist": "generic_quality",
    "zipper_snag": "generic_quality",
    "air_retention": "generic_quality",
    "liner_chafe": "generic_quality",
    "elastic_wear": "generic_quality",
    "zipper_leak": "generic_quality",
    "band_tight": "generic_quality",
    "velcro_wear": "generic_quality",
    "strap_rub": "generic_quality",
    "surface_peel": "generic_quality",
    "chlorine_fade": "generic_quality",
    "pilling": "generic_quality",
    "grip_wear": "generic_quality",
    "bottle_bounce": "generic_quality",
    "heel_slip": "generic_quality",
}


PDF_DOCUMENTS = [
    {
        "doc_id": "PDF-RETURNS-OPERATIONS-2026",
        "source_type": "policy_pdf",
        "product_id": "",
        "title": "Retail Returns Operations Playbook - Launch Season 2026",
        "effective_date": "2026-08-01",
        "file_name": "retail_returns_operations_playbook_2026.pdf",
        "sections": [
            ("Operating context", "Launch season return handling focuses on preserving customer trust while separating preventable content issues from true product quality issues. The support team should prioritize exchange when a fit adjustment can resolve the issue and inventory is available."),
            ("Footwear fit returns", "For close-fit running shoes, agents should capture usual size, purchased size, foot width if volunteered, distance used, surface, sock thickness, and exact discomfort language. Narrow toe box, numb toes, and forefoot pressure are treated as fit evidence, not customer fault."),
            ("Exchange decision", "A half-size exchange is recommended when the product is lightly used, the customer reports fit pressure, and stock exists in the replacement size. Return shipping can be waived when fit guidance was not visible at size selection."),
            ("Warranty boundary", "Warranty review is appropriate for cracked components, outsole separation, abnormal colorfastness failure, or early material breakdown under approved use. Photos and use context are required before final decision."),
            ("Operations measurement", "Operations should monitor exchange rate, refund rate, return reason mix, and inventory ability to satisfy exchange recommendations. A successful intervention reduces refund share while maintaining customer satisfaction."),
        ],
    },
    {
        "doc_id": "PDF-RUN-ULTRA-LAUNCH-BRIEF",
        "source_type": "product_brief_pdf",
        "product_id": "RUN-ULTRA-01",
        "title": "RUN-ULTRA-01 Launch Brief and Fit Calibration",
        "effective_date": "2026-08-01",
        "file_name": "run_ultra_01_launch_fit_brief.pdf",
        "sections": [
            ("Product intent", "RUN-ULTRA-01 is positioned as a tempo and race-day shoe with close forefoot lockdown, aggressive rocker geometry, and a grippy rubber outsole for interval sessions."),
            ("Fit calibration", "The intended fit is secure, but it should not cause numbness. Wide-foot runners and customers between sizes should be guided toward a half-size increase. Fit content should appear near size selection rather than only in long description text."),
            ("Early warning language", "Launch monitoring should watch for narrow toe box, numb toes, forefoot pressure, size too small, black toenail, laces loosen, and half-size exchange. These phrases usually indicate either content clarity risk or size recommendation mismatch."),
            ("Support handling", "Agents should ask whether discomfort occurs immediately or after mileage, whether the customer usually wears wide-fit shoes, and whether they want exchange or refund. Lacing advice may help lockdown, but it should not be used to dismiss pain."),
            ("Commercial risk", "A preventable sizing issue can reduce rating momentum and increase paid-media inefficiency. Exchange success and rating recovery should be reviewed daily during launch week."),
        ],
    },
    {
        "doc_id": "PDF-SOC-CLEAT-QA-2026",
        "source_type": "qa_pdf",
        "product_id": "SOC-CLEAT-04",
        "title": "SOC-CLEAT-04 Firm-Ground QA and Warranty Field Guide",
        "effective_date": "2026-08-05",
        "file_name": "soc_cleat_04_fg_qa_warranty_guide.pdf",
        "sections": [
            ("Approved use", "SOC-CLEAT-04 is designed for firm natural grass. Artificial turf, concrete walkways, gravel, and indoor courts can overstress studs and should be documented during warranty triage."),
            ("Stud crack triage", "A cracked stud within two matches on approved firm ground is high-confidence warranty evidence. Agents need photos of the cracked stud, the full outsole, field surface, purchase date, and number of matches."),
            ("Heel rubbing investigation", "Heel rubbing can result from break-in, sock thickness, lacing, heel shape, or collar geometry. Repeated heel-collar language across tickets should be escalated even when individual cases may be fit-related."),
            ("Club buyer handling", "Team and club orders have higher service impact because one equipment issue can affect several players before a weekend match. Replacement options should be checked before promising a resolution."),
            ("QA package", "The QA package should include ticket ids, return ids, photos if available, market, channel, size range, surface type, and repeated customer phrases."),
        ],
    },
    {
        "doc_id": "PDF-CONTENT-OPS-2026",
        "source_type": "sop_pdf",
        "product_id": "",
        "title": "Product Content Operations SOP - Feedback Driven Updates",
        "effective_date": "2026-08-01",
        "file_name": "product_content_operations_feedback_sop.pdf",
        "sections": [
            ("Monitoring cadence", "During launch week, content operations reviews top complaint phrases twice daily. The review compares customer reviews, support tickets, returns, and current inventory to decide whether product-page changes are needed."),
            ("Fit copy standards", "Fit guidance should be short, visible, and placed near size selection. For close-fit footwear, approved wording includes close race fit through the forefoot and consider half size up for wide feet or between sizes."),
            ("Warranty copy standards", "Surface and use restrictions must be direct. Firm-ground football boots should clearly state approved surfaces. Apparel care guidance should be visible before customers reach support."),
            ("Change governance", "Product managers approve product-specific fit wording. Support operations approves agent guidance. Marketplace content must be synchronized after owned-site updates so customers see consistent guidance."),
            ("Post-change measurement", "After a content update, measure return contacts, exchange rate, average rating, conversion, and support handle time for seven days."),
        ],
    },
    {
        "doc_id": "PDF-APPAREL-COLORFASTNESS-2026",
        "source_type": "quality_pdf",
        "product_id": "TRN-HOOD-03",
        "title": "Training Apparel Colorfastness Quality Review",
        "effective_date": "2026-07-25",
        "file_name": "training_apparel_colorfastness_quality_review.pdf",
        "sections": [
            ("Scope", "This review covers dark-color fleece and training apparel complaints involving cuff fade, seam fade, wash-panel discoloration, and color mismatch across team orders."),
            ("Evidence quality", "High-confidence evidence includes verified purchase, cold-wash confirmation, no tumble dry, photo evidence, and repeated complaints for the same product and colorway."),
            ("Agent questions", "Agents should ask for wash temperature, detergent type, dryer use, photo evidence, number of affected units, and whether the care label was followed."),
            ("Replacement rules", "Replacement or refund is appropriate when abnormal fading appears after care-label instructions were followed. Team-order cases should be handled with priority because inconsistent appearance affects group buyers."),
            ("QA escalation", "Three verified colorfastness complaints in seven days for the same product and colorway should trigger supplier batch and dye-lot review."),
        ],
    },
    {
        "doc_id": "PDF-INVENTORY-EXCHANGE-2026",
        "source_type": "operations_pdf",
        "product_id": "",
        "title": "Inventory-Aware Exchange Operations Guide",
        "effective_date": "2026-08-01",
        "file_name": "inventory_aware_exchange_operations_guide.pdf",
        "sections": [
            ("Exchange principle", "An exchange recommendation is only useful when inventory can support it. Support agents should check available size, warehouse proximity, and replacement lead time before recommending an exchange."),
            ("Stock status interpretation", "Healthy stock supports exchange-first handling. Watch status requires agents to avoid overpromising replacement dates. Low stock may require refund, alternative product recommendation, or waitlist guidance."),
            ("Launch-week allocation", "High-demand launch products should reserve a small percentage of inventory for service recovery when sizing or warranty issues emerge early."),
            ("Operational risk", "If a product has high exchange demand and low replacement stock, category teams should decide whether to update content, slow promotion, or recommend adjacent models."),
            ("Reporting", "Daily reporting should combine return reason, ticket count, on-hand inventory, reserved units, and exchange conversion by market."),
        ],
    },
    {
        "doc_id": "PDF-VOC-LAUNCH-DIGEST-2026",
        "source_type": "voc_pdf",
        "product_id": "",
        "title": "Voice of Customer Launch Digest - Week 33",
        "effective_date": "2026-08-20",
        "file_name": "voice_of_customer_launch_digest_week_33.pdf",
        "sections": [
            ("Executive view", "The strongest launch-week signals are sizing pressure on RUN-ULTRA-01, stud durability and heel comfort on SOC-CLEAT-04, and colorfastness complaints on TRN-HOOD-03."),
            ("Running shoes", "RUN-ULTRA-01 receives positive performance language around rocker, speed, and grip, but negative feedback clusters around narrow forefoot fit and half-size exchange requests."),
            ("Football boots", "SOC-CLEAT-04 receives positive traction and touch feedback, while negative contacts mention front stud cracks, heel collar rubbing, and surface eligibility questions."),
            ("Apparel", "TRN-HOOD-03 negative contacts are mostly about dark color fade after cold wash. Positive comments still mention fleece comfort and team-order fit consistency."),
            ("Actions", "Recommended actions are size copy update for RUN-ULTRA-01, QA evidence collection for SOC-CLEAT-04, and colorway-level review for TRN-HOOD-03."),
        ],
    },
    {
        "doc_id": "PDF-MARKETPLACE-CONTENT-2026",
        "source_type": "marketplace_pdf",
        "product_id": "",
        "title": "Marketplace Product Content Alignment Guide",
        "effective_date": "2026-08-01",
        "file_name": "marketplace_product_content_alignment_guide.pdf",
        "sections": [
            ("Problem", "Owned-site content changes do not help marketplace customers unless copy is synchronized across partner listings. Mismatch can continue to create avoidable returns after the owned site is corrected."),
            ("Fit content", "Marketplace titles and bullets should include the same fit warning used on owned pages when a product has close fit, compression fit, or surface restrictions."),
            ("Policy content", "Return and warranty instructions should avoid promising outcomes that differ from support policy. Surface restrictions and evidence requirements should be clear for football boots and outdoor gear."),
            ("Monitoring", "Marketplace reviews should be tagged separately because customers may not have seen owned-site fit guidance. Compare marketplace return rate against web and mobile app return rate."),
            ("Escalation", "If marketplace return share exceeds owned-channel return share by more than four points, content operations should audit listing copy and image order."),
        ],
    },
    {
        "doc_id": "PDF-SUPPORT-MACROS-2026",
        "source_type": "support_pdf",
        "product_id": "",
        "title": "Support Macro Library - Product Issue Intake",
        "effective_date": "2026-08-01",
        "file_name": "support_macro_library_product_issue_intake.pdf",
        "sections": [
            ("Fit intake", "For footwear fit contacts, ask for usual size, purchased size, foot width if volunteered, activity type, distance used, sock type, and where pressure occurs."),
            ("Warranty intake", "For cracked, broken, or separated components, ask for photos, purchase date, number of uses, playing surface or activity environment, and whether the product was used as intended."),
            ("Apparel quality intake", "For color fade, seam twist, pilling, or fabric defects, ask for wash method, care-label adherence, photos, and number of affected items."),
            ("Tone standards", "Acknowledge the customer impact first. Avoid blaming the customer for size or use conditions. Explain the next evidence needed and expected resolution path."),
            ("Routing", "Route fit guidance to support operations, repeated product copy issues to content operations, warranty clusters to QA, and stock-constrained exchanges to inventory operations."),
        ],
    },
    {
        "doc_id": "PDF-QUALITY-ESCALATION-2026",
        "source_type": "quality_pdf",
        "product_id": "",
        "title": "Product Quality Escalation Thresholds",
        "effective_date": "2026-08-01",
        "file_name": "product_quality_escalation_thresholds.pdf",
        "sections": [
            ("Threshold logic", "Escalation thresholds are based on repeated evidence across source types, severity, and commercial impact. One isolated complaint is monitored. Repeated complaints with matching product id and issue theme are escalated."),
            ("High priority", "High priority includes cracked studs under approved use, injury language, team-order quality failures, or abnormal colorfastness with photo evidence across multiple customers."),
            ("Medium priority", "Medium priority includes repeated fit discomfort, product-page mismatch, unclear surface guidance, or rising return rate without clear defect evidence."),
            ("Evidence package", "Escalations should include product id, issue theme, market, channel, ticket ids, review ids, return ids, inventory status, and recommended owner."),
            ("Closure", "An escalation is closed only after owner response, action taken, measurement window, and residual risk are documented."),
        ],
    },
]


def product_rows() -> list[dict[str, object]]:
    """Return product master rows in the exact CSV schema expected by Bronze.

    The tuple constants above are convenient for generation. This function
    converts them into dictionaries so the CSV header is explicit and stable.
    """

    return [
        {
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
            "gender": gender,
            "launch_date": launch_date,
            "price_usd": price_usd,
            "fit_profile": fit_profile,
            "known_risk_theme": issue_theme.replace("_", " "),
        }
        for product_id, product_name, category, gender, launch_date, price_usd, fit_profile, issue_theme, _base in PRODUCTS
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to a CSV file with a deterministic header.

    The Databricks ingestion notebook reads this file with `header=True`.
    Keeping the field order stable makes schema inference and classroom
    inspection easier.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one JSON object per line.

    JSONL is a realistic landing format for event feeds because each record can
    be appended independently. Spark can read the full folder efficiently.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_partitioned_jsonl(rows: list[dict[str, object]], partition_column: str, base_path: Path, file_name: str) -> None:
    """Write rows into Hive-style date folders.

    Example output:

        data/raw/orders/event_date=2026-08-14/orders.jsonl

    This mirrors how many production raw zones organize daily event feeds and
    lets the Databricks notebook read all dates with one wildcard path.
    """

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[partition_column]), []).append(row)
    for partition_value, partition_rows in sorted(grouped.items()):
        write_jsonl(base_path / f"{partition_column}={partition_value}" / file_name, partition_rows)


def issue_profile(product: tuple) -> dict[str, object]:
    """Return the complaint/positive-language profile for one product.

    A few hero products have named risk themes such as `narrow_toe_box` or
    `stud_crack`. The remaining catalog uses a generic quality profile so the
    dataset has both strong signals and normal background noise.
    """

    theme = product[7]
    return ISSUES.get(theme) or ISSUES[THEME_ALIAS.get(theme, "generic_quality")]


def pdf_escape(text: str) -> str:
    """Escape text before writing it into a raw PDF content stream."""

    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def expand_pdf_sections(document: dict[str, object]) -> list[tuple[str, str]]:
    """Create detailed, artifact-like document sections without copilot meta text."""

    base_sections = list(document["sections"])
    title = str(document["title"])
    operational_appendix = [
        (
            "Field examples",
            f"For {title}, field teams should separate isolated preference language from repeated issue evidence. Example A: a verified customer reports the issue after one or two normal uses, provides order context, and uses language that matches other recent contacts. Treat this as meaningful early evidence. Example B: a customer reports preference language without operational detail. Treat this as low-confidence but useful for trend monitoring. Example C: several customers in the same market report the same product id and issue phrase within three days. Treat this as a pattern requiring owner review.",
        ),
        (
            "Required intake fields",
            f"Required fields for {title} workflows are product id, order id when available, market, channel, customer segment, issue phrase, product condition, preferred outcome, and evidence attachments when relevant. Footwear also needs activity type, distance or match count, surface, sock or lace context, and usual size. Apparel also needs wash method, care-label status, and photo evidence. Accessories and equipment need number of uses and environment.",
        ),
        (
            "Decision matrix",
            f"The decision matrix for {title} prioritizes commercial recovery and accurate customer outcome. Use exchange when the product likely fits the customer after size or fit adjustment and inventory can support the exchange. Use refund when replacement is not practical, the customer no longer wants the product, or policy conditions require refund. Use warranty review when there is possible material or component failure under approved use. Use content update when repeated evidence suggests the product page did not set accurate expectations.",
        ),
        (
            "Owner routing",
            f"Owner routing for {title} depends on the issue class and customer impact. Support operations owns customer handling and macro updates. Product content owns product-page language. Category management owns commercial impact and promotion decisions. Product QA owns physical product investigation. Inventory operations owns stock recovery, warehouse allocation, and exchange feasibility. Marketplace operations owns partner listing alignment.",
        ),
        (
            "Measurement window",
            f"After an action related to {title} is taken, monitor seven-day movement in ticket count, return count, exchange rate, refund rate, average rating, negative sentiment share, and inventory availability. If contacts decline but return rate remains high, the issue may be deeper than content clarity. If exchange rate improves, the intervention may be preserving demand while reducing refund impact.",
        ),
        (
            "Market variation",
            f"Market review for {title} should compare language and outcome by US, IN, UK, DE, FR, JP, and AU. A sizing issue concentrated in one market may reflect local size conversion or marketplace copy. A warranty issue appearing across markets is more likely a product or usage-pattern concern. Channel splits matter because partner listings may omit fit or care guidance that is present on owned digital channels.",
        ),
        (
            "Commercial impact readout",
            f"The commercial readout for {title} should combine units sold, order count, refund amount, exchange count, current stock position, and negative sentiment share. High order volume with moderate complaint rate can still represent a large service workload. Low order volume with severe complaints may require quality review because early adopters often influence launch reputation.",
        ),
        (
            "Support manager review",
            f"Support managers reviewing {title} should check whether agents are collecting enough context before choosing outcome. Missing surface details, missing wash details, or missing fit history can make a valid customer issue look ambiguous. Managers should coach agents to document the customer language exactly because repeated phrasing is often the fastest path to root-cause detection.",
        ),
        (
            "Product team review",
            f"Product teams reviewing {title} should compare intended design against customer expectation. If the product behaves as designed but customers are surprised, content guidance may be the fix. If the product fails under intended use, QA review is needed. If customers buy the product for an unintended use case, category and content teams should decide whether to redirect demand or change messaging.",
        ),
        (
            "Operational scenario",
            f"Scenario for {title}: a spike appears on day three, with rising ticket count, repeated issue phrases, and a return rate moving faster than order growth. The first response is not a broad recall or blanket refund. The first response is evidence consolidation, owner routing, inventory check, and a narrowly scoped customer-handling recommendation.",
        ),
    ]
    return base_sections + operational_appendix * 4


def wrap_pdf_pages(title: str, sections: list[tuple[str, str]]) -> list[list[str]]:
    """Convert PDF sections into fixed-size page line groups.

    The simple PDF writer below does not use ReportLab or external libraries.
    It needs already-wrapped lines so each page fits inside a standard letter
    page. This keeps the data generator dependency-free.
    """

    lines = [title, ""]
    for section_title, body in sections:
        lines.append(section_title.upper())
        for paragraph in body.split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=92))
        lines.append("")

    pages: list[list[str]] = []
    for start in range(0, len(lines), 34):
        pages.append(lines[start : start + 34])
    return pages


def write_simple_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> int:
    """Write a readable PDF using only the Python standard library."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pages = wrap_pdf_pages(title, sections)

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_object_numbers = []
    content_object_numbers = []
    next_object_number = 3
    for _ in pages:
        page_object_numbers.append(next_object_number)
        content_object_numbers.append(next_object_number + 1)
        next_object_number += 2

    font_object_number = next_object_number
    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1"))

    for page_number, lines in enumerate(pages, start=1):
        page_obj = page_object_numbers[page_number - 1]
        content_obj = content_object_numbers[page_number - 1]
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        )
        objects.append(page.encode("latin-1"))

        stream_lines = ["BT", "/F1 10 Tf", "46 748 Td", "14 TL"]
        for index, line in enumerate(lines):
            prefix = "" if index == 0 else "T* "
            stream_lines.append(f"{prefix}({pdf_escape(line)}) Tj")
        stream_lines.append(f"T* (Page {page_number} of {len(pages)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        objects.append(b"<< /Length " + str(len(stream)).encode("latin-1") + b" >>\nstream\n" + stream + b"\nendstream")

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_number} 0 obj\n".encode("latin-1"))
        content.extend(obj)
        content.extend(b"\nendobj\n")

    xref_start = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    content.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_start}\n"
            "%%EOF\n"
        ).encode("latin-1")
    )
    path.write_bytes(bytes(content))
    return len(pages)


def build_orders() -> list[dict[str, object]]:
    """Create synthetic order-line facts across seven business days.

    The volume is based on product-level demand from the `PRODUCTS` constants.
    Weekend and launch-ramp factors create realistic movement over time instead
    of perfectly flat daily counts.
    """

    random.seed(7001)
    rows: list[dict[str, object]] = []
    sequence = 1
    for day_index, event_date in enumerate(EVENT_DATES):
        weekend_boost = 1.12 if event_date.weekday() in {5, 6} else 1.0
        launch_ramp = 1.0 + (day_index * 0.025)
        for product in PRODUCTS:
            product_id, _name, _category, _gender, _launch, price, _fit, _theme, base_demand = product
            order_count = int((base_demand + random.randint(-14, 18)) * weekend_boost * launch_ramp)
            for _ in range(max(order_count, 12)):
                quantity = random.choices([1, 2, 3], weights=[82, 15, 3])[0]
                rows.append(
                    {
                        "order_id": f"ORD-{event_date.strftime('%Y%m%d')}-{sequence:06d}",
                        "event_date": event_date.isoformat(),
                        "product_id": product_id,
                        "quantity": quantity,
                        "unit_price_usd": price,
                        "market": random.choices(MARKETS, weights=[32, 18, 14, 10, 9, 8, 9])[0],
                        "channel": random.choices(CHANNELS, weights=[38, 31, 14, 12, 5])[0],
                        "customer_segment": random.choice(CUSTOMER_SEGMENTS),
                    }
                )
                sequence += 1
    return rows


def build_reviews(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create reviews connected to order volume and product risk themes.

    Review volume is proportional to order count. Products with known launch
    risks intentionally receive a higher negative share so RAG questions have
    meaningful evidence patterns to retrieve.
    """

    random.seed(7002)
    rows: list[dict[str, object]] = []
    sequence = 1
    by_date_product: dict[tuple[str, str], int] = {}
    for order in orders:
        key = (order["event_date"], order["product_id"])
        by_date_product[key] = by_date_product.get(key, 0) + 1

    products_by_id = {product[0]: product for product in PRODUCTS}
    for event_date in EVENT_DATES:
        for product_id, order_count in sorted((key[1], value) for key, value in by_date_product.items() if key[0] == event_date.isoformat()):
            product = products_by_id[product_id]
            profile = issue_profile(product)
            review_count = max(8, int(order_count * random.uniform(0.14, 0.23)))
            risk_bias = 0.62 if product[7] in {"narrow_toe_box", "stud_crack", "color_fade"} else 0.34
            for _ in range(review_count):
                sentiment_group = "negative" if random.random() < risk_bias else "positive"
                title, body, rating = random.choice(profile[sentiment_group])
                sentiment = "negative" if rating <= 2 else "positive" if rating >= 4 else "neutral"
                rows.append(
                    {
                        "review_id": f"REV-{event_date.strftime('%Y%m%d')}-{sequence:06d}",
                        "event_date": event_date.isoformat(),
                        "product_id": product_id,
                        "rating": rating,
                        "review_title": title,
                        "review_body": body,
                        "sentiment": sentiment,
                        "issue_tag": "positive_experience" if rating >= 4 else product[7],
                        "channel": random.choice(CHANNELS),
                        "locale": "en-US",
                        "market": random.choice(MARKETS),
                        "verified_purchase": random.choices([True, False], weights=[91, 9])[0],
                    }
                )
                sequence += 1
    return rows


def build_tickets(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create support tickets tied to real orders.

    Tickets are more operational than reviews: they include reason, priority,
    status, channel, and customer segment. This makes them valuable for both RAG
    evidence and support workload metrics.
    """

    random.seed(7003)
    rows: list[dict[str, object]] = []
    sequence = 1
    orders_by_date_product: dict[tuple[str, str], list[dict[str, object]]] = {}
    for order in orders:
        orders_by_date_product.setdefault((order["event_date"], order["product_id"]), []).append(order)

    products_by_id = {product[0]: product for product in PRODUCTS}
    for event_date in EVENT_DATES:
        for product_id, product_orders in sorted((key[1], value) for key, value in orders_by_date_product.items() if key[0] == event_date.isoformat()):
            product = products_by_id[product_id]
            profile = issue_profile(product)
            ticket_rate = 0.105 if product[7] in {"narrow_toe_box", "stud_crack", "color_fade"} else 0.045
            ticket_count = max(3, int(len(product_orders) * random.uniform(ticket_rate * 0.75, ticket_rate * 1.25)))
            for _ in range(ticket_count):
                title, body, _rating = random.choice(profile["negative"])
                order = random.choice(product_orders)
                priority = "high" if product[7] in {"stud_crack", "color_fade"} and random.random() < 0.45 else random.choices(["low", "medium", "high"], weights=[25, 55, 20])[0]
                rows.append(
                    {
                        "ticket_id": f"TCK-{event_date.strftime('%Y%m%d')}-{sequence:06d}",
                        "event_date": event_date.isoformat(),
                        "order_id": order["order_id"],
                        "product_id": product_id,
                        "reason": profile["ticket_reason"],
                        "ticket_body": f"{title}. {body} Customer wants {random.choice(['exchange', 'refund', 'warranty review', 'fit advice'])}.",
                        "status": random.choices(["open", "pending_customer", "closed"], weights=[24, 28, 48])[0],
                        "priority": priority,
                        "channel": random.choices(["support_chat", "email", "phone"], weights=[55, 35, 10])[0],
                        "customer_segment": order["customer_segment"],
                    }
                )
                sequence += 1
    return rows


def build_returns(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create return events from the order population.

    Returns are intentionally not random noise. Products with stronger issue
    themes have higher return probability and issue-specific return reasons.
    That gives the Gold tables realistic analytical signal.
    """

    random.seed(7004)
    rows: list[dict[str, object]] = []
    sequence = 1
    products_by_id = {product[0]: product for product in PRODUCTS}
    for order in orders:
        product = products_by_id[order["product_id"]]
        profile = issue_profile(product)
        base_rate = 0.155 if product[7] in {"narrow_toe_box", "stud_crack", "color_fade"} else 0.055
        if random.random() < base_rate:
            reason = random.choice(profile["return_reasons"])
            rows.append(
                {
                    "return_id": f"RET-{str(order['event_date']).replace('-', '')}-{sequence:06d}",
                    "event_date": order["event_date"],
                    "order_id": order["order_id"],
                    "product_id": order["product_id"],
                    "return_reason": reason,
                    "refund_amount_usd": round(float(order["unit_price_usd"]) * int(order["quantity"]), 2),
                    "is_exchange": reason in {"exchange_half_size_up", "size_too_small", "fit_preference", "waistband_roll"},
                    "condition": random.choices(["new", "light_use", "photo_review_required", "used"], weights=[42, 34, 18, 6])[0],
                }
            )
            sequence += 1
    return rows


def build_inventory(orders: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create daily warehouse-level inventory snapshots.

    Inventory starts from product demand assumptions, decreases with sold units,
    and receives small daily replenishments. This lets the agent explain whether
    an exchange recommendation is operationally feasible.
    """

    random.seed(7005)
    rows: list[dict[str, object]] = []
    product_daily_units: dict[tuple[str, str], int] = {}
    for order in orders:
        key = (order["event_date"], order["product_id"])
        product_daily_units[key] = product_daily_units.get(key, 0) + int(order["quantity"])

    stock_position = {product[0]: product[8] * 9 for product in PRODUCTS}
    for event_date in EVENT_DATES:
        for product in PRODUCTS:
            product_id = product[0]
            sold = product_daily_units.get((event_date.isoformat(), product_id), 0)
            stock_position[product_id] = max(0, stock_position[product_id] - sold + random.randint(8, 32))
            remaining = stock_position[product_id]
            for warehouse in WAREHOUSES:
                warehouse_share = random.uniform(0.18, 0.32)
                on_hand = int(remaining * warehouse_share)
                reserved = random.randint(3, max(8, int(on_hand * 0.18)))
                reorder_point = 55 if product[8] > 80 else 32
                status = "low" if on_hand < reorder_point else "watch" if on_hand < reorder_point * 1.7 else "healthy"
                rows.append(
                    {
                        "snapshot_date": event_date.isoformat(),
                        "product_id": product_id,
                        "warehouse_id": warehouse,
                        "on_hand_units": on_hand,
                        "reserved_units": reserved,
                        "reorder_point": reorder_point,
                        "stock_status": status,
                    }
                )
    return rows


def build_pdf_manifest() -> list[dict[str, object]]:
    """Write PDF files and return the manifest rows that describe them.

    The PDF files provide unstructured enterprise knowledge for RAG. The
    manifest provides structured metadata such as doc_id, source_type,
    product_id, title, effective_date, and page_count.
    """

    rows: list[dict[str, object]] = []
    for document in PDF_DOCUMENTS:
        page_count = write_simple_pdf(
            PDF_DIR / str(document["file_name"]),
            str(document["title"]),
            expand_pdf_sections(document),
        )
        rows.append(
            {
                "doc_id": document["doc_id"],
                "source_type": document["source_type"],
                "product_id": document["product_id"],
                "title": document["title"],
                "effective_date": document["effective_date"],
                "file_name": document["file_name"],
                "page_count": page_count,
            }
        )
    return rows


def main() -> None:
    """Regenerate the complete raw data landing zone.

    This function deletes the previous synthetic raw folder and rebuilds it
    from scratch. That makes the demo deterministic: every instructor and
    learner sees the same row counts, issue patterns, and PDF corpus.
    """

    if RAW_DIR.exists():
        # The generator owns `data/raw`; removing it avoids stale files from a
        # previous run being accidentally loaded by the wildcard ingestion paths.
        shutil.rmtree(RAW_DIR)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    products = product_rows()
    write_csv(RAW_DIR / "products.csv", products)

    orders = build_orders()
    returns = build_returns(orders)
    reviews = build_reviews(orders)
    tickets = build_tickets(orders)
    inventory = build_inventory(orders)
    pdf_manifest = build_pdf_manifest()

    write_partitioned_jsonl(orders, "event_date", RAW_DIR / "orders", "orders.jsonl")
    write_partitioned_jsonl(returns, "event_date", RAW_DIR / "returns", "returns.jsonl")
    write_partitioned_jsonl(reviews, "event_date", RAW_DIR / "reviews", "reviews.jsonl")
    write_partitioned_jsonl(tickets, "event_date", RAW_DIR / "support_tickets", "tickets.jsonl")
    write_partitioned_jsonl(inventory, "snapshot_date", RAW_DIR / "inventory_snapshots", "inventory_snapshots.jsonl")
    write_csv(RAW_DIR / "knowledge_pdf_manifest.csv", pdf_manifest)

    print(f"Generated products: {len(products)}")
    print(f"Generated order lines: {len(orders)}")
    print(f"Generated return events: {len(returns)}")
    print(f"Generated reviews: {len(reviews)}")
    print(f"Generated support tickets: {len(tickets)}")
    print(f"Generated inventory snapshots: {len(inventory)}")
    print(f"Generated PDF documents: {len(pdf_manifest)}")
    print(f"Generated PDF pages: {sum(int(row['page_count']) for row in pdf_manifest)}")


if __name__ == "__main__":
    main()
