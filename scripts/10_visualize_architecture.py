import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Resolve paths
project_root = Path(__file__).resolve().parent.parent
fig_dir = project_root / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
ax.set_xlim(0, 12)
ax.set_ylim(-0.5, 6.5)
ax.axis('off')

box_w = 3.2
box_h = 2.8
y_center = 3
y_bottom = y_center - box_h/2

# Colors
BOX_FACE = "#eff9ff"
BOX_EDGE = "#003859"
TITLE_COL = "#184683"
TEXT_COL = "#003859"

def draw_box(x, title, subtitle, details):
    rect = patches.FancyBboxPatch((x, y_bottom), box_w, box_h, boxstyle="round,pad=0.1,rounding_size=0.15", 
                                  linewidth=2.5, edgecolor=BOX_EDGE, facecolor=BOX_FACE)
    ax.add_patch(rect)
    ax.text(x + box_w/2, y_bottom + box_h - 0.4, title, ha='center', va='center', 
            fontsize=14, fontweight='bold', color=TITLE_COL)
    ax.text(x + box_w/2, y_bottom + box_h - 0.9, subtitle, ha='center', va='center', 
            fontsize=15, fontstyle='italic', color='#ff4444', fontweight='bold')
    ax.text(x + box_w/2, y_bottom + box_h - 1.7, details, ha='center', va='center', 
            fontsize=11.5, color=TEXT_COL, multialignment='center', linespacing=1.6)

# Draw the 3 phases
draw_box(0.5, "TẦNG 1: CLUSTERING", "« ĐỊNH VỊ »", "Phân khúc thị trường\n(Market Segmentation)\n\n=> Truyền 'Cluster_ID'\nlàm Ngữ cảnh")
draw_box(4.4, "TẦNG 2: REGRESSION", "« ĐỊNH GIÁ »", "Dự báo mức lương\n(Market Valuation)\n\n=> Truyền 'Lương dự báo'\nlàm Lợi thế cạnh tranh")
draw_box(8.3, "TẦNG 3: CLASSIFICATION", "« ĐỊNH HƯỚNG »", "Dự báo Độ Hot\n(Behavior Prediction)\n\n=> Khuyến nghị chiến lược\ncho HR và Ứng viên")

# Arrows
arrow_kw = dict(arrowstyle="simple,head_width=1.0,head_length=1.0", color="#1c5a91")
ax.add_patch(patches.FancyArrowPatch((3.7, y_center), (4.4, y_center), **arrow_kw))
ax.add_patch(patches.FancyArrowPatch((7.6, y_center), (8.3, y_center), **arrow_kw))

# Draw data input/output arrows (optional stylistic touch)
ax.annotate("Raw Data", xy=(0.5, y_center), xytext=(-0.5, y_center), 
            arrowprops=dict(arrowstyle="->", color=TEXT_COL, lw=2),
            ha='center', va='center', fontsize=11, fontweight='bold', color=TEXT_COL)

ax.annotate("Actionable\nInsights", xy=(12.5, y_center), xytext=(11.5, y_center), 
            arrowprops=dict(arrowstyle="<-", color="#ff4444", lw=2),
            ha='center', va='center', fontsize=11, fontweight='bold', color="#ff4444")

# Main Title & Subtitle
plt.text(6, 6.0, "THIẾT KẾ HỆ SINH THÁI MÔ HÌNH HỖ TRỢ RA QUYẾT ĐỊNH (DSS)", 
         ha='center', va='center', fontsize=18, fontweight='bold', color=BOX_EDGE)
plt.text(6, 5.5, "Giải quyết bài toán Bất cân xứng thông tin trên LinkedIn", 
         ha='center', va='center', fontsize=13, color=TEXT_COL, fontstyle='italic')

plt.tight_layout()
out_path = fig_dir / "ml_architecture.png"
plt.savefig(out_path, dpi=300, facecolor="#ffffff", edgecolor='none')
print(f"Saved architecture visualization to {out_path}")
