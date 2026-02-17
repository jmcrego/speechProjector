# Minimal NeurIPS/ICLR-style diagram with larger grouped boxes
# - Depthwise + Pointwise grouped
# - Positional + Transformer grouped
# - Linear + Activation grouped
# - Titles of grouped boxes on the left, vertical
# - Data shapes to the right of arrows
# - Boxes width reduced to fit content, arrows touch blocks

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(8, 9))

block_height = 0.42

def block(center_y, text, width=0.25, lw=0.9, fontsize=10, color="black"):
    x = 0.55 - width / 2
    rect = Rectangle((x, center_y - block_height/2),
                     width, block_height,
                     fill=False,
                     color=color,
                     linewidth=lw)
    ax.add_patch(rect)
    ax.text(0.55, center_y, text, ha='center', va='center', fontsize=fontsize)
    return rect

def group_box(top_y, bottom_y, width=0.45, lw=1.5, title=None):
    x = 0.55 - width / 2
    height = top_y - bottom_y
    rect = Rectangle((x, bottom_y), width, height + 0.05, fill=False, linewidth=lw)
    ax.add_patch(rect)
    if title:
        # Vertical title on the left of the box
        ax.text(x + 0.04, bottom_y + height/2, title,
                ha='center', va='center', rotation=90, fontsize=12, fontweight='bold')
    return rect

# Y positions
ys = {
    "input": 9.6,
    "rms_pre": 8.7,
    "dw": 7.8,
    "pw": 6.9,
    "pos": 5.8,
    "tr": 4.9,
    "lin": 3.8,
    "act": 2.9,
    "rms_post": 2.0,
    "output": 1.1
}

# Data shapes (output of previous block)
shapes = {
    "input": "[B, T, D_audio]",
    "rms_pre": "[B, T, D_audio]",
    "dw": "[B, T, D_audio]",
    "pw": "[B, T', D_audio]",
    "pos": "[B, T', D_audio]",
    "tr": "[B, T', D_audio]",
    "lin": "[B, T', D_llm]",
    "act": "[B, T', D_llm]",
    "rms_post": "[B, T', D_llm]",
    "output": "[B, T', D_llm]"
}

# Draw blocks
blocks = {}
blocks["input"] = block(ys["input"], "Audio Frames", color="white")
blocks["rms_pre"] = block(ys["rms_pre"], "RMSNorm")
blocks["dw"] = block(ys["dw"], "Depthwise Conv1D")
blocks["pw"] = block(ys["pw"], "Pointwise Conv1D")
blocks["pos"] = block(ys["pos"], "Positional Encoding")
blocks["tr"] = block(ys["tr"], "Transformer Encoder")
blocks["lin"] = block(ys["lin"], "Linear Projection")
blocks["act"] = block(ys["act"], "Activation")
blocks["rms_post"] = block(ys["rms_post"], "Post RMSNorm")
blocks["output"] = block(ys["output"], "LLM Embeddings", color="white")

# Draw larger group boxes with vertical titles
group_box(ys["dw"] + 0.3, ys["pw"] - 0.3, title="Conv\nFront-End")
group_box(ys["pos"] + 0.3, ys["tr"] - 0.3, title="Temporal\nModeling")
group_box(ys["lin"] + 0.3, ys["act"] - 0.3, title="Projection\nHead")

# Draw arrows with data shapes to the right
ordered = [
    "input", "rms_pre",
    "dw", "pw",
    "pos", "tr",
    "lin", "act",
    "rms_post", "output"
]

for i in range(len(ordered)-1):
    start = ordered[i]
    end = ordered[i+1]
    y_start = ys[start] - block_height/2 
    y_end = ys[end] + block_height/2 
    ax.annotate("",
        xy=(0.55, y_end-0.03),
        xytext=(0.55, y_start+0.03),
        arrowprops=dict(arrowstyle="-|>", linewidth=0.9)
    )
    # Data shape to the right
    ax.text(0.57, (y_start + y_end)/2, shapes[start],
            ha='left', va='center', fontsize=10)

ax.set_xlim(0, 1)
ax.set_ylim(0.5, 10.2)
ax.axis("off")

plt.tight_layout()
plt.savefig("/Users/josepcrego/Desktop/projector_block_diagram_grouped_titles_vertical.png", dpi=300, bbox_inches="tight")
plt.show()
