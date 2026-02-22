"""
Manim animation: Constrained GradCAM Annotation Loss (no LaTeX required)
Run: manim -pql manim_animation.py ConstrainedCAMScene
"""
from manim import *


# Colour palette matching the app's dark theme
BG      = "#111827"
ACCENT  = "#f59e0b"   # amber – matches "Focus Here"
TEAL_C  = "#14b8a6"
RED_C   = "#ef4444"
GREEN_C = "#22c55e"
GRAY_C  = "#6b7280"


def bold(s, size=28, color=WHITE):
    return Text(s, font_size=size, weight=BOLD, color=color)


def normal(s, size=22, color=WHITE):
    return Text(s, font_size=size, color=color)


def code_text(s, size=22):
    return Text(s, font_size=size, color=ACCENT, font="Courier New")


class ConstrainedCAMScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── 0. Title ─────────────────────────────────────────────────────────────
        title = bold("Constrained GradCAM Annotation Loss", 34)
        sub   = normal("How doctor feedback guides attention\nwithout disrupting the rest", 20, GRAY_C)
        sub.next_to(title, DOWN, buff=0.4)
        self.play(Write(title))
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(sub))

        # ── 1. GradCAM recap ──────────────────────────────────────────────────────
        h1 = bold("1 — GradCAM: where is the model looking?", 26, YELLOW).to_edge(UP)
        self.play(Write(h1))

        cam_line1 = normal("CAM(x) = ReLU( Σ_k  α_k · A_k(x) )", 32, WHITE).shift(UP * 1.1)
        cam_line2 = normal("where  α_k = global-avg-pool( ∂y / ∂A_k )", 24, GRAY_C).next_to(cam_line1, DOWN, 0.4)
        cam_line3 = normal("A_k = activation map of filter k  (shape: 7×7)", 20, GRAY_C).next_to(cam_line2, DOWN, 0.3)

        self.play(Write(cam_line1))
        self.wait(0.3)
        self.play(FadeIn(cam_line2))
        self.play(FadeIn(cam_line3))
        self.wait(2)
        self.play(*[FadeOut(m) for m in [h1, cam_line1, cam_line2, cam_line3]])

        # ── 2. The annotation mask ────────────────────────────────────────────────
        h2 = bold("2 — Doctor annotates: 'Focus Here'", 26, YELLOW).to_edge(UP)
        self.play(Write(h2))

        tissue = Rectangle(width=3.2, height=2.4, color=GRAY_C, fill_color="#1c2333", fill_opacity=1).shift(LEFT * 2.8)
        tissue_lbl = normal("tissue image x", 16, GRAY_C).next_to(tissue, DOWN, 0.15)

        heatmap = Rectangle(width=3.2, height=2.4, color=ORANGE, fill_color=ORANGE, fill_opacity=0.3).move_to(tissue)
        heat_lbl = normal("CAM(x)  overlaid", 16, ORANGE).next_to(tissue, DOWN, 0.15)

        focus_box = RoundedRectangle(corner_radius=0.15, width=1.3, height=1.0,
                                     color=ACCENT, fill_color=ACCENT, fill_opacity=0.45,
                                     stroke_width=3).move_to(tissue).shift(RIGHT * 0.5 + UP * 0.3)
        focus_lbl = bold("𝒜", 38, ACCENT).next_to(focus_box, RIGHT, 0.2)
        focus_note = normal("doctor draws this region\n→ attention should grow here", 18, WHITE).shift(RIGHT * 2.8 + UP * 0.4)

        self.play(FadeIn(tissue), Write(tissue_lbl))
        self.play(FadeIn(heatmap), Transform(tissue_lbl, heat_lbl))
        self.wait(0.5)
        self.play(DrawBorderThenFill(focus_box), Write(focus_lbl))
        self.play(FadeIn(focus_note))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in [h2, tissue, heatmap, tissue_lbl, focus_box, focus_lbl, focus_note]])

        # ── 3. Optimization problem ───────────────────────────────────────────────
        h3 = bold("3 — The Constrained Optimization Problem", 26, YELLOW).to_edge(UP)
        self.play(Write(h3))

        obj = normal("min   L_CE( f_θ(x), y )   −   λ · mean( CAM_θ · 𝒜 )", 28, WHITE).shift(UP * 1.5)
        obj_note = normal("Minimise task loss, while maximising attention inside 𝒜", 20, GREEN_C).next_to(obj, DOWN, 0.35)

        con = normal("subject to:   mean( CAM_θ · (1−𝒜) )  ≤  mean( CAM_θ₀ · (1−𝒜) )  +  ε", 22, TEAL_C).next_to(obj_note, DOWN, 0.55)
        con_note = normal("outside-region activation must not increase beyond baseline", 18, GRAY_C).next_to(con, DOWN, 0.3)

        self.play(Write(obj))
        self.play(FadeIn(obj_note))
        self.wait(0.5)
        self.play(Write(con))
        self.play(FadeIn(con_note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in [h3, obj, obj_note, con, con_note]])

        # ── 4. Lagrangian ─────────────────────────────────────────────────────────
        h4 = bold("4 — Lagrangian (what the code optimises)", 26, YELLOW).to_edge(UP)
        self.play(Write(h4))

        terms = VGroup(
            normal("ℒ  =", 28, WHITE),
            normal("L_CE", 28, WHITE),
            normal("−  λ · mean( CAM · 𝒜 )", 28, GREEN_C),
            normal("+  2λ · mean( CAM · (1−𝒜) )", 28, RED_C),
        ).arrange(RIGHT, buff=0.25).shift(UP * 0.8)

        labels = VGroup(
            normal("task loss", 17, GRAY_C),
            normal("↑ inside region", 17, GREEN_C),
            normal("↓ outside region  (2× penalty)", 17, RED_C),
        )
        labels[0].next_to(terms[1], DOWN, 0.25)
        labels[1].next_to(terms[2], DOWN, 0.25)
        labels[2].next_to(terms[3], DOWN, 0.25)

        for t in terms:
            self.play(Write(t), run_time=0.5)
        self.play(*[FadeIn(l) for l in labels])

        impl = VGroup(
            code_text("# In weighted_loss.py", 18),
            code_text("l_inside  = -(cam_norm * focus_mask).mean()", 18),
            code_text("l_outside =  (cam_norm * (1 - focus_mask)).mean()", 18),
            code_text("loss += l_inside + 2.0 * l_outside", 18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(labels, DOWN, 0.55)
        impl_bg = SurroundingRectangle(impl, corner_radius=0.1, color=GRAY_C, fill_color="#1f2937", fill_opacity=0.85, buff=0.2)

        self.play(FadeIn(impl_bg), FadeIn(impl))
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in [h4, terms, labels, impl, impl_bg]])

        # ── 5. Before / After ─────────────────────────────────────────────────────
        h5 = bold("5 — Effect on the heatmap", 26, YELLOW).to_edge(UP)
        self.play(Write(h5))

        def make_tile(label_str, blob_size, blob_color, blob_opacity):
            bg = Rectangle(width=2.8, height=2.2, color="#1f2937", fill_color="#1f2937", fill_opacity=1)
            blob = Ellipse(width=blob_size[0], height=blob_size[1],
                           color=blob_color, fill_color=blob_color, fill_opacity=blob_opacity,
                           stroke_width=0).move_to(bg).shift(RIGHT * 0.3 + UP * 0.25)
            ann_box = DashedVMobject(
                Rectangle(width=1.2, height=0.9, color=ACCENT, stroke_width=2.5)
            ).move_to(bg).shift(RIGHT * 0.3 + UP * 0.25)
            lbl = bold(label_str, 18, WHITE).next_to(bg, DOWN, 0.15)
            return VGroup(bg, blob, ann_box, lbl)

        before_tile = make_tile("Before", (0.7, 0.5), BLUE, 0.45).shift(LEFT * 3.0)
        after_tile  = make_tile("After",  (1.5, 1.1), ORANGE, 0.70).shift(RIGHT * 3.0)
        arrow = Arrow(LEFT * 0.9, RIGHT * 0.9, color=WHITE, buff=0.15, stroke_width=4)

        note_before = normal("attention scattered\noutside dashed box", 16, GRAY_C).next_to(before_tile, UP, 0.2)
        note_after  = normal("attention concentrated\ninside dashed box ⚡", 16, ACCENT).next_to(after_tile, UP, 0.2)

        self.play(FadeIn(before_tile), FadeIn(note_before))
        self.play(GrowArrow(arrow))
        self.play(FadeIn(after_tile), FadeIn(note_after))
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in [h5, before_tile, after_tile, arrow, note_before, note_after]])

        # ── 6. End card ───────────────────────────────────────────────────────────
        end_title = bold("CancerScope · Constrained GradCAM Feedback", 28, WHITE)
        end_sub = normal(
            "Draw  ⚡ Focus Here  on the annotation canvas, then retrain.\n"
            "The heatmap will shift toward your region without disrupting the rest.",
            20, GRAY_C,
        ).next_to(end_title, DOWN, 0.5)
        self.play(Write(end_title))
        self.play(FadeIn(end_sub, shift=UP * 0.15))
        self.wait(4)
