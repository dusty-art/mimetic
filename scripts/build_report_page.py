"""Build the shareable report page (experiments/report/index.html) from result JSONs and figures."""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image  # noqa: E402

from mimetic.paths import EXPERIMENTS  # noqa: E402

OUT = EXPERIMENTS / "report"


def img(path, width=900, q=82):
    p = Path(path)
    if not p.exists():
        return ""
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def video(path, max_mb=3.5):
    p = Path(path)
    if not p.exists() or p.stat().st_size > max_mb * 1e6:
        return ""
    return "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()


def load(name):
    p = EXPERIMENTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def main():
    ev = load("eval_results.json") or {}
    comp = load("composition.json") or {}
    rl = load("rl_articulation.json") or {}
    rl_long = load("rl_ppo_long.json") or {}
    fit = load("fit_report.json") or {}
    labels = {"ground_truth": "Real face (tracked)", "muscle_fit": "Muscle fit (ceiling)", "viseme_table": "Viseme table + emotion offsets",
              "direct": "Direct regression (same network, no muscles)", "noartic": "Mimetic without articulation constraints",
              "mimetic": "Mimetic (text path)", "live_audio": "Mimetic live audio (causal)",
              "live_audio_raw": "Live audio without post-processing"}
    rows = []
    for k in ("ground_truth", "muscle_fit", "viseme_table", "direct", "noartic", "mimetic", "live_audio", "live_audio_raw"):
        if k not in ev:
            continue
        r = ev[k]
        cls = ' class="ours"' if k in ("mimetic", "live_audio") else (' class="ref"' if k in ("ground_truth", "muscle_fit") else "")
        rows.append(f"<tr{cls}><th scope=row>{labels[k]}</th><td>{fmt(r['LVE_mm'])}</td><td>{fmt(r['ApCorr'])}</td>"
                    f"<td>{fmt(100 * r['Closure'], 0)}%</td><td>{fmt(100 * r['EmoAcc'], 0)}%</td><td>{fmt(r['Jerk_rel'])}</td>"
                    f"<td>{fmt(r['Pen_pct'], 2)}%</td><td>{fmt(r.get('AV_offset_ms'), 0) if k != 'ground_truth' else '—'}</td></tr>")
    comp_rows = []
    for s in ("mimetic", "direct", "viseme_table", "noartic"):
        if s not in comp:
            continue
        d = comp[s]
        for e in ("neutral", "happy", "angry", "sad", "surprised"):
            if e not in d:
                continue
            v = d[e]
            comp_rows.append(f"<tr{' class=ours' if s == 'mimetic' else ''}><th scope=row>{labels.get(s, s)}</th><td>{e}</td>"
                             f"<td>{fmt(v['rounding_mm'])}</td><td>{fmt(v.get('rounding_vs_neutral'))}</td>"
                             f"<td>{fmt(100 * v['closure'], 0)}%</td><td>{fmt(v['corner_lift_mm'])}</td><td>{fmt(v['cheek_raise_mm'])}</td></tr>")
    cues = load("emotion_cues.json") or {}
    cue_rows = []
    if cues:
        cols = [k for k in ("real", "muscle_fit", "mimetic", "direct") if k in cues]
        for e in ("happy", "sad", "angry", "fearful", "disgusted", "surprised"):
            if e not in cues.get("real", {}):
                continue
            for c, v in cues["real"][e].items():
                cells = "".join(f"<td>{cues[k][e][c]:+.2f}</td>" for k in cols)
                cue_rows.append(f"<tr><th scope=row>{e}</th><td style='text-align:left'>{c.replace('_', ' ')}</td>{cells}</tr>")
        share = "".join(f"<td>{100 * cues[k].get('_signature_share', 1.0):.0f}%</td>" for k in cols)
        cue_rows.append(f"<tr class=ours><th scope=row>share of real signature</th><td></td>{share}</tr>")
        cue_head = "".join(f"<th>{ {'real': 'Real', 'muscle_fit': 'Muscle fit', 'mimetic': 'Mimetic', 'direct': 'Direct regr.'}[k]}</th>" for k in cols)
    else:
        cue_head = ""
    mim = ev.get("mimetic", {})
    gt_close = ev.get("ground_truth", {}).get("Closure")
    angry = {s: comp.get(s, {}).get("angry", {}).get("closure") for s in ("mimetic", "direct")}
    apg = rl.get("apg", {}).get("final_mean_abs_err_mm")
    ppo = rl.get("ppo", {}).get("final_mean_abs_err_mm")
    ppo20 = rl_long.get("ppo", {}).get("final_mean_abs_err_mm")
    live = ev.get("live_audio", {})
    tiles = [
        (fmt(100 * mim["Closure"], 0) + "%" if mim else "—",
         "of p / b / m sounds seal the lips on unseen speakers" + (f" (tracked real faces: {fmt(100 * gt_close, 0)}%)" if gt_close else "")),
        (fmt(100 * angry["mimetic"], 0) + "%" if angry["mimetic"] is not None else "—",
         "of seals kept under anger" + (f" (same network without muscles: {fmt(100 * angry['direct'], 0)}%)" if angry["direct"] is not None else "")),
        ("57×", "fewer samples than PPO to learn 52-muscle articulation"),
        ("~135 ms", "from voice to face in live mode, using 3 ms of CPU per 20 ms of audio"),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)
    F = EXPERIMENTS / "figures"
    page = TEMPLATE.format(
        tiles=tiles_html,
        hero=img(F / "speech_emotion_composition.png", 1100),
        anatomy=img(EXPERIMENTS / "anatomy" / "expressions_rest2.png", 1000),
        fit=img(EXPERIMENTS / "fit_examples_final.png", 560),
        mocap=img(EXPERIMENTS / "mocap_compare.png", 960),
        emogrid=img(F / "emotion_grid.png", 880),
        rl=img(F / "rl_curves.png", 860),
        bars=img(F / "eval_bars.png", 1100),
        blender=img(EXPERIMENTS / "blender" / "frames.png", 768),
        web=img(EXPERIMENTS / "web_demo.png", 900),
        tongue=img(F / "tongue.png", 1100),
        v_emotions=video(EXPERIMENTS / "demos" / "emotions.mp4"), v_muscles=video(EXPERIMENTS / "demos" / "muscles.mp4"),
        v_bilabials=video(EXPERIMENTS / "demos" / "bilabials.mp4"), v_live=video(EXPERIMENTS / "demos" / "live_audio.mp4"),
        eval_rows="\n".join(rows) or "<tr><td colspan=8>Evaluation pending</td></tr>",
        comp_rows="\n".join(comp_rows) or "<tr><td colspan=7>Composition test pending</td></tr>",
        cue_rows="\n".join(cue_rows) or "<tr><td colspan=6>pending</td></tr>", cue_head=cue_head,
        fit_lips=fmt(fit.get("mean_lips_mm")), apg=fmt(apg), ppo=fmt(ppo), ppo20=fmt(ppo20),
        zero=fmt(rl_long.get("zero_policy_err_mm") or rl.get("zero_policy_err_mm")),
        n_test=ev.get("_meta", {}).get("test_clips", "—"), n_bil=ev.get("_meta", {}).get("bilabial_occurrences", "—"),
        live_lve=fmt(live.get("LVE_mm")), live_close=fmt(100 * live["Closure"], 0) + "%" if live else "—",
        live_emo=fmt(100 * live["EmoAcc"], 0) + "%" if live else "—",
        raw_emo=fmt(100 * ev["live_audio_raw"]["EmoAcc"], 0) + "%" if "live_audio_raw" in ev else "—",
        live_jerk=fmt(live.get("Jerk_rel")), raw_jerk=fmt(ev.get("live_audio_raw", {}).get("Jerk_rel")),
        mim_emo=fmt(100 * mim["EmoAcc"], 0) + "%" if mim else "—",
        dir_emo=fmt(100 * ev["direct"]["EmoAcc"], 0) + "%" if "direct" in ev else "—",
        dir_pen=fmt(ev.get("direct", {}).get("Pen_pct")),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page)
    print(OUT / "index.html", f"{len(page) / 1e6:.1f} MB")


TEMPLATE = """<title>Mimetic Muscle Face</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: one reading column (~70ch) with full-bleed figure bands; results first, method after. */
:root {{
  --bg: #f4f6f8; --panel: #ffffff; --ink: #18202b; --muted: #5a6776; --line: #dde3ea;
  --accent: #a3213a; --accent-soft: #f6e3e7; --ours: #fbf1f3;
  --display: "Bricolage Grotesque", "Avenir Next", system-ui, sans-serif;
  --body: "Source Sans 3", "Helvetica Neue", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f141a; --panel: #161d26; --ink: #e5eaf0; --muted: #98a5b3; --line: #273240;
  --accent: #ff7b90; --accent-soft: #3a1d25; --ours: #24161c; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #0f141a; --panel: #161d26; --ink: #e5eaf0; --muted: #98a5b3; --line: #273240;
  --accent: #ff7b90; --accent-soft: #3a1d25; --ours: #24161c; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font: 17px/1.6 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
main {{ max-width: 1100px; margin: 0 auto; padding-block: 40px 80px; display: grid; gap: 56px; }}
.col {{ max-width: 70ch; }}
h1, h2, h3 {{ font-family: var(--display); line-height: 1.12; text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(2.2rem, 5vw, 3.6rem); letter-spacing: -0.02em; }}
h2 {{ font-size: clamp(1.5rem, 3vw, 2rem); letter-spacing: -0.01em; }}
h3 {{ font-size: 1.1rem; }}
p {{ margin: 0; }}
section {{ display: grid; gap: 18px; }}
.eyebrow {{ font: 600 0.78rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }}
.lede {{ font-size: 1.2rem; color: var(--muted); max-width: 62ch; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
.tile {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; display: grid; gap: 4px; }}
.tile b {{ font: 700 1.9rem/1.1 var(--display); color: var(--accent); font-variant-numeric: tabular-nums; }}
.tile span {{ color: var(--muted); font-size: 0.92rem; line-height: 1.35; }}
figure {{ margin: 0; display: grid; gap: 8px; }}
figure img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); background: #fff; }}
figcaption {{ color: var(--muted); font-size: 0.92rem; max-width: 80ch; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; align-items: start; }}
.two > * {{ min-width: 0; }}
.table {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th[scope=row], thead th:first-child {{ text-align: left; white-space: normal; min-width: 220px; }}
thead th {{ font: 600 0.75rem/1.2 var(--mono); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }}
tr.ours {{ background: var(--ours); }} tr.ours th {{ color: var(--accent); }}
tr.ref th, tr.ref td {{ color: var(--muted); }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
code, .mono {{ font-family: var(--mono); font-size: 0.88em; }}
ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }}
.note {{ background: var(--accent-soft); border-radius: 10px; padding: 14px 18px; }}
.vids {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
.vids video {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); background: #e9ecef; display: block; }}
.aus {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.aus span {{ font: 0.8rem/1 var(--mono); border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: var(--panel); }}
</style>
<main>
<header style="display:grid;gap:18px">
  <div class="eyebrow">Research prototype · trained on one laptop</div>
  <h1>Mimetic: talking faces driven by muscles</h1>
  <p class="lede">Text or live voice goes in. A controller decides which of 47 facial muscles and 5 jaw actuators fire, and a soft-tissue simulation turns those forces into skin motion. The same muscle signals drive a simulated head, ARKit avatars, Live2D, and Blender.</p>
  <div class="tiles">{tiles}</div>
</header>

<section>
  <div class="eyebrow">Speech and emotion share the mouth</div>
  <h2>A smile yields to an “oo”; lips still seal on “m”</h2>
  <p class="col">Speech and emotion pathways add their drives in muscle activation space, where muscles saturate and antagonize. Articulation constraints from phonetics are optimized through the differentiable face, so emotion cannot steal a lip closure.</p>
  <figure><img alt="Simulated face saying oo and m under neutral, happy and angry emotion, with muscle overlay" src="{hero}"><figcaption>Rows: /UW/ in “Who knew?” and /M/ in “Mom made more.” Columns: neutral, happy, angry, each with the muscle overlay (red = active).</figcaption></figure>
  <div class="table"><table>
    <thead><tr><th>System</th><th>Emotion</th><th>Rounding (mm)</th><th>vs neutral</th><th>p/b/m closed</th><th>Corner lift (mm)</th><th>Cheek raise (mm)</th></tr></thead>
    <tbody>{comp_rows}</tbody></table></div>
  <p class="col" style="color:var(--muted);font-size:.92rem">Eight new sentences synthesized by TTS (not in the training data). Rounding = mouth width at unrounded vowels minus width at rounded sounds; corner lift and cheek raise show how much of the emotion remains visible.</p>
</section>

<section>
  <div class="eyebrow">Inside the mouth</div>
  <h2>Teeth ride the jaw; the tongue follows phonetics</h2>
  <p class="col">No talking-face dataset records the tongue, because trackers cannot see it. So the tongue follows phonetic targets: tip to the ridge for t/d/n/l/s, between the teeth for th, back raised for k/g, low for open vowels. Critical consonants dominate and transitions anticipate, as in coarticulation. In live mode the audio model's phoneme predictions drive it, and it exports as ARKit <code>tongueOut</code> and a Blender point cache.</p>
  <figure><img alt="Close-ups of the mouth showing tongue positions for th, t, l, k and ah" src="{tongue}"></figure>
</section>

<section>
  <div class="eyebrow">Watch it</div>
  <h2>Generated from text, with sound</h2>
  <div class="vids">
    <figure><video controls playsinline preload="metadata" src="{v_emotions}"></video><figcaption>One sentence, five emotions (the tag switches mid-stream).</figcaption></figure>
    <figure><video controls playsinline preload="metadata" src="{v_muscles}"></video><figcaption>Muscle overlay: happy, disgusted, fearful.</figcaption></figure>
    <figure><video controls playsinline preload="metadata" src="{v_bilabials}"></video><figcaption>“Peter Piper picked a peck…”: every p and b seals.</figcaption></figure>
    <figure><video controls playsinline preload="metadata" src="{v_live}"></video><figcaption>Live audio model on a real recording of an actor it never heard: happy, angry, surprised. Emotion comes from the voice.</figcaption></figure>
  </div>
  <p class="col" style="color:var(--muted);font-size:.92rem">Press play for sound. The first three are generated from text alone: Kokoro voice and timing, then muscles, then skin. The fourth is generated from audio alone.</p>
</section>

<section>
  <div class="eyebrow">Held-out speakers · {n_test} clips · {n_bil} bilabials</div>
  <h2>Accuracy against real faces</h2>
  <div class="table"><table>
    <thead><tr><th>System</th><th>Lip error (mm) ↓</th><th>Aperture corr ↑</th><th>p/b/m closed ↑</th><th>Emotion read ↑</th><th>Jerk vs real</th><th>Lips interpenetrate ↓</th><th>Face behind voice (ms)</th></tr></thead>
    <tbody>{eval_rows}</tbody></table></div>
  <div class="note col">What the muscle layer buys, against the same network trained to output the face directly: the same lip accuracy, every seal kept under emotion, lips that interpenetrate on essentially no frames (vs {dir_pen}%), and motion closer to real dynamics. What it does not buy: the direct model's emotions read more clearly ({dir_emo} vs {mim_emo}). Both read above real faces because they produce each label's typical expression.</div>
  <figure><img alt="Bar charts comparing systems on lip error, closure, aperture correlation and emotion recognition" src="{bars}"></figure>
  <h3>How much of each emotion shows on the face</h3>
  <div class="table"><table>
    <thead><tr><th>Emotion</th><th style="text-align:left">Cue (mm vs neutral)</th>{cue_head}</tr></thead>
    <tbody>{cue_rows}</tbody></table></div>
  <p class="col" style="color:var(--muted);font-size:.92rem">Lip error is the per-frame maximum over lip vertices (the MeshTalk/FaceFormer LVE), averaged over frames. The emotion judge is a classifier on expression geometry trained on real faces and their muscle fits. Real tracked faces close on fewer bilabials than the threshold expects because the tracker smooths fast lip contacts.</p>
</section>

<section>
  <div class="eyebrow">The body</div>
  <h2>52 actuators on a soft-tissue face</h2>
  <div class="two">
    <div style="display:grid;gap:14px">
      <p>The face is MediaPipe's 468-vertex topology, so every frame tracked from any video corresponds vertex-for-vertex to the simulated skin. The rest shape is the measured average neutral face of 78 speakers.</p>
      <p>Each muscle has a bony origin, a skin insertion, a FACS action unit and a maximal excursion. The solver adds skin elasticity, volume conservation (cheeks bulge, lips thicken), lip and eyelid contact, eyelids sliding on the eyes, force-length saturation, and a temporomandibular jaw whose forward glide was calibrated from real video.</p>
      <div class="aus"><span>AU1 frontalis medialis</span><span>AU4 corrugator</span><span>AU6 orbicularis oculi</span><span>AU9 levator labii alaeque nasi</span><span>AU10 levator labii sup.</span><span>AU12 zygomaticus major</span><span>AU14 buccinator</span><span>AU15 depressor anguli oris</span><span>AU17 mentalis</span><span>AU18/22 orbicularis oris</span><span>AU20 risorius</span><span>AU26 jaw</span></div>
    </div>
    <figure><img alt="Twelve expressions and speech shapes produced by combining muscles, front and side views" src="{anatomy}"><figcaption>Hand-set muscle combinations: smile, big smile, sad, angry, surprise, disgust, fear; mouth shapes for oo, m, ee, aa, f.</figcaption></figure>
  </div>
</section>

<section>
  <div class="eyebrow">Muscle motion capture</div>
  <h2>31 hours of ordinary video, turned into muscle activations</h2>
  <div class="two">
    <div style="display:grid;gap:14px">
      <p>A physics-decoder autoencoder fits muscle activations to every tracked frame: the decoder is the differentiable surrogate of the simulator, so the only way to reconstruct a face is to find muscles that physically produce it. Mean lip residual across all clips: <b>{fit_lips} mm</b>, compared with 3.85 mm for a neutral face.</p>
      <p>The decompositions read like FACS: disgust uses the nose wrinkler, sadness the chin raiser, anger the lip-corner depressor.</p>
      <div class="note">Three data problems were found and fixed. GRID's transcript folders are mislabeled for 13+ speakers. The CREMA-D video mirror has no audio. And lip contact does not transfer between faces, which made 18% of ground-truth frames interpenetrate until contact-preserving transfer was added.</div>
    </div>
    <figure><img alt="Tracked real faces beside their muscle fits and the active muscles" src="{fit}"><figcaption>Left: tracked real face. Middle: muscle fit. Right: muscles, red = active.</figcaption></figure>
  </div>
  <figure><img alt="Unseen speaker video frames above the muscle-driven head that reproduces them" src="{mocap}"><figcaption>`mimetic mocap`: an unseen speaker with glasses (top) retargeted through muscle space onto the simulated head (bottom).</figcaption></figure>
</section>

<section>
  <div class="eyebrow">Reinforcement learning</div>
  <h2>Learning to articulate with 52 muscles</h2>
  <div class="two">
    <div style="display:grid;gap:14px">
      <p>The policy commands muscle excitations through first-order activation dynamics to reach a new mouth target every 100 ms. Same network, same environment, two learners.</p>
      <ul>
        <li>No movement: {zero} mm error</li>
        <li>PPO, 3 M steps: {ppo} mm; 20 M steps: {ppo20} mm</li>
        <li><b>Analytic policy gradient through the muscle face, 3 M steps: {apg} mm</b></li>
      </ul>
      <p>The analytic gradient matched PPO's best training error after 55 k steps, 57× fewer. For overactuated robot faces, this favors learning on a calibrated differentiable muscle model, then transferring.</p>
    </div>
    <figure><img alt="Learning curves: analytic policy gradient falls far below PPO" src="{rl}"></figure>
  </div>
</section>

<section>
  <div class="eyebrow">Using it</div>
  <h2>Avatars, VTubers, Blender</h2>
  <div class="two">
    <div style="display:grid;gap:12px">
      <p><code>mimetic say "[happy:0.8] Hello!" -o hello.mp4</code>: text with emotion tags to video. Kokoro TTS gives exact phoneme timing.</p>
      <p><code>mimetic live --vmc</code> / <code>--vts</code>: microphone to VSeeFace, Warudo, VNyan (52 ARKit blendshapes) or VTube Studio (Live2D parameters).</p>
      <p><code>mimetic export</code>: Live Link Face CSV, JSON, OBJ + PC2 point cache. The Blender add-on imports the head with jaw, teeth, eyes and voice, or keyframes ARKit shape keys on your own character.</p>
      <p>Live mode: a causal convolutional network with 80 ms lookahead, then a post-processor that removes jitter (jerk {raw_jerk}× real → {live_jerk}×) and shows the voice's emotion on the face ({raw_emo} → {live_emo} recognized). The emotion yields to lip seals on p/b/m/f/v. On held-out speakers it reaches {live_lve} mm lip error with {live_close} of bilabials sealed.</p>
    </div>
    <figure><img alt="The animated head rendered in Blender at two frames" src="{blender}"><figcaption>Generated by the Blender add-on (Eevee): skin through a mesh cache, plus jaw, teeth, irises and gaze.</figcaption></figure>
  </div>
  <figure><img alt="Same sentence spoken in eight emotions" src="{emogrid}"><figcaption>“I can't believe it.” in neutral, happy, sad, angry, fearful, disgusted, surprised and contempt, at the /IY/ vowel.</figcaption></figure>
</section>

<section class="col">
  <div class="eyebrow">Limits</div>
  <h2>What it does not do yet</h2>
  <ul>
    <li>The face is a single 468-vertex shell with procedural teeth and eyes, and no wrinkles. The tongue follows phonetic rules, since no corpus records it.</li>
    <li>The controller under-produces the inner-brow raise of surprise, which the muscle fit captures.</li>
    <li>Live emotion is capped by voice-emotion recognition, about 50% on unseen actors. Pinning the emotion is more reliable.</li>
    <li>Risorius and buccinator are nearly redundant, so the fit's split between them is not identifiable from video.</li>
    <li>Ground truth comes from a monocular tracker, which under-reports fast lip contacts and depth.</li>
    <li>Emotional training speech is acted, from 14 sentences. Generalization was measured on new TTS sentences, not on real recordings of them.</li>
    <li>RAVDESS is non-commercial, so this model is for research only. Each clip records its source, so retraining on GRID + CREMA-D only is possible.</li>
  </ul>
</section>
</main>
"""

if __name__ == "__main__":
    main()
