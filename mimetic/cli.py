"""mimetic — command line.

  mimetic say "[happy] Hello there!" -o hello.mp4 [--muscles] [--voice af_heart]
  mimetic export --text "..." -o out/            (json, Live Link csv, wav, obj + pc2 for Blender)
  mimetic export --audio talk.wav [--transcript "..."] -o out/
  mimetic live --vmc 127.0.0.1:39539 | --vts     (microphone -> VSeeFace / Warudo / VNyan / VTube Studio)
  mimetic serve                                  (web demo on http://127.0.0.1:8765)
  mimetic mocap video.mp4 -o out/ --render       (muscle motion capture from any face video)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def cmd_say(a):
    from mimetic.runtime.engine import Engine
    from mimetic.runtime.video import render_video
    eng = Engine.load()
    t0 = time.time()
    anim = eng.animate_text(a.text, voice=a.voice, fps=a.fps, speech_gain=a.gain, seed=a.seed,
                            emotion_gain=a.expressiveness)
    print(f"animated {anim.n / anim.fps:.2f}s in {time.time() - t0:.2f}s")
    render_video(anim, a.output, size=(a.size, a.size), show_muscles=a.muscles, yaw=a.yaw)
    print("wrote", a.output)


def cmd_export(a):
    import numpy as np
    import soundfile as sf
    from mimetic.runtime import export3d, outputs
    from mimetic.runtime.engine import Engine
    eng = Engine.load()
    if a.text:
        anim = eng.animate_text(a.text, voice=a.voice, fps=a.fps, speech_gain=a.gain, seed=a.seed,
                                emotion_gain=a.expressiveness)
    else:
        audio, sr = sf.read(a.audio, dtype="float32", always_2d=True)
        anim = eng.animate_audio(audio.mean(1), sr, text=a.transcript, emotion=(a.emotion, a.intensity), fps=a.fps)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    outputs.write_json(out / "animation.json", anim)
    outputs.write_livelink_csv(out / "livelink.csv", anim)
    if anim.audio is not None:
        sf.write(out / "audio.wav", anim.audio, anim.sr)
    topo = export3d.SkinTopology()
    frames = export3d.to_blender(topo.expand(anim.vertices))
    export3d.write_obj(out / "head.obj", frames[0], topo.faces)
    export3d.write_pc2(out / "head.pc2", frames, anim.fps)
    np.save(out / "jaw.npy", export3d.jaw_tracks(anim.acts))
    export3d.write_props(out)
    export3d.write_tongue_cache(out, anim)
    np.save(out / "gaze.npy", anim.gaze)
    np.save(out / "head_rotation.npy", anim.head)
    print(f"wrote {out}/ ({anim.n} frames at {anim.fps:g} fps)")


def cmd_live(a):
    import asyncio
    from mimetic.runtime import outputs
    from mimetic.runtime.engine import Engine
    from mimetic.runtime.live import AudioDriver, LiveSession
    eng = Engine.load()
    drv = AudioDriver.load()
    senders = []
    if a.vmc:
        host, _, port = a.vmc.partition(":")
        vmc = outputs.VMCSender(host or "127.0.0.1", int(port or 39539))
        senders.append(("vmc", vmc))
    vts = None
    loop = None
    if a.vts:
        loop = asyncio.new_event_loop()
        vts = outputs.VTubeStudioClient(token_path=str(Path.home() / ".mimetic_vts_token"))
        loop.run_until_complete(vts.connect())
        print("connected to VTube Studio")
    state = {"n": 0, "t": time.time()}

    def on_frame(acts, head, gaze):
        state["n"] += 1
        if state["n"] % 2:            # 50 Hz out
            return
        A_ = acts[None]
        if a.vmc:
            bs = eng.to_arkit(A_, gaze[None])[0]
            tg = getattr(drv, "last_tongue", None)
            vmc.send(bs, head, float(outputs.tongue_out(np.array(tg[-1:]))[0]) if tg else None)
        if vts is not None:
            p = outputs.vts_params(eng, A_, head[None], gaze[None])
            loop.run_until_complete(vts.send({k: v[0] for k, v in p.items()}))
    sess = LiveSession(eng, drv, on_frame, emotion=(a.emotion, a.intensity))
    sess.start()
    print("live: speak into the microphone (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sess.stop()


def cmd_mocap(a):
    import soundfile as sf
    import numpy as np
    from mimetic.runtime import export3d, outputs
    from mimetic.runtime.engine import Engine, upsample
    from mimetic.runtime.mocap import video_to_animation
    from mimetic.runtime.video import render_video
    eng = Engine.load()
    anim = video_to_animation(eng, a.video)
    if a.fps and a.fps != anim.fps:
        anim = upsample(anim, a.fps)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    outputs.write_json(out / "animation.json", anim)
    outputs.write_livelink_csv(out / "livelink.csv", anim)
    topo = export3d.SkinTopology()
    frames = export3d.to_blender(topo.expand(anim.vertices))
    export3d.write_obj(out / "head.obj", frames[0], topo.faces)
    export3d.write_pc2(out / "head.pc2", frames, anim.fps)
    np.save(out / "jaw.npy", export3d.jaw_tracks(anim.acts))
    np.save(out / "gaze.npy", anim.gaze)
    export3d.write_props(out)
    export3d.write_tongue_cache(out, anim)
    if anim.audio is not None:
        sf.write(out / "audio.wav", anim.audio, anim.sr)
    if a.render:
        render_video(anim, out / "muscle_mocap.mp4", show_muscles=a.muscles)
    print(f"wrote {out}/ ({anim.n} frames at {anim.fps:g} fps)")


def cmd_serve(a):
    from mimetic.web.server import main
    main()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mimetic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("say"); s.add_argument("text"); s.add_argument("-o", "--output", default="mimetic.mp4")
    s.add_argument("--voice", default="af_heart"); s.add_argument("--fps", type=float, default=30)
    s.add_argument("--size", type=int, default=512); s.add_argument("--muscles", action="store_true")
    s.add_argument("--gain", type=float, default=1.0); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--yaw", type=float, default=0.0)
    s.add_argument("--expressiveness", type=float, default=1.0, help="emotion gain (1 = natural, 1.5-2 = cartoony)")
    s.set_defaults(fn=cmd_say)
    e = sub.add_parser("export"); e.add_argument("--text"); e.add_argument("--audio"); e.add_argument("--transcript")
    e.add_argument("-o", "--output", default="mimetic_export"); e.add_argument("--voice", default="af_heart")
    e.add_argument("--fps", type=float, default=30); e.add_argument("--gain", type=float, default=1.0)
    e.add_argument("--emotion", default="neutral"); e.add_argument("--intensity", type=float, default=0.0)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--expressiveness", type=float, default=1.0)
    e.set_defaults(fn=cmd_export)
    l = sub.add_parser("live"); l.add_argument("--vmc", nargs="?", const="127.0.0.1:39539")
    l.add_argument("--vts", action="store_true"); l.add_argument("--emotion", default="auto")
    l.add_argument("--intensity", type=float, default=0.8)
    l.set_defaults(fn=cmd_live)
    w = sub.add_parser("serve"); w.set_defaults(fn=cmd_serve)
    m = sub.add_parser("mocap"); m.add_argument("video"); m.add_argument("-o", "--output", default="mimetic_mocap")
    m.add_argument("--fps", type=float, default=0); m.add_argument("--render", action="store_true")
    m.add_argument("--muscles", action="store_true")
    m.set_defaults(fn=cmd_mocap)
    a = ap.parse_args(argv)
    if a.cmd == "export" and not (a.text or a.audio):
        ap.error("export needs --text or --audio")
    a.fn(a)


if __name__ == "__main__":
    main()
