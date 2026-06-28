import json

path = r"D:\Users\SUPRATIK\HCI Research\IEM-IIT_HCI_Project\results\system_architecture_corrected.excalidraw"
doc = json.load(open(path, encoding="utf-8"))
els = doc["elements"]
by_id = {e["id"]: e for e in els}

# ---------------------------------------------------------------------------
# Clarify the two GENERATOR feedback loops.
# Both dashed arrows physically land on the FC-projection box (G's first
# trainable layer); relabel them so the reader knows the target is the whole
# Generator G, not only that one layer.
# ---------------------------------------------------------------------------
spec = by_id["HbvmbtFNRM3xn_KXRkON7"]          # label on aspecfb
spec["text"] = "update G\n(+0.1*L_spectral)"
spec["originalText"] = spec["text"]
spec["width"] = 110

adv = by_id["LNmyJXTTxZfsOSnzd2FS1"]           # label on aadvfb
adv["text"] = "update G\n(-E[C(G(z))])"
adv["originalText"] = adv["text"]
adv["width"] = 110

# Make both feedback arrowheads land cleanly on the LEFT edge of the generator
# trainable stack (top of FC projection, y=206) so they unmistakably point at G.
asp = by_id["aspecfb"]      # start (92.5, 693)
# left margin -> up -> into FC projection top-left (end at 100, 210)
asp["points"] = [[0, 0], [-40.5, 0], [-40.5, -483], [7.5, -483]]
asp["width"] = 48.0
asp["height"] = 483.0

adva = by_id["aadvfb"]      # start (338.977..., 789.955...)
sx, sy = adva["x"], adva["y"]
# route right -> up -> into FC projection top-RIGHT edge (end at 320, 210)
ex, ey = 320.0, 210.0
adva["points"] = [[0, 0], [54.41, 0], [54.41, sy - ey - (sy - 789.955)],
                  [ex - sx, -(sy - ey)]]
# rebuild cleanly with absolute targets to avoid drift
adva["points"] = [[0, 0], [54.41, 0], [54.41, ey - sy], [ex - sx, ey - sy]]
adva["width"] = max(abs(p[0]) for p in adva["points"])
adva["height"] = max(abs(p[1]) for p in adva["points"])

# ---------------------------------------------------------------------------
# Fix the SYNTH-DATA arrow: it used to stop in mid-air inside Column B.
# Re-point it so the head clearly enters the decoder input box B1
# (Filtered EEG input) at its top-centre.
# ---------------------------------------------------------------------------
syn = by_id["asynth"]       # start (304.6027..., 504.9219...)
sx, sy = syn["x"], syn["y"]
# along -> up to top -> right -> down into B1 top-centre (605, 118)
syn["points"] = [[0, 0], [105.0, -0.5], [105.0, 100.0 - sy],
                 [605.0 - sx, 100.0 - sy], [605.0 - sx, 118.0 - sy]]
syn["width"] = max(abs(p[0]) for p in syn["points"])
syn["height"] = max(abs(p[1]) for p in syn["points"])

json.dump(doc, open(path, "w", encoding="utf-8"), ensure_ascii=False)
print("Patched:", path)
print("aspecfb end :", (asp["x"] + asp["points"][-1][0], asp["y"] + asp["points"][-1][1]))
print("aadvfb  end :", (adva["x"] + adva["points"][-1][0], adva["y"] + adva["points"][-1][1]))
print("asynth  end :", (syn["x"] + syn["points"][-1][0], syn["y"] + syn["points"][-1][1]))
print("spec label  :", repr(spec["text"]))
print("adv  label  :", repr(adv["text"]))
