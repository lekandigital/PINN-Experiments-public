import os

projects = [
    "projects/01-GeoPINN-Manifold__project-space/GeoPINN-Manifold",
    "projects/02-WavePINN-NIF-ComplexMedia__project-space/WavePINN-NIF-ComplexMedia",
    "projects/05-clothgnn__project-space/clothgnn",
    "projects/08-hgnn-clothdyn__project-space/hgnn-clothdyn",
    "projects/09-hgnn-nif-cloth__project-space_animation/hgnn-nif-cloth",
    "projects/12-nif-cloth4d-temporal__project-space/nif-cloth4d-temporal",
    "projects/13-nif-cloth4d__project-space/nif-cloth4d",
    "projects/17-wavepinn-nif__project-space/wavepinn-nif",
]

prefer = [
    "poster.png",
    "poster_nif.png",
    "poster_baseline.png",
    "preview.png",
    "wave_demo.gif",
    "comparison.gif",
]

for p in projects:
    candidates = []
    for root, _, files in os.walk(p):
        for f in files:
            if f.lower().endswith((".png", ".gif", ".jpg", ".jpeg")):
                candidates.append(os.path.join(root, f))

    pick = None
    for name in prefer:
        for c in candidates:
            if c.endswith(name):
                pick = c
                break
        if pick:
            break

    if not pick:
        for c in candidates:
            if "frame_0000" in c:
                pick = c
                break

    if not pick and candidates:
        pick = candidates[0]

    print(f"{p} -> {pick}")
