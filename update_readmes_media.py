import os
import re

PROJECTS = {
    "projects/01-GeoPINN-Manifold__project-space": {
        "video": "GeoPINN-Manifold/artifacts/taichi_final/comparison.mp4",
        "poster": "GeoPINN-Manifold/artifacts/taichi_final/poster.png",
    },
    "projects/02-WavePINN-NIF-ComplexMedia__project-space": {
        "video": "WavePINN-NIF-ComplexMedia/artifacts/taichi_final/comparison.mp4",
        "poster": "WavePINN-NIF-ComplexMedia/artifacts/taichi_final/poster.png",
    },
    "projects/05-clothgnn__project-space": {
        "video": "clothgnn/artifacts/taichi_final/comparison.mp4",
        "poster": "clothgnn/artifacts/taichi_final/poster.png",
    },
    "projects/08-hgnn-clothdyn__project-space": {
        "video": "hgnn-clothdyn/artifacts/taichi_final/comparison.mp4",
        "poster": "hgnn-clothdyn/artifacts/taichi_final/poster.png",
    },
    "projects/09-hgnn-nif-cloth__project-space_animation": {
        "video": "hgnn-nif-cloth/artifacts/taichi_final/comparison.mp4",
        "poster": "hgnn-nif-cloth/artifacts/taichi_final/poster.png",
    },
    "projects/12-nif-cloth4d-temporal__project-space": {
        "video": "nif-cloth4d-temporal/artifacts/taichi_final/comparison.mp4",
        "poster": "nif-cloth4d-temporal/artifacts/taichi_final/poster.png",
    },
    "projects/13-nif-cloth4d__project-space": {
        "video": "nif-cloth4d/artifacts/taichi_final/comparison.mp4",
        "poster": "nif-cloth4d/artifacts/taichi_final/poster_nif.png",
    },
    "projects/17-wavepinn-nif__project-space": {
        "video": "wavepinn-nif/artifacts/taichi_final/wave_demo.mp4",
        "poster": "wavepinn-nif/artifacts/taichi_final/poster.png",
    },
}

VIDEO_TAG_RE = re.compile(r"<video\s+src=\"[^\"]+\"[^>]*>.*?</video>", re.DOTALL)


def update_project_readme(project_dir, data):
    readme_path = os.path.join(project_dir, "README.md")
    if not os.path.exists(readme_path):
        return

    with open(readme_path, "r") as f:
        content = f.read()

    rel_video = data["video"]
    rel_poster = data["poster"]
    demo_md = f"[![Demo]({rel_poster})]({rel_video})"

    if "### Demo" in content:
        content = VIDEO_TAG_RE.sub(demo_md, content)
    else:
        content = content.replace("<video", f"### Demo\n\n{demo_md}\n\n<video", 1)

    with open(readme_path, "w") as f:
        f.write(content)


def update_root_readme():
    root_readme = "README.md"
    if not os.path.exists(root_readme):
        return

    with open(root_readme, "r") as f:
        content = f.read()

    # Replace each video tag in order of the projects list
    for project_dir, data in PROJECTS.items():
        rel_video = f"{project_dir}/{data['video']}"
        rel_poster = f"{project_dir}/{data['poster']}"
        demo_md = f"[![Demo]({rel_poster})]({rel_video})"
        content = VIDEO_TAG_RE.sub(demo_md, content, count=1)

    with open(root_readme, "w") as f:
        f.write(content)


for project_dir, data in PROJECTS.items():
    update_project_readme(project_dir, data)

update_root_readme()
print("Updated README media embeds.")
