import os

projects = {
    "projects/01-GeoPINN-Manifold__project-space": "GeoPINN-Manifold/artifacts/taichi_final/comparison.mp4",
    "projects/02-WavePINN-NIF-ComplexMedia__project-space": "WavePINN-NIF-ComplexMedia/artifacts/taichi_final/comparison.mp4",
    "projects/05-clothgnn__project-space": "clothgnn/artifacts/taichi_final/comparison.mp4",
    "projects/08-hgnn-clothdyn__project-space": "hgnn-clothdyn/artifacts/taichi_final/comparison.mp4",
    "projects/09-hgnn-nif-cloth__project-space_animation": "hgnn-nif-cloth/artifacts/taichi_final/comparison.mp4",
    "projects/12-nif-cloth4d-temporal__project-space": "nif-cloth4d-temporal/artifacts/taichi_final/comparison.mp4",
    "projects/13-nif-cloth4d__project-space": "nif-cloth4d/artifacts/taichi_final/comparison.mp4",
    "projects/17-wavepinn-nif__project-space": "wavepinn-nif/artifacts/taichi_final/wave_demo.mp4"
}

html_template = """
### Demo

<video src="{video_path}" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>
"""

for proj_dir, video_rel in projects.items():
    readme_path = os.path.join(proj_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()
            
        if "### Demo" not in content and "<video" not in content:
            lines = content.split('\n')
            insert_idx = 1
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    insert_idx = i + 1
                    break
            
            lines.insert(insert_idx, html_template.format(video_path=video_rel))
            
            with open(readme_path, "w") as f:
                f.write("\n".join(lines))
        print(f"Updated {readme_path}")

root_readme = "README.md"
if os.path.exists(root_readme):
    with open(root_readme, "r") as f:
        root_content = f.read()
        
    if "## Demos" not in root_content:
        demos_html = "\n## Demos\n\n"
        for proj_dir, video_rel in projects.items():
            name = proj_dir.split("/")[1].replace("__project-space", "").replace("_animation", "")
            vid_path = f"{proj_dir}/{video_rel}"
            demos_html += f"### {name}\n"
            demos_html += f'<video src="{vid_path}" controls="controls" style="max-width: 100%;"></video>\n\n'
            
        lines = root_content.split('\n')
        insert_idx = 1
        for i, line in enumerate(lines):
            if line.startswith("# "):
                insert_idx = i + 1
                break
                
        lines.insert(insert_idx, demos_html)
        with open(root_readme, "w") as f:
            f.write("\n".join(lines))
        print("Updated Root README")
