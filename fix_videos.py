import os
import re

projects = [
    "projects/01-GeoPINN-Manifold__project-space",
    "projects/02-WavePINN-NIF-ComplexMedia__project-space",
    "projects/05-clothgnn__project-space",
    "projects/08-hgnn-clothdyn__project-space",
    "projects/09-hgnn-nif-cloth__project-space_animation",
    "projects/12-nif-cloth4d-temporal__project-space",
    "projects/13-nif-cloth4d__project-space",
    "projects/17-wavepinn-nif__project-space"
]

def fix_file(filepath, is_root=False):
    if not os.path.exists(filepath): return
    with open(filepath, 'r') as f:
        content = f.read()

    def repl(m):
        src = m.group(1)
        if src.startswith("http"):
            return m.group(0)
        if not is_root and not src.startswith("projects/"):
            proj_dir = os.path.dirname(filepath)
            src = os.path.join(proj_dir, src)
        abs_url = f"https://raw.githubusercontent.com/lekandigital/PINN-Experiments-public/main/{src}"
        return f'<video src="{abs_url}"'

    new_content = re.sub(r'<video\s+src="([^"]+)"', repl, content)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Fixed {filepath}")

fix_file("README.md", True)
for p in projects:
    fix_file(os.path.join(p, "README.md"), False)
