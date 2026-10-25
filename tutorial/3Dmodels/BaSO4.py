import html as html_lib
import json
from IPython.display import HTML, display

def compare_cif(s1, s2, title1="Prototype", title2="Structure", cutoffs=None):
    if cutoffs is None:
        cutoffs = {"Ba-O": 3.15, "O-Ba": 3.15, "S-O": 1.80, "O-S": 1.80}
        
    cif_1 = json.dumps(s1.to(fmt="cif"))
    cif_2 = json.dumps(s2.to(fmt="cif"))
    
    js_cutoffs = json.dumps(cutoffs)

    srcdoc = f"""<!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
      <style>
        body {{ margin: 0; background: #fff; font-family: sans-serif; display: flex; gap: 1em; justify-content: center; }}
        h2 {{ font-weight: lighter; font-size: 22px; text-align: center; margin: 0.4em 0; }}
        .box {{ border: 2px solid black; border-radius: 10px; padding: 0.5em; width: 400px; height: 400px; position: relative; }}
      </style>
    </head>
    <body>
      <div><h2>{html_lib.escape(title1)}</h2><div id="v1" class="box"></div></div>
      <div><h2>{html_lib.escape(title2)}</h2><div id="v2" class="box"></div></div>
      <script>
        const cutoffs = {js_cutoffs};

        function render(id, cif) {{
          const v = $3Dmol.createViewer(document.getElementById(id), {{ backgroundColor: "white" }});
          const m = v.addModel(cif, "cif", {{ doAssembly: true, duplicateAssemblyAtoms: true, assignBonds: false }});
          const atoms = m.selectedAtoms({{}});
          atoms.forEach(a => {{ a.bonds = []; a.bondOrder = []; }});

          for (let i = 0; i < atoms.length; i++) {{
            for (let j = i + 1; j < atoms.length; j++) {{
              const a = atoms[i], b = atoms[j];
              const maxD = cutoffs[`${{a.elem}}-${{b.elem}}`] || 0.0;
              if (Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) <= maxD) {{
                a.bonds.push(b.index); a.bondOrder.push(1);
                b.bonds.push(a.index); b.bondOrder.push(1);
              }}
            }}
          }}
          v.setStyle({{}}, {{ sphere: {{ scale: 0.3 }}, stick: {{ radius: 0.15 }} }});
          v.addUnitCell(); v.zoomTo(); v.render();
        }}
        render("v1", {cif_1});
        render("v2", {cif_2});
      </script>
    </body>
    </html>"""

    display(HTML(f'<iframe srcdoc="{html_lib.escape(srcdoc, quote=True)}" style="width:100%;height:500px;border:none;"></iframe>'))