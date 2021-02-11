from base64 import b64decode
from rdkit import Chem
from flask import Flask, send_file
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem import AllChem
from io import BytesIO

app = Flask(__name__)


def draw_reaction_smarts(smarts: str, use_smiles: bool = True, highlight: bool = True):
    rxn = AllChem.ReactionFromSmarts(smarts, useSmiles=use_smiles)
    d = Draw.MolDraw2DSVG(900, 300)
    if highlight:
        colors = [(0.3, 0.7, 0.9), (0.9, 0.7, 0.9), (0.6, 0.9, 0.3), (0.9, 0.9, 0.1)]
        d.DrawReaction(rxn, highlightByReactant=True, highlightColorsReactants=colors)
    else:
        d.DrawReaction(rxn)
    d.FinishDrawing()

    svg = d.GetDrawingText()
    return svg.replace('svg:', '').replace(':svg', '')


@app.route("/render_reaction/<image>.svg")
def render_reaction(image):
    smarts = b64decode(image).decode('ascii')
    svg = draw_reaction_smarts(smarts, use_smiles=True, highlight=True)
    return send_file(BytesIO(svg.encode('ascii')), mimetype='image/svg+xml')


if __name__ == '__main__':
    app.run()
