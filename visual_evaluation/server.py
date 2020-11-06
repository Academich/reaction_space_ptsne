from base64 import b64decode
from rdkit import Chem
from flask import Flask, send_file
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem import AllChem
from io import BytesIO

app = Flask(__name__)


@app.route("/render_molecule/<image>.svg")
def render_molecule(image):
    mol = Chem.MolFromSmiles(b64decode(image).decode('ascii'))
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(100, 100)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace(':svg', '').replace('svg:', '')
    return send_file(BytesIO(svg.encode('ascii')), mimetype='image/svg+xml')


@app.route("/render_reaction/<image>.svg")
def render_reaction(image):
    smarts = b64decode(image).decode('ascii')
    rxn = AllChem.ReactionFromSmarts(smarts, useSmiles=True)
    drawer = Draw.MolDraw2DSVG(900, 300)
    colors = [(0.3, 0.7, 0.9), (0.9, 0.7, 0.9), (0.6, 0.9, 0.3), (0.9, 0.9, 0.1)]
    drawer.DrawReaction(rxn, highlightByReactant=True, highlightColorsReactants=colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace(':svg', '').replace('svg:', '')
    return send_file(BytesIO(svg.encode('ascii')), mimetype='image/svg+xml')


if __name__ == '__main__':
    app.run()
