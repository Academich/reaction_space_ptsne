from base64 import b64encode
from bokeh.models import ColumnDataSource
import holoviews as hv
import numpy as np
from bokeh.plotting import figure, output_file, show
import torch

from utils.descriptors import ecfp

hv.extension('bokeh')

DATASET = "tox21"
N_BITS = 4096

dataset = f"data/{DATASET}.smi"
model_filename = "model/model_dist_jaccard_per_30_bs_4000_epoch_100.pt"


def main_html_render():
    # construct model and load weights
    model = torch.load(model_filename, map_location='cpu')

    with open(dataset, "r") as f:
        smiles = [line.split()[0] for line in f.readlines()]
    fps = np.array([ecfp(smi) for smi in smiles])
    input = torch.from_numpy(fps)
    out = model(input)
    res = out.detach().cpu().numpy()

    s = ColumnDataSource(data=dict(x=res[:, 0], y=res[:, 1],
                                   smiles=["http://localhost:5000/render/{}.svg".format(
                                       b64encode(s.encode('ascii')).decode('ascii')) for s in smiles]))
    output_file(f"htmls/{DATASET}.html", title=f"{DATASET}", mode="cdn")

    TOOLS = "box_zoom,reset"
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@smiles" alt="@smiles" height="100" width="100"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="1"
            >
            </div>
        </div>
    """
    # create a new plot with the tools above, and explicit ranges
    p = figure(tooltips=TOOLTIPS, tools=TOOLS,
               x_range=(res[:, 0].min() - res[:, 0].std(), res[:, 0].max() + res[:, 0].std()),
               y_range=(res[:, 1].min() - res[:, 1].std(), res[:, 1].max() + res[:, 1].std()))
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False
    # add a circle renderer with vectorized colors and sizes
    p.circle('x', 'y', source=s, fill_alpha=0.6, line_color=None)
    # show the results
    show(p)


if __name__ == '__main__':
    main_html_render()
