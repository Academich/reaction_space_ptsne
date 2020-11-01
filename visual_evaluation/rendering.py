from base64 import b64encode
import json

import holoviews as hv
import numpy as np
import torch
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show

from utils.descriptors import ecfp
from utils.reactions import reaction_fps

hv.extension('bokeh')

with open("rend_config.json") as f:
    config_all = json.load(f)

mode = config_all['mode']
config = config_all[f"{mode}_rend_params"]

dataset = f"../data/{config['dataset_name']}"
model_filename = f"../model/{config['model_filename']}"

print(config)


def main_html_render_molecule():
    dev = "cpu"
    # construct model and load weights
    model = torch.load(model_filename, map_location='cpu')

    with open(dataset, "r") as f:
        smiles = [line.split()[0] for line in f.readlines()]
    fps = np.array([ecfp(smi) for smi in smiles])
    input = torch.from_numpy(fps)
    out = model(input)
    res = out.detach().cpu().numpy()

    s = ColumnDataSource(data=dict(x=res[:, 0], y=res[:, 1],
                                   smiles=["http://localhost:5000/render_molecule/{}.svg".format(
                                       b64encode(s.encode('ascii')).decode('ascii')) for s in smiles]))
    output_file(f"htmls/{config['dataset_name']}.html", title=f"{config['dataset_name']}", mode="cdn")

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


def main_html_render_reaction(**kwargs):
    # construct model and load weights
    model = torch.load(model_filename, map_location='cpu')

    with open(dataset, "r") as f:
        smarts = []
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            smarts.append(line.replace(",", ">").rstrip("\n"))
    fps = np.array([reaction_fps(s, **kwargs) for s in smarts])
    input = torch.from_numpy(fps).float()
    out = model(input)
    res = out.detach().cpu().numpy()

    s = ColumnDataSource(data=dict(x=res[:, 0], y=res[:, 1],
                                   sm=["http://localhost:5000/render_reaction/{}.svg".format(
                                       b64encode(s.encode('ascii')).decode('ascii')) for s in smarts]))
    output_file(f"htmls/{config['dataset_name']}.html", title=f"{config['dataset_name']}", mode="cdn")

    TOOLS = "box_zoom,reset"
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@sm" alt="@sm" height="100" width="300"
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
    if mode == "molecule":
        print("molecule")
        main_html_render_molecule()
    elif mode == "reaction":
        print("reaction")
        main_html_render_reaction(**config['params'])
    else:
        pass
