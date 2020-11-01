from base64 import b64encode

import holoviews as hv
import numpy as np
import torch
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show

from utils.descriptors import ecfp
from utils.reactions import reaction_fps
from config import config

hv.extension('bokeh')

mode = config.problem
settings = config.problem_settings[mode]

dataset = f"../data/{settings['filename']}"

MODEL_FILENAME = f"../model/rxn_dist_jaccard_per_30_bs_5000_epoch_10.pt"


def main_html_render_molecule():
    dev = "cpu"
    # construct model and load weights
    model = torch.load(MODEL_FILENAME, map_location='cpu')

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
    model = torch.load(MODEL_FILENAME, map_location='cpu')

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
        params = {"fp_method": settings["fp_method"],
                  "n_bits": settings["n_bits"],
                  "fp_type": settings["fp_type"],
                  "include_agents": settings["include_agents"],
                  "agent_weight": settings["agent_weight"],
                  "non_agent_weight": settings["non_agent_weight"],
                  "bit_ratio_agents": settings["bit_ratio_agents"]
                  }
        main_html_render_reaction(**params)
    else:
        pass
