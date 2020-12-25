import argparse
import json
import pickle
from base64 import b64encode

import holoviews as hv
import numpy as np
import torch
import pandas as pd
from bokeh.models import ColumnDataSource, CategoricalColorMapper, Div, Row
from bokeh.events import Tap
from bokeh.plotting import figure, output_file, show
from bokeh.server.server import Server
from bokeh.palettes import d3

from utils.descriptors import ecfp
from utils.reactions import reaction_fps
from datasets import ReactionSmartsTemplatesDataset

hv.extension('bokeh')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help='path to dataset')
parser.add_argument('--model', '-m', help='path to saved model')
args = parser.parse_args()

dataset = args.dataset
MODEL_FILENAME = args.model
with open(args.model.replace(".pt", ".json")) as f:
    settings = json.load(f)["settings"]


def main_html_render_templates_model(doc):
    # construct model and load weights
    model = torch.load(MODEL_FILENAME, map_location='cpu')

    with open(dataset, "r") as f:
        reactions = []
        templates = []
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            rxn, tpl = line.split(",")
            reactions.append(rxn.rstrip())
            templates.append(tpl.rstrip())

    ds = ReactionSmartsTemplatesDataset(dataset, "cpu", settings["binary"])
    input = torch.stack([ds[i][0] for i in range(len(ds))])
    out = model(input)
    res = out.detach().cpu().numpy()

    data_dict = dict(x=res[:, 0],
                     y=res[:, 1],
                     sm=["http://localhost:5000/render_template/{}.svg".format(
                         b64encode(s.encode('ascii')).decode('ascii')) for s in templates])
    s = ColumnDataSource(data=data_dict)

    TOOLS = "box_zoom,reset"
    # create a new plot with the tools above, and explicit ranges
    p = figure(tools=TOOLS,
               x_range=(res[:, 0].min() - res[:, 0].std(), res[:, 0].max() + res[:, 0].std()),
               y_range=(res[:, 1].min() - res[:, 1].std(), res[:, 1].max() + res[:, 1].std()))
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False

    render_params = {"source": s,
                     "legend": "reaction_class",

                     "fill_alpha": 1,
                     "fill_color": "red",
                     "line_color": "red",
                     "line_alpha": 1,

                     "selection_fill_alpha": 1,
                     "selection_fill_color": "red",
                     "selection_line_color": "black",
                     "selection_line_alpha": 1,

                     "nonselection_fill_alpha": 1,
                     "nonselection_fill_color": "red",
                     "nonselection_line_color": "red",
                     "nonselection_line_alpha": 1
                     }
    p.circle('x', 'y', **render_params)

    def callback(event):
        indexes_active = s.selected.indices

        html_code_head = """
                <div>
                    <div>
            """
        html_code_body = """
                        <img
                            src="@sm" alt="@sm" height="100" width="300"
                            style="float: left; margin: 0px 15px 15px 0px;"
                    >
            """
        html_code_end = """
                    </div>
                </div>
            """
        final_html_code = html_code_head + "".join(
            [html_code_body.replace("@sm", data_dict["sm"][j]) for j in indexes_active]) + html_code_end

        layout.children[1] = Div(text=final_html_code)  # adjust the info on the right

    p.on_event(Tap, callback)

    div = Div()

    layout = Row(p, div)
    doc.add_root(layout)


if __name__ == '__main__':
    server = Server({'/': main_html_render_templates_model}, num_procs=1)
    server.start()
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
