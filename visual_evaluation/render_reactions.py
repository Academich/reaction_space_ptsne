import argparse
import json
import pickle
from base64 import b64encode

import holoviews as hv
import numpy as np
import torch
import pandas as pd
from bokeh.models import ColumnDataSource, CategoricalColorMapper, Row, Div
from bokeh.events import Tap
from bokeh.plotting import figure
from bokeh.palettes import d3, Category10_10
from bokeh.server.server import Server

from utils.reactions import reaction_fps

hv.extension('bokeh')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help='path to dataset')
parser.add_argument('--model', '-m', help='path to saved model')
parser.add_argument('--classes', '-c', help='labels of classes to render for reactions; example: 0,1,2,3',
                    default='1,2,3,4,5,6,7,8,9,10')
parser.add_argument('--additional', '-a', help='Paths to additional file with reaction smiles separated by commas',
                    default=None)
args = parser.parse_args()

dataset = args.dataset
with open(args.model.replace(".pt", ".json")) as f:
    settings = json.load(f)["settings"]
params = {"fp_method": settings["fp_method"],
          "n_bits": settings["n_bits"],
          "fp_type": settings["fp_type"],
          "include_agents": settings["include_agents"],
          "agent_weight": settings["agent_weight"],
          "non_agent_weight": settings["non_agent_weight"],
          "bit_ratio_agents": settings["bit_ratio_agents"]
          }
with open("data/visual_validation/rxnClasses.pickle", "rb") as f:
    classes = pickle.load(f)

additional_classes = []

model = torch.load(args.model, map_location='cpu')
model.eval()


def main_html_render_reaction(doc):
    with open(dataset, "r") as ff:
        smiles = []
        labels = []
        for i, line in enumerate(ff.readlines()):
            try:
                smi, lab = line.split(';')
                lab = lab.strip()
            except ValueError:
                smi = line.strip()
                lab = 0
            smiles.append(smi)
            labels.append(lab)
    fps = np.array([reaction_fps(s, **params) for s in smiles])
    input = torch.from_numpy(fps).float()
    out = model(input)
    res = out.detach().cpu().numpy()
    classes_to_render = {classes[i] for i in args.classes.split(',')}

    all_addit_smiles = []
    if args.additional is not None:
        for p in args.additional.split(','):
            addit_name = p.split("/")[-1].split('.')[0]
            addit_name = addit_name.capitalize()
            with open(p, 'r') as af:
                addit_smiles = [line.strip() for line in af.readlines()]
                addit_labels = [addit_name for _ in addit_smiles]
            fps_addit = np.array([reaction_fps(s, **params) for s in addit_smiles])
            input_addit = torch.from_numpy(fps_addit).float()
            out_addit = model(input_addit)
            res_addit = out_addit.detach().cpu().numpy()
            classes_to_render.add(addit_name)
            additional_classes.append(addit_name)
            res = np.vstack((res, res_addit))
            all_addit_smiles += addit_smiles
            labels += addit_labels
    smiles += all_addit_smiles

    data_dict = {"x": res[:, 0],
                 "y": res[:, 1],
                 "sm": [
                     "http://localhost:5000/render_reaction/{}.svg".format(b64encode(s.encode('ascii')).decode('ascii'))
                     for s in smiles]}

    if args.additional is not None:
        sizes = [5 for _ in range(len(smiles) - len(all_addit_smiles))] + [20 for _ in range(len(all_addit_smiles))]
        data_dict["sizes"] = sizes

    if len(set(labels)) > 1:
        if args.additional is not None:
            reaction_classes = ["Validation dataset points" if i in classes else i for i in labels]
            classes_to_render.add("Validation dataset points")
        else:
            reaction_classes = [classes[i] for i in labels]
        data_dict["reaction_class"] = reaction_classes
        data_ds = pd.DataFrame.from_dict(data_dict)
        data_ds = data_ds[data_ds["reaction_class"].isin(classes_to_render)]
        s = ColumnDataSource(data_ds)
    else:
        reaction_classes = None
        data_ds = pd.DataFrame.from_dict(data_dict)
        s = ColumnDataSource(data_ds)

    TOOLS = "box_zoom,reset,tap"

    # create a new plot with the tools above, and explicit ranges
    p = figure(tools=TOOLS,
               x_range=(res[:, 0].min() - res[:, 0].std(), res[:, 0].max() + res[:, 0].std()),
               y_range=(res[:, 1].min() - res[:, 1].std(), res[:, 1].max() + res[:, 1].std()),
               plot_width=1600,
               plot_height=800)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = True
    # add a circle renderer with vectorized colors and sizes

    if additional_classes:
        palette = Category10_10[:len(additional_classes)] + ("#C0C0C0",)
        factors = sorted(additional_classes) + ["Validation dataset points"]
    else:
        # factors = sorted(list(classes_to_render))
        factors = [v for v in classes.values()]
        palette = d3['Category10'][len(factors)]

    color_map = CategoricalColorMapper(factors=factors,
                                       palette=palette)
    color_transform = {'field': 'reaction_class',
                       'transform': color_map}
    if args.additional is not None:
        size = "sizes"
        alpha = 0.85
    else:
        size = 5
        alpha = 1

    render_params = {"source": s,
                     "legend": "reaction_class",
                     "size": size,

                     "fill_alpha": alpha,
                     "fill_color": color_transform,
                     "line_color": color_transform,
                     "line_alpha": alpha,

                     "selection_fill_alpha": alpha,
                     "selection_fill_color": color_transform,
                     "selection_line_color": "black",
                     "selection_line_alpha": alpha,

                     "nonselection_fill_alpha": alpha,
                     "nonselection_fill_color": color_transform,
                     "nonselection_line_color": color_transform,
                     "nonselection_line_alpha": alpha
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
                            src="@sm" alt="@sm" height="300" width="900"
                            style="float: left; margin: 0px 15px 15px 0px;"
                    >
            """
        html_code_end = """
                    </div>
                </div>
            """
        final_html_code = html_code_head + "".join(
            [html_code_body.replace("@sm", data_ds["sm"].iloc[j]) for j in indexes_active]) + html_code_end

        layout.children[1] = Div(text=final_html_code)  # adjust the info on the right

    p.on_event(Tap, callback)

    div = Div()

    layout = Row(p, div)
    doc.add_root(layout)


if __name__ == '__main__':
    server = Server({'/': main_html_render_reaction},
                    num_procs=1,
                    address="0.0.0.0",
                    allow_websocket_origin=["0.0.0.0:5006", "localhost:5006"])
    server.start()
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
