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
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import d3
from bokeh.server.server import Server

from utils.descriptors import ecfp
from utils.reactions import reaction_fps
from datasets import ReactionSmartsTemplatesDataset

hv.extension('bokeh')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help='path to dataset')
parser.add_argument('--model', '-m', help='path to saved model')
parser.add_argument('--classes', '-c', help='labels of classes to render for reactions; example: 0,1,2,3',
                    default='1,2,3,4,5,6,7,8,9,10')
parser.add_argument('--additional', '-a', help='path to additional file with reaction smiles', default=None)
args = parser.parse_args()

dataset = args.dataset
MODEL_FILENAME = args.model
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
    factors = [v for v in classes.values()]

def main_html_render_reaction(doc):
    # construct model and load weights
    model = torch.load(MODEL_FILENAME, map_location='cpu')
    model.eval()

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

    if args.additional is not None:
        P1 = "/home/ma/Рабочий стол/paper_reaxys_data/oseltamivir_cut.csv"
        with open(P1, 'r') as af:
            addit_smiles1 = [line.strip() for line in af.readlines()]
            addit_labels1 = ["11" for i in addit_smiles1]
        fps_addit1 = np.array([reaction_fps(s, **params) for s in addit_smiles1])
        input_addit1 = torch.from_numpy(fps_addit1).float()
        out_addit1 = model(input_addit1)
        res_addit1 = out_addit1.detach().cpu().numpy()
        classes_to_render.add("Oseltamivir")
        classes["11"] = "Oseltamivir"
        res = np.vstack((res, res_addit1))
        smiles += addit_smiles1
        labels += addit_labels1

        P2 = "/home/ma/Рабочий стол/paper_reaxys_data/darunavir_cut.csv"
        with open(P2, 'r') as af:
            addit_smiles2 = [line.strip() for line in af.readlines()]
            addit_labels2 = ["12" for i in addit_smiles2]
        fps_addit2 = np.array([reaction_fps(s, **params) for s in addit_smiles2])
        input_addit2 = torch.from_numpy(fps_addit2).float()
        out_addit2 = model(input_addit2)
        res_addit2 = out_addit2.detach().cpu().numpy()
        classes_to_render.add("Darunavir")
        classes["12"] = "Darunavir"
        res = np.vstack((res, res_addit2))
        smiles += addit_smiles2
        labels += addit_labels2

    data_dict = {"x": res[:, 0],
                 "y": res[:, 1],
                 "sm": [
                     "http://localhost:5000/render_reaction/{}.svg".format(b64encode(s.encode('ascii')).decode('ascii'))
                     for s in smiles]}

    if args.additional is not None:
        sizes = [5 for _ in range(len(smiles) - len(addit_smiles1) - len(addit_smiles2))] + [20 for _ in range(len(addit_smiles1) + len(addit_smiles2))]
        data_dict["sizes"] = sizes
        data_dict["colors"] = ["grey" for _ in range(len(smiles) - len(addit_smiles1) - len(addit_smiles2))] + ["red" for _ in range(len(addit_smiles1))] + ["purple" for _ in range(len(addit_smiles2))]

    if len(set(labels)) > 1:
        reaction_classes = [classes[i] for i in labels]
        for i in range(reaction_classes.index("Oseltamivir")):
            reaction_classes[i] = "Validation dataset points"
        print(reaction_classes)
        data_dict["reaction_class"] = reaction_classes
        data_ds = pd.DataFrame.from_dict(data_dict)
        # data_ds = data_ds[data_ds["reaction_class"].isin(classes_to_render)]
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

    palette = d3['Category10'][len(factors)]
    if args.additional is not None:
        factors.append("Additional")
        palette += ("#73472d",)
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
                     "fill_color": "colors",
                     "line_color": "colors",
                     "line_alpha": alpha,

                     "selection_fill_alpha": alpha,
                     "selection_fill_color": "colors",
                     "selection_line_color": "black",
                     "selection_line_alpha": alpha,

                     "nonselection_fill_alpha": alpha,
                     "nonselection_fill_color": "colors",
                     "nonselection_line_color": "colors",
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
    server = Server({'/': main_html_render_reaction}, num_procs=1)
    server.start()
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
