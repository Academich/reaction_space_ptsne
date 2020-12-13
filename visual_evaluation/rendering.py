import argparse
import json
import pickle
from base64 import b64encode

import holoviews as hv
import numpy as np
import torch
import pandas as pd
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import d3

from utils.descriptors import ecfp
from utils.reactions import reaction_fps
from datasets import ReactionSmartsTemplatesDataset

hv.extension('bokeh')

parser = argparse.ArgumentParser()
parser.add_argument('--problem', '-p',
                    help='type of problem: mnist, reactions, reaction_templates, molecules')
parser.add_argument('--dataset', '-d', help='path to dataset')
parser.add_argument('--model', '-m', help='path to saved model')
parser.add_argument('--output', '-o', help='path to output html file')
parser.add_argument('--classes', '-c', help='labels of classes to render for reactions; example: 0,1,2,3',
                    default='1,2,3,4,5,6,7,8,9,10')
args = parser.parse_args()

dataset = args.dataset
MODEL_FILENAME = args.model
mode = args.problem
with open(args.model.replace(".pt", ".json")) as f:
    settings = json.load(f)["settings"]
if mode == "reactions":
    with open("data/visual_validation/rxnClasses.pickle", "rb") as f:
        classes = pickle.load(f)
output_filename = args.output


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
    output_file(f"htmls/{settings['filename']}.html", title=f"{settings['filename']}", mode="cdn")

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
        smiles = []
        labels = []
        for i, line in enumerate(f.readlines()):
            try:
                smi, lab = line.split(';')
                lab = lab.strip()
            except ValueError:
                smi = line.strip()
                lab = 0
            smiles.append(smi)
            labels.append(lab)
    fps = np.array([reaction_fps(s, **kwargs) for s in smiles])
    input = torch.from_numpy(fps).float()
    out = model(input)
    res = out.detach().cpu().numpy()
    classes_to_render = {classes[i] for i in args.classes.split(',')}
    data_dict = {"x": res[:, 0],
                 "y": res[:, 1],
                 "sm": [
                     "http://localhost:5000/render_reaction/{}.svg".format(b64encode(s.encode('ascii')).decode('ascii'))
                     for s in smiles]}
    if len(set(labels)) > 1:
        reaction_classes = [classes[i] for i in labels]
        data_dict["reaction_class"] = reaction_classes
        data_ds = pd.DataFrame.from_dict(data_dict)
        data_ds = data_ds[data_ds["reaction_class"].isin(classes_to_render)]
        s = ColumnDataSource(data_ds)
    else:
        reaction_classes = None
        s = ColumnDataSource(data=data_dict)

    output_file(output_filename, title=f"{settings['filename']}", mode="cdn")

    TOOLS = "box_zoom,reset"
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@sm" alt="@sm" height="300" width="900"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="1"
            >
            </div>
        </div>
    """
    # create a new plot with the tools above, and explicit ranges
    p = figure(tooltips=TOOLTIPS, tools=TOOLS,
               x_range=(res[:, 0].min() - res[:, 0].std(), res[:, 0].max() + res[:, 0].std()),
               y_range=(res[:, 1].min() - res[:, 1].std(), res[:, 1].max() + res[:, 1].std()),
               plot_width=1800,
               plot_height=900)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False
    # add a circle renderer with vectorized colors and sizes
    if reaction_classes is not None:
        palette = d3['Category10'][len(classes)]
        color_map = CategoricalColorMapper(factors=[v for v in classes.values()],
                                           palette=palette)
        color_transform = {'field': 'reaction_class',
                           'transform': color_map}
        p.circle('x', 'y', source=s,
                 fill_alpha=1,
                 fill_color=color_transform,
                 line_color=color_transform,
                 legend='reaction_class')
    else:
        p.circle('x', 'y', source=s,
                 fill_alpha=0.6)
    # show the results
    show(p)


def main_html_render_templates_model():
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

    # reactions
    # s = ColumnDataSource(data=dict(x=res[:, 0], y=res[:, 1],
    #                                sm=["http://localhost:5000/render_reaction/{}.svg".format(
    #                                    b64encode(s.encode('ascii')).decode('ascii')) for s in reactions]))

    # templates
    s = ColumnDataSource(data=dict(x=res[:, 0], y=res[:, 1],
                                   sm=["http://localhost:5000/render_template/{}.svg".format(
                                       b64encode(s.encode('ascii')).decode('ascii')) for s in templates]))

    output_file(f"htmls/{settings['filename']}.html", title=f"{settings['filename']}", mode="cdn")

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
    if mode == "molecules":
        print("molecule")
        main_html_render_molecule()
    elif mode == "reactions":
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
    elif mode == "reaction_templates":
        print("rxn_templates")
        main_html_render_templates_model()
    else:
        pass
