# -*- coding: utf-8 -*-
# originally by haidi@hfut.edu.cn
# modified by Weiyi Xia
import dash
#import dash_table
#import dash_core_components as dcc
#import dash_html_components as html
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
#from flask_rest_service.crystal_toolkit.components.structure import StructureMoleculeComponent as smc
from uuid import uuid4
#from server import app

#def get_structure(filename='./POSCAR'):
#from pymatgen import Structure
#    return Structure.from_file(filename)

lattice_parameter_keys = ["a/Ang","b/Ang","c/Ang","alpha","beta","gamma"]

def get_lattice_df(structure):
    data = dict(zip(lattice_parameter_keys,
                    [[ii] for ii in structure.lattice.parameters]))
    return pd.DataFrame(data)

def get_coords_df(structure):
    data = {'index': range(1,len(structure)+1),
            'species': [x.symbol for x in structure.species],
            "x": structure.frac_coords[:,0],
            "y": structure.frac_coords[:,1],
            "z": structure.frac_coords[:,2]}
    return pd.DataFrame(data)

def dtable(df):
    return dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            editable=False,
            style_table={
                'maxHeight': '50ex',
                'overflowY': 'scroll',
                'width': '100%',
                'minWidth': '100%',
            },
            style_cell={
                'fontFamily': 'Open Sans',
                'textAlign': 'center',
                'height': '60px',
                'padding': '2px 22px',
                'whiteSpace': 'inherit',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'State'},
                    'textAlign': 'left'
                },
            ],
            # style header
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': 'white',
            },
            # style filter
            # style data
            style_data_conditional=[
                {
                    # stripped rows
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    # highlight one row
                    'if': {'row_index': 4},
                    "backgroundColor": "#3D9970",
                    'color': 'white'
                }
            ],
        )    

def get_table(structure):
    return html.Table([
         html.Thead(html.Tr(lattice_parameter_keys)),
         html.Tbody(html.Tr([[ii] for ii in structure.lattice.parameters]))
    ])

def get_layer(struct):
    #struct=get_structure(mm_id)
    lat_df=get_lattice_df(struct)
    coord_df=get_coords_df(struct)
    #structure_component = ctc.StructureMoleculeComponent(struct)
    #structure_component = smc(struct)
    my_button = html.Button("Swap Structure", id="change_structure_button")
    my_layout = [] # html.Div([structure_component.layout(), my_button])

#    return structure_component, html.Div(children=[
    return html.Div(children=[
        html.H1(children='STRUCTURE INFORMATION',style={"color":'blue'}),
        html.P(children="id: %s"%str(uuid4())),
        html.H2(children=struct.formula),
        html.H3(children='GEOMETRY STRUCTURE',style={"color":'blue'}),
        my_layout,
        html.H4(children='Lattice parameters',style={"color":'blue'}),
        dtable(lat_df),
        html.H4(children='Fractional coordinates',style={"color":'blue'}),
        dtable(coord_df)
    ])


#@app.callback(
#    Output(structure_component.id(), "data"),
#    [Input("change_structure_button", "n_clicks")],
#)
#def update_structure(n_clicks):
#    # on load, n_clicks will be None, and no update is required
#    # after clicking on the button, n_clicks will be an int and incremented
#    if not n_clicks:
#        raise PreventUpdate
#    return structures[n_clicks % 2]


#if __name__=='__main__':
#   import dash_bootstrap_components as dbc
#   from uuid import uuid4
#   external_stylesheets =  [dbc.themes.BOOTSTRAP]
#   app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#   app.title="MAG_DB"
#   app.server.secret_key = str(uuid4())
#   server = app.server
#   _,app.layout=get_layer(mm_id)
#   app.run_server( port=8052,host="0.0.0.0")
