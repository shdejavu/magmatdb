#coding=utf-8
import os
import time
import json
import pprint
import base64
from io import StringIO, BytesIO
from bson.objectid import ObjectId

# Flask
from flask import render_template, make_response, request, abort, Flask, jsonify, send_from_directory
from flask_restful import Api, Resource, reqparse
from flask_rest_service import app, api, mongo, cache, ALLOWED_EXTENSIONS, basedir, MAPI_KEY
#from flask_cache import Cache

from werkzeug.utils import secure_filename
#from werkzeug.contrib.cache import GAEMemcachedCache

# pymatgen
from pymatgen.core import Composition,Structure,Molecule
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import *
#from pymatgen.core.composition import Composition
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as sga

# own scripts (under 'flask_rest_service')
from flask_rest_service.lammps import set_lammps_data
from flask_rest_service.structure_dash import *

# algebra, plot
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import seaborn as sns
import ternary

# stats
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# additional libraries
from collections import Counter
import random

mpr=MPRester(MAPI_KEY)
#cache = GAEMemcachedCache()

# physical constants, unit conversions
abohr = 0.529177
Ry = 13.6057
Tesla_to_emu_per_cm3 = 625 / math.atan(1) # 10**4 / (4 pi)

# formats
fmt_d1="%.1f"
fmt_d2="%.2f"
fmt_d3="%.3f"
fmt_d4="%.4f"
fmt_d6="%10.6f"

# Tolerances
stol=0.1  # to be used in SpacegroupAnalyzer for symmetry finding

# I/O
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

def save_file(entry_id, fmt='vasp'):
    #assert isinstance(struct,Structure)
    file_dir = os.path.join(basedir, app.config['DATA_FOLDER'])
    check_folder(file_dir)
    file_name = os.path.join(file_dir, entry_id+'.'+fmt) 
    if not os.path.isfile(file_name):
       struct = get_structure(entry_id)
       if fmt == 'vasp':
          struct.to('poscar', file_name)
       elif fmt == 'lammps':
          set_lammps_data(file_name, struct, struct.symbol_set)
       else:
          struct.to(fmt, file_name)

def check_folder(dir):   
    if not os.path.exists(dir):
        os.makedirs(dir)

# calculate stability
def calculate_stability(entry):
    functional = d["pseudo_potential"]["functional"]
    syms = ["{} {}".format(functional, l)
            for l in d["pseudo_potential"]["labels"]]
    entry = ComputedEntry(Composition(d["unit_cell_formula"]),
                          d["output"]["final_energy"],
                          parameters={"hubbards": d["hubbards"],
                                      "potcar_symbols": syms})
    data = m.get_stability([entry])[0]
    for k in ("e_above_hull", "decomposes_to"):
        d["analysis"][k] = data[k]

def get_stability(comp,energy,mpara):
    compat=MaterialsProjectCompatibility()
    mycomp=Composition(comp)
    unprocessed_entries=mpr.get_entries_in_chemsys([e.symbol for e in mycomp.elements])
    print("len unprocessed entries: %d"%(len(unprocessed_entries)))
    my_eng_per_atom=energy
    processed_entries=compat.process_entries(unprocessed_entries)
    print("len processed entries: %d"%(len(processed_entries)))
    my_en=ComputedEntry(mycomp,my_eng_per_atom*mycomp.num_atoms,parameters=mpara)
    corrections_dict=compat.get_corrections_dict(my_en)
    pretty_corr=["{}:{}".format(k,round(v,3)) for k,v in corrections_dict.items()]
    print('pretty_corr')
    print(pretty_corr)
    my_en.correction=sum(corrections_dict.values())
    processed_entries.append(my_en)
    pd=PhaseDiagram(processed_entries)
    print("formation energy %f "%pd.get_form_energy(my_en))
    #print("len stable phases: %" %)
    #print(pd.stable_entries)
    #print("e above the hull %f"%pd.get_e_above_hull(my_en))
    #print("decompose to: ")
    #decomp=[]
    #for comp  in pd.get_decomp_and_e_above_hull(my_en)[0]:
    #    decomp.append(comp.composition.formula.replace(' ',''))
    #print(decomp)
    return pd.get_form_energy(my_en),mpr.get_stability([my_en])[0]

# Find entries
def get_entry(mm_id):
    return mongo.db.data.find_one({'entry_id':mm_id})

def get_entries(mm_ids):
    return [get_entry(mm_id) for mm_id in mm_ids]

def get_structure(mm_id):
    ret = mongo.db.data.find_one({'entry_id':mm_id},{'structure':1})
    return Structure.from_dict(ret['structure'])

# Find stats data
def get_stats():
    return mongo.db.data.find_one({'stats_id': 'mag-201'})

def get_total_count():
    #total_count = mongo.db.data.find({},{}).count() - 1
    total_count = mongo.db.data.count_documents({}) - 1
    return total_count

def get_counts(symbols):
    elements = [Element(i) for i in symbols]
    filter = {"composition."+element.symbol: {'$gt':0}
              for element in elements}
    filter["data.system.nelem"] = len(elements)
    #return mongo.db.data.find(filter=filter).count()
    return mongo.db.data.count_documents(filter=filter)

# Make data from a given structure
def get_coords(structure):
    nat = structure.num_sites
    data = {'index': range(1, nat+1),
            'species': [x.symbol for x in structure.species],
            'wl':  get_wyckoffs(structure),
            'x':   [fmt_d6%structure.frac_coords[i,0] for i in range(nat)],
            'y':   [fmt_d6%structure.frac_coords[i,1] for i in range(nat)],
            'z':   [fmt_d6%structure.frac_coords[i,2] for i in range(nat)],
            'lmm': [fmt_d2%structure.site_properties['lmm'][i] for i in range(nat)],
            'soc_100': ['.' for i in range(nat)],
            'soc_001': ['.' for i in range(nat)]
    }
    return pd.DataFrame(data)

def get_wyckoffs(structure, stol=stol):
    return sga(structure, symprec=stol).get_symmetry_dataset()['wyckoffs']

def get_symmetrized_structure_data(structure, stol=stol):
    ss = sga(structure, symprec=stol).get_symmetrized_structure()
    el = [x.symbol for x in structure.species]
    wp = []
    for i, g in enumerate(ss.equivalent_indices):
        for j in g:
            wp.append( ss.wyckoff_symbols[i] )
    data = {'equivalent_indices': ss.equivalent_indices,
            'equivalent_sites':   ss.equivalent_sites,
            'wyckoff_letters:':   ss.wyckoff_letters,
            'wyckoff_symbols':    ss.wyckoff_symbols,
            'wyckoff_positions':  wp,
            'wyckoff_sites':      ["{}({})".format(el[i],
                                                   wp[i]) for i in range(structure.num_sites)]
}
    return data

# PNG
def generate_png(fig):
    # print out png to canvas
    canvas = FigureCanvasAgg(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    data = png_output.getvalue()
    # response
    response = make_response(data)
    response.headers['Content-Type'] = 'image/png'
    response.headers['Content-Length'] = len(data)
    return response

# formats
def html_formula(f):
    return re.sub(r"([\d.]+)", r"<sub>\1</sub>", f)

def formula_to_symbols(formula):
    return re.sub('\d+', '',
                  re.sub('\(|\)', '',
                         re.sub('([A-Z])', r' \1', str(formula) ))).split()

def count_format(num):
    if num == 0:
        return 'No entries'
    elif num == 1:
        return str(num) + ' entry'
    else:
        return '{:,}'.format(num) + ' entries'

def get_reduced_formula(comp):
    return {'reduced': comp.reduced_formula,
            'html':    html_formula(comp.reduced_formula),
            'unit':    comp.get_reduced_formula_and_factor()[1]}

# get data about magnetic moments
def get_magmom(m, n):
    if m['value'] is None:
        data = {'tot': '.',
                'ave': '.'}
    else:
        if 'cell' in m['per']:
            data = {'tot': fmt_d2%( m['value'] ),
                    'ave': fmt_d2%( m['value'] / n )}
        else:
            data = {'tot': fmt_d2%( m['value'] * n ),
                    'ave': fmt_d2%( m['value'] )}
    return data

def get_magpol(value):
    if value is None:
        data = {'SI': '',
                'cgs': ''}
    else:
        data = {'SI':  fmt_d2%( value ),
                'cgs': fmt_d1%( value * Tesla_to_emu_per_cm3 )}
    return data

# data for index.html
def thumbnails_information(max_entries=50, filter={}):

    data=[]

    if max_entries is None:
        entries = mongo.db.data.find(filter)
    elif max_entries == -1:
        total_count = get_total_count()
        ii = 6
        nn = list(range(1, ii)) + sorted(random.sample(range(ii, total_count), 5))
        entries = get_entries( ['MMD-'+str(n) for n in nn] )
    else:
        total_count = get_total_count()
        number = min(max_entries, total_count)
        entries = mongo.db.data.find(filter)[0:number]

    for entry in entries:
        # structure data and composition
        st = Structure.from_dict(entry['structure'])
        comp = st.composition

        # ID, Formula
        thumb_dict = {'mmid': entry['entry_id'],
                      'formula': get_reduced_formula(comp),
                      'num_sites': st.num_sites}

        # Space group
        for key in ['crystal_system', 'symbol', 'number']:
            thumb_dict[key] = entry['data']['spacegroup'][key]

        # Thermodynamic properties:
        # Formation energies
        for key in ['decomposition', 'above_hull']:
            value = entry['data']['formation_energy'][key]['value']
            unit = entry['data']['formation_energy'][key]['unit']
            if value is None:
                thumb_dict[key] = '.'
            elif value < 0.001 and key == 'above_hull':
                thumb_dict[key] = '0 (stable)'
            else:
                if unit == 'eV':
                    thumb_dict[key] = fmt_d3%( value)
                elif unit == 'meV':
                    thumb_dict[key] = fmt_d3%( value * 0.001 )

        # Synthesis
        thumb_dict['synthesis'] = search_method( entry['data']['notes']['synthesis']['status'] )

        # Magnetic properties:
        # Averaged magnetic moment
        thumb_dict['magmom'] = get_magmom(entry['data']['magnetic_moment']['total_magnetic_moment'], st.num_sites)

        # Magnetic polarization
        thumb_dict['magpol'] = get_magpol(entry['data']['magnetic_moment']['magnetic_polarization']['value'])

        # Magnetic anisotropy constant
        for key, key2 in zip(['MAC_ac', 'MAC_bc', 'MAC_ba', 'MAC_da'],
                             ['a-c', 'b-c', 'b-a', 'd-a']):
            value = entry['data']['magnetic_anisotropy']['constant'][key2]
            if value is None:
                thumb_dict[key] = '.'
            else:
                thumb_dict[key] = fmt_d2%( value )

        # Magnetic easy axis
        value = entry['data']['magnetic_anisotropy']['easy_axis']
        if value is None:
            thumb_dict['easy_axis'] = '.'
        else:
            thumb_dict['easy_axis'] = value

        # Curie temperature
        value = entry['data']['critical_temperature']['Curie_temperature']['value']
        if value is None:
            thumb_dict['Curie_T'] = '.'
        else:
            thumb_dict['Curie_T'] = fmt_d1%( value )

        # Method: theory or experiment
        thumb_dict['method'] = 'DFT' #entry['data']['notes']['method']

        # References
        thumb_dict['lnk'] = ''
        thumb_dict['url'] = '.'
        mpid = entry['data']['notes']['mpid']
        if mpid:
            thumb_dict['lnk'] = mpid
            thumb_dict['url'] = 'https://materialsproject.org/materials/' + mpid + '/'
        elif entry['data']['notes']['references']['doi']:
            thumb_dict['lnk'] = 'DOI link'
            thumb_dict['url'] = entry['data']['notes']['references']['url']
        elif entry['data']['notes']['entry']['by']:
            thumb_dict['lnk'] = entry['data']['notes']['entry']['by']
            thumb_dict['url'] = '/about'
        else:
            thumb_dict['lnk'] = ''
            thumb_dict['url'] = '.'

        data.append(thumb_dict)

    return data

def random_selector():
    stats = get_stats()
    total_count = get_total_count()
    for i in range(total_count):
        st = get_structure( 'MMD-' + str(random.randint(4, total_count)) )
        symbols = formula_to_symbols(st.composition.reduced_formula)
        counts = get_counts(symbols)
        if len(st.symbol_set) >= 2 and counts >= 3:
            random_selection = {'link':   '+'.join(symbols),
                                'system': '-'.join(symbols),
                                'counts': count_format(counts)}
            break
    return random_selection

def pd_selector():
    binary = [['Fe','N'],
              ['Fe', 'S'],
              ['Fe', 'Si'],
              ['Co', 'N'],
              ['Zr', 'Co']]

    mm_ids = ['MMD-174',
              'MMD-765',
              'MMD-746',
              'MMD-134',
              'MMD-170']

    ternary = [['Fe', 'Co', 'N'],
               ['Fe', 'Co', 'S'],
               ['Zr', 'Co', 'B'],
               ['Zr', 'Co', 'C'],
               ['Zr', 'Co', 'N']]

    data = {'binary': [dict( system = '-'.join(symbols),
                             link   = '+'.join(symbols),
                             counts = count_format(get_counts(symbols)),
                             mm_id  = mm_id )
                       for symbols, mm_id in zip(binary, mm_ids)],
            'ternary': [dict( system = '-'.join(symbols),
                              link   = '+'.join(symbols),
                              counts = count_format(get_counts(symbols)))
                        for symbols in ternary]}

    return data

def search_method(status=None):
    if status is None:
        val = '.'
        url = None
    elif 'GA' in status:
        val = 'AGA search'
        url = '/about#AGA'
    else:
        val = status
        url = None
    return dict(val=val, url=url)

def references():
    # mongo.db.ref
    data = {'D0ME00050G':
            dict(
                author = "Balasubramanian, Balamurugan and Sakurai, Masahiro and Wang, Cai-Zhuang and Xu, Xiaoshan and Ho, Kai-Ming and Chelikowsky, James R. and Sellmyer, David J.",
                title = "Synergistic computational and experimental discovery of novel magnetic materials",
                journal = "Mol. Syst. Des. Eng.",
                year = "2020",
                volume = "5",
                issue = "6",
                pages = "1098-1117",
                publisher = "The Royal Society of Chemistry",
                doi = "10.1039/D0ME00050G",
                url = "http://dx.doi.org/10.1039/D0ME00050G"),
            'PhysRevMaterials.4.114408':
            dict(
                title = "Discovering rare-earth-free magnetic materials through the development of a database",
                author = "Sakurai, Masahiro and Wang, Renhai and Liao, Timothy and Zhang, Chao and Sun, Huaijun and Sun, Yang and Wang, Haidi and Zhao, Xin and Wang, Songyou and Balasubramanian, Balamurugan and Xu, Xiaoshan and Sellmyer, David J. and Antropov, Vladimir and Zhang, Jianhua and Wang, Cai-Zhuang and Ho, Kai-Ming and Chelikowsky, James R.",
                journal = "Phys. Rev. Materials",
                volume = "4",
                issue = "11",
                pages = "114408",
                numpages = "15",
                year = "2020",
                month = "Nov",
                publisher = "American Physical Society",
                doi = "10.1103/PhysRevMaterials.4.114408",
                url = "https://link.aps.org/doi/10.1103/PhysRevMaterials.4.114408")
        }
    return data

@app.route('/',methods=['GET'])
@cache.cached(timeout=300,key_prefix='index')
def index():
    # authors = {"original": "haidi", "revised": "MS"}
    num_default = 10
    total_count = get_total_count()

    num_entries = {'showing': count_format(num_default),
                   'total_found': 'a total of ' + count_format(total_count)}

    thumb_data = thumbnails_information(max_entries=-1)

    return make_response(render_template('index.html',
                                         num=num_entries,
                                         total_count='{:,}'.format(total_count),
                                         random_selection=random_selector(),
                                         pd_data=pd_selector(),
                                         data=thumb_data
    ))

@app.route('/query', methods=['GET'])
def query():
    message=None
    in_string = request.args.get("in_string")
    sel_type = request.args.get("select_type")
    assert sel_type in ['formula','reduced_formula','element','groupid','materialid','easy_axis']

    if sel_type=="formula":
       try:
          filter = {"data.system.reduced_formula": Composition(in_string.strip()).reduced_formula}
       except Exception as ex:
          message = str(ex)

    elif sel_type=="reduced_formula":
       try:
          filter = {"data.system.reduced_formula": in_string}
       except Exception as ex:
          message = str(ex)

    elif sel_type=="element":
       try:
          elements=[Element(i) for i in in_string.strip().split()]
          filter={}
          for element in elements:
              filter["composition."+element.symbol]={'$gt':0}
          filter["data.system.nelem"] = len(elements)
       except Exception as ex:
          message = str(ex)

    elif sel_type=='groupid':
       try:
          filter = {'data.spacegroup.number': int(in_string)}
       except Exception as ex:
          message = str(ex)

    elif sel_type=='easy_axis':
       try:
          filter = {'data.magnetic_anisotropy.easy_axis': in_string}
       except Exception as ex:
          message = str(ex)

    else:
       try:
          filter = {'entry_id': in_string}
       except Exception as ex:
          message = str(ex)
 
    if message is None:
        num_to_show = 50
        #num_found = mongo.db.data.find(filter=filter).count()
        num_found = mongo.db.data.count_documents(filter=filter)
        if num_found == 0:
            thumb_data = {}
            num_entries = {}
        else:
            thumb_data = thumbnails_information(max_entries=num_to_show,filter=filter)
            num_entries = {'showing': count_format(min(num_to_show, num_found)),
                           'total_found': count_format(num_found) + ' found'}
    else:
        thumb_data = {}
        num_entries = {}

    return make_response(render_template('index.html',
                                         message=message,
                                         data=thumb_data,
                                         num=num_entries,
                                         total_count='{:,}'.format(get_total_count()),
                                         random_selection=random_selector(),
                                         pd_data=pd_selector()
    ))

@app.route('/info/<mm_id>')
def show_info(mm_id):
    # entry
    entry = get_entry(mm_id)
    st = Structure.from_dict(entry['structure'])
    sst = get_symmetrized_structure_data(st)
    comp = st.composition
    symbols = formula_to_symbols(st.composition.reduced_formula)

    # material properties for this entry
    material_details = {'mm_id': mm_id,
                        'formula': {'html': html_formula(comp.reduced_formula),
                                    'unit': comp.get_reduced_formula_and_factor()[1]},
                        'num_sites': {'total': st.num_sites,
                                      'inequiv': len(sst['equivalent_indices'])},
                        'search': search_method( entry['data']['notes']['synthesis']['status'] )}


    # Data for similar structures (with the same elements)
    data_list = []
    for key in ['formula', 'elements']:
        if key == 'formula':
            data_list.append({'type': key,
                              'select_type': 'reduced_formula',
                              'formula': {'link':              comp.reduced_formula,
                                          'html': html_formula(comp.reduced_formula)},
                              'counts': count_format( mongo.db.data.find({"data.system.reduced_formula": comp.reduced_formula}).count() )})
        else:
            data_list.append({'type': key,
                              'select_type': 'element',
                              'formula': {'link': '+'.join(symbols),
                                          'html': '-'.join(symbols)},
                              'counts': count_format( get_counts(symbols) )})
    material_details['similar_structures'] = data_list

    # Data for similar structures (part 2)
    data_list = []
    subsys = ['elemental', 'binary', 'ternary']
    if len(symbols) >= 3:
        for el in reversed(symbols):
            sys = [x for x in symbols if x != el]
            data_list.append( {'system': subsys[len(symbols)-2].capitalize(),
                               'select_type': 'element',
                               'formula': {'link': '+'.join(sys),
                                           'html': '-'.join(sys)},
                               'counts': count_format( get_counts(sys) )} )
    material_details['related_systems'] = data_list

    # Space group: entry['data']['spacegroup']

    # Lattice parameters, volume, density, number of sites, formula units
    lattice = {'a':     fmt_d4%st.lattice.abc[0],
               'b':     fmt_d4%st.lattice.abc[1],
               'c':     fmt_d4%st.lattice.abc[2],
               'alpha': fmt_d3%st.lattice.angles[0],
               'beta':  fmt_d3%st.lattice.angles[1],
               'gamma': fmt_d3%st.lattice.angles[2],
               'volume': fmt_d3%st.volume,
               'density': fmt_d3%st.density}

    # Atomic positions and site-specific magnetic data
    material_details['onsite'] = [{'index': i+1,
                                   'specie': site.specie.value,
                                   'wp': sst['wyckoff_positions'][i],
                                   'x': fmt_d6%site.frac_coords[0],
                                   'y': fmt_d6%site.frac_coords[1],
                                   'z': fmt_d6%site.frac_coords[2],
                                   'lmm': fmt_d2%st.site_properties['lmm'][i],
                                   'soc_100': '.',
                                   'soc_001': '.'}
                                  for i, site in enumerate(st)]

    # pair-wise magnetic data: exchange coupling parameters, J_ij
    data_list = []
    dd = st.distance_matrix.tolist()
    for i, site_i in enumerate(st):
        for j, site_j in enumerate(st):
            if j > i and dd[i][j] > 0:
                data_list.append({'index_i': i+1,
                                  'index_j': j+1,
                                  'element_i': site_i.specie.value,
                                  'element_j': site_j.specie.value,
                                  'wp_i': sst['wyckoff_positions'][i],
                                  'wp_j': sst['wyckoff_positions'][j],
                                  'R_ij': fmt_d2%( dd[i][j] ),
                                  'J_ij': '.'})
    material_details['pairwise'] = data_list

    # Thermodynamic properties:
    # Formation energy per atom and above hull
    data_list = []
    for key, desc in zip(['decomposition','above_hull'],
                         ['Formation energy (vs. elemental phases)',
                          'Formation energy above hull']):
        data = {}
        data['description'] = desc
        #
        value = entry['data']['formation_energy'][key]['value']
        unit  = entry['data']['formation_energy'][key]['unit']
        if value is None:
            data['value'] = ''
        elif key == 'above_hull' and value < 0.001:
            data['value'] = '0 ' + unit + ', (stable)'
        else:
            if unit == 'meV':
                data['value'] = fmt_d1%( value ) + ' meV/atom'
            elif unit == 'eV':
                data['value'] = fmt_d1%( value * 10**3 ) + ' meV/atom'
        data_list.append(data)
    material_details['formation_energy'] = data_list

    # Phase diagram
    if st.ntypesp == 2:
        material_details['pd'] = {'plot': True,
                                  'text': None}
    elif st.ntypesp == 3:
        material_details['pd'] = {'plot': True,
                                  'text': 'The convex hull is temporarily not available while we perform maintenance.'}
    else:
        material_details['pd'] = {'plot': False,
                                  'text': 'Phase diagram is not available for this entry.'}

    # Magnetic properties:
    # magnetic ordering
    try:
        material_details['mag_ordering'] = entry['data']['magnetic_moment']['magnetic_ordering']
    except:
        material_details['mag_ordering'] = ''

    # total magnetic moment
    material_details['magmom'] = get_magmom(entry['data']['magnetic_moment']['total_magnetic_moment'], st.num_sites)

    # magnetic polarization
    material_details['magpol'] = get_magpol(entry['data']['magnetic_moment']['magnetic_polarization']['value'])

    # Curie temperature ("Tc")
    try:
        material_details['curie_T'] = fmt_d1%(entry['data']['critical_temperature']['Curie_temperature']['value']) + ' K'
    except:
        material_details['curie_T'] = ' '

    # Magnetic anisotropy constant
    if entry['data']['magnetic_moment']['total_magnetic_moment']['value'] < 0.01:
        material_details['mag_ordering'] = 'non-magnetic'
        material_details['magnetic_anisotropy'] = None
        material_details['curie_T'] = None
        material_details['J_ij'] = {'plot': False}
    else:
        data_list = []
        for key in ['a-c', 'b-c', 'b-a', 'd-a']:
            data={}
            if entry['data']['magnetic_anisotropy']['energy'][key] is not None:
                data['xy'] = key
                for kkey in ['energy', 'constant']:
                    value = entry['data']['magnetic_anisotropy'][kkey][key]
                    if value is None:
                        data[kkey] = ''
                    else:
                        data[kkey] = fmt_d2%( value )
                data_list.append(data)
        material_details['magnetic_anisotropy'] = {'dft': data_list}

        # Magnetic easy axis
        value = entry['data']['magnetic_anisotropy']['easy_axis']
        if value is None:
            material_details['magnetic_anisotropy']['easy_axis'] = ''
        else:
            material_details['magnetic_anisotropy']['easy_axis'] = value

        # Magnetic hardness parameter ("kappa")
        try:
            material_details['magnetic_anisotropy']['kappa'] = fmt_d2%(entry['data']['magnetic_anisotropy']['parameters']['kappa'])
        except:
            material_details['magnetic_anisotropy']['kappa'] = ''

        # J_ij = entry['data']['pairwise']['J_ij']
        if len(sst['wyckoff_symbols']) <= 20 and entry['data']['magnetic_moment']['magnetic_polarization']['value'] >= 0.5:
            material_details['J_ij'] = {'plot': True}
        else:
            material_details['J_ij'] = {'plot': False}

    # Methods
    data = entry['parameters']['DFT']
    kgrid = entry['data']['magnetic_anisotropy']['parameters']['kgrid']
    setup_dft = [
        {'key': 'exchange-correlation energy functional',
         'value': 'GGA-{}'.format(data['setup']['functional'])},
        {'key': 'pseudopotential type', 
         'value': data['setup']['pot_type']},
        {'key': 'cutoff energy',
         'value': '{ecutwfc} {unit}'.format(**data['cutoff'])},
        {'key': 'k-point grid',
         'value': '(kx, ky, kz) = ({kx}, {ky}, {kz}) for magnetic anisotropy calculations'.format(**kgrid)},
        {'key': 'package',         
         'value': '{} (v6.3)'.format(data['setup']['package']) }
    ]

    setup_lmto = [
        {'key': 'k-point grid',
         'value': '(kx, ky, kz) ='},
        {'key': 'energy mesh',
         'value': '41 points on an elliptical contour'},
        {'key': 'package',
         'value': 'Questaal (v7)'}
    ]

    material_details['method'] = {'DFT': setup_dft,
                                  'LMTO': setup_lmto}

    # References
    data_list = []
    if entry['data']['notes']['references']['doi']:
        data_list.append({'txt': entry['data']['notes']['references']['paper'] + '. DOI: ',
                          'lnk': entry['data']['notes']['references']['doi'],
                          'url': entry['data']['notes']['references']['url']})

    mpid = entry['data']['notes']['mpid']
    if mpid:
        data_list.append({'txt': 'Materials Project: ',
                          'lnk': mpid,
                          'url': 'https://materialsproject.org/materials/' + mpid + '/'})

    material_details['references'] = data_list

    return make_response(render_template('info.html',
                                         mat_detail=material_details,
                                         spg=entry['data']['spacegroup'],
                                         lat=lattice,
                                         tmp=[],
                                         dash_data=get_layer(st)
    ))

def find_entries(stable, symbols):
    elements = [Element(i) for i in symbols]
    filter = {"composition."+element.symbol: {'$gt':0}
              for element in elements}
    filter["data.system.nelem"] = len(elements)

    if stable == 'stable':
        filter["data.formation_energy.above_hull.value"] = {'$lt':0.001}
    elif stable == 'metastable':
        filter["data.formation_energy.above_hull.value"] = {'$gt':0}

    #total_count = mongo.db.data.find(filter).count()
    total_count = mongo.db.data.count_documents(filter)
    entries = mongo.db.data.find(filter)[0:total_count]
    return [entry['entry_id'] for entry in entries]

def get_formation_energy(mm_ids):
    data = []
    # get composition (atomic_fraction) and formation energy
    for mm_id in mm_ids:
        entry = get_entry(mm_id) 
        st = Structure.from_dict(entry['structure'])
        tmp = [st.composition.get_atomic_fraction(x) for x in st.composition.elements]
        value = entry['data']['formation_energy']['decomposition']['value']
        if entry['data']['formation_energy']['decomposition']['unit'] == 'eV':
            tmp.append(value*0.001)
        else:
            tmp.append(value)
        data.append(tmp)
    return data

@app.route('/info/<mm_id>/pd.png')
def plot_phase_diagram(mm_id):
    # get data
    st = get_structure(mm_id)

    # element symbols of this entry
    symbols = st.composition.elements
    chemsys = "-".join(sorted([str(el) for el in symbols]))

    if len(symbols) == 2:
        # binary plot
        fig = plt.figure()
        eloc = len(symbols)
        
        # horizontal line (E=0)
        plt.hlines(y=0, xmin=0, xmax=1, colors='gray', linestyle='dotted')

        # stable and metastable phases
        # [['ratio', 'energy'], ...] -> [['ratio', ...], ['energy', ...]]
        for stable in ['stable', 'metastable']:
            mm_ids = find_entries(stable, symbols)
            if stable == 'stable':
                edges = [[0,1,0], [1,0,0]]
                data = np.array(sorted( get_formation_energy(mm_ids) + edges )).T
                emin = min(data[eloc])

                # convex hull
                plt.plot(data[0], data[eloc], c='black', linestyle='dashed')

                # scatter plot (without mm_id)
                mm_ids = [x for x in mm_ids if x != mm_id]
                data = np.array(sorted( get_formation_energy(mm_ids) + edges )).T
                plt.scatter(data[0], data[eloc],
                            marker='o', color='blue', edgecolor='none', s=100,
                            label='{} phases ({})'.format(stable, count_format(len(mm_ids)+2)))
            else:
                mm_ids = [x for x in mm_ids if x != mm_id]
                data = np.array(sorted(get_formation_energy(mm_ids))).T
                emax = max(data[eloc])
                plt.scatter(data[0], data[eloc],
                            marker='s', facecolors='none', edgecolor='red', s=80,
                            label='{} phases ({})'.format(stable, count_format(len(mm_ids))))

                # this entry
                data = np.array(get_formation_energy([mm_id,])).T
                plt.scatter(data[0], data[eloc],
                            marker="D", facecolors='none', edgecolor='green', s=80,
                            label='{} (this entry)'.format(mm_id))

        # upper and lower bound
        dx = 0.02
        xmin, xmax = 0-dx, 1+dx
        plt.xlim(xmin, xmax)

        if emax <= 0:
            ymin = 100*(emin//100)
            ymax = 400
            lloc="upper left"
        elif emax > 300:
            ymin = 100*(emin//100 - 3)
            ymax = 300
            lloc="lower left"
        else:
            ymin = 100*(emin//100 - 3)
            ymax = min(100*(emax//100 + 1), 300)
            lloc="lower left"
        plt.ylim(ymin, ymax)

        # title, labels, font sizes
        fontsize = 16
        #plt.title("Phase diagram", fontsize=fontsize+4)
        plt.xlabel('{} fraction in {}'.format(symbols[0], chemsys), fontsize=fontsize)
        plt.ylabel("Formation energy (meV/atom)", fontsize=fontsize)
        plt.tick_params( labelsize=fontsize-4 )
        plt.legend(loc=lloc, fontsize=fontsize) # bbox_to_anchor=(1.05, 1), borderaxespad=0
        plt.tight_layout()

    elif len(symbols) == 3:
        # ternary plot
        # Boundary and Gridlines (ternary axes object)
        scale=1
        fig, tax = ternary.figure(scale=scale)

        # Draw Boundary and Gridlines
        tax.boundary(linewidth=1.5)
        tax.gridlines(color="gray", multiple=scale/10, linewidth=0.8, linestyle='dotted')

        # Set Axis, labels, and Title
        fontsize = 16
        offset = 0.15
        tax.set_title("Phase diagram", fontsize=fontsize+4, loc='left')

        tax.right_corner_label( symbols[0], fontsize=fontsize)
        tax.top_corner_label(   symbols[1], fontsize=fontsize)
        tax.left_corner_label(  symbols[2], fontsize=fontsize)

        tax.bottom_axis_label("{} fraction in {}".format(symbols[0], chemsys), fontsize=fontsize, offset=offset)
        tax.right_axis_label("{} fraction".format(symbols[1]),                 fontsize=fontsize, offset=offset)
        tax.left_axis_label("{} fraction".format(symbols[2]),                  fontsize=fontsize, offset=offset)

        # Set ticks
        tax.ticks(axis='lbr', multiple=scale/5, linewidth=1, offset=0.025, tick_formats='%.1f')
        tax.get_axes().axis('off')
        tax.clear_matplotlib_ticks()

        # hull
        #df = find_hull(symbols)
        #tax.line(df['x'], df['y'], linewidth=2, color='black')

        # Scatter plot for stable and metastable phases
        # [['ratio', 'energy'], ...] -> [['ratio', ...], ['energy', ...]]
        for stable, color in zip(['stable', 'metastable'], ['blue', 'red']):
            mm_ids = find_entries(stable,symbols)
            if stable == 'stable':
                edges = [[0,0,1,0], [0,1,0,0], [1,0,0,0]]
                data = np.array(sorted( get_formation_energy(mm_ids) + edges )).T
            else:
                data = np.array(sorted( get_formation_energy(mm_ids) )).T

        # this entry
        data = np.array(get_formation_energy([mm_id,])).T
        tax.scatter(get_formation_energy([mm_id,]),
                    marker='D', facecolors='none', edgecolor='green',
                    label='{} (this entry)'.format(mm_id))

        # show labels
        tax.legend()
        tax._redraw_labels()

    return generate_png(fig)

@app.route('/info/<mm_id>/lmm.png')
def plot_local_magnetic_moments(mm_id):
    # matplotlib.pyplot as plt
    fig = plt.figure()
    fontsize = 16

    # get data
    st = get_structure(mm_id)
    sst = get_symmetrized_structure_data(st)
    nat = st.num_sites

    df = pd.DataFrame({'index': range(1, nat+1),
                       'species': [x.symbol for x in st.species],
                       'lmm': st.site_properties["lmm"]})

    # ticks
    num_x = 10
    if nat <= num_x:
        step = 1
    else:
        step = int( nat / 5 )
        num_x = step*6
    plt.xticks( np.arange(1, num_x+0.1, step) )
    plt.yticks( np.arange(-1, 5, 1) )

    # ymin
    if df['lmm'].min() < -0.5:
        ymin = -1.0
    else:
        ymin = -0.5

    # ymax
    if df['lmm'].max() > 3.0:
        ymax = 4.0
    else:
        ymax = 3.0

    # xmin, xmax
    xs = (nat - 1) * 0.05
    xmin, xmax = 1 - xs, nat + xs

    # set boundary
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # horizontal lines (m=0, 1, 2)
    plt.hlines(y=0,                         xmin=xmin, xmax=xmax, colors='gray')
    plt.hlines(y=list(range(1, int(ymax))), xmin=xmin, xmax=xmax, colors='gray', linestyle='dotted')

    # plot local magnetic moments
    colors = ['green', 'red', 'blue', 'orange']
    for i, el in enumerate([str(el) for el in st.composition.elements]):
        ds = df.query('species == @el')
        plt.scatter(x=ds['index'],
                    y=ds['lmm'],
                    marker='o', facecolors='none', edgecolor=colors[i], zorder=3, s=80,
                    label='{}'.format(str(el))
        )

    # wyckoff positions to appear on the top
    for i in [i[0] for i in sst['equivalent_indices']]:
        plt.text(i+1, st.site_properties["lmm"][i]+0.2,
                 sst['wyckoff_positions'][i], 
                 ha='center', fontsize=fontsize-4)

    # title, labels
    #plt.title("Local magnetic moments", fontsize=fontsize+4)
    plt.xlabel("Site index", fontsize=fontsize)
    plt.ylabel("Local magnetic moment ($\mu_B$/atom)", fontsize=fontsize)
    plt.tick_params( labelsize=fontsize-4 )
    if 1.5 < st.site_properties['lmm'][0] < 4:
        plt.legend(loc="lower left", fontsize=fontsize)
    else:
        plt.legend(loc="upper left", fontsize=fontsize)

    plt.tight_layout()
    return generate_png(fig)

def get_J_ij(mm_id):
    # Star, nstar, |R|, direction, J(mRy), sum_J
    if mm_id == 'MMD-1':
        # Fe-Fe in bcc Fe (MMD-1)
        # entry = get_entry(mm_id)
        # data = np.array( entry['data']['pairwise']['J_ij'][0]['vs_R'] ).T
        data = np.array([[2, 8, 0.866025, 0.500, -0.500, -0.500, 1.07882, 8.631],
                         [3, 6, 1.000000, 0.000, 0.000, -1.000, 0.76965, 13.248],
                         [4, 12, 1.414214, 0.000, -1.000, -1.000, 0.08962, 14.324],
                         [5, 24, 1.658312, -1.500, -0.500, 0.500, -0.16622, 10.335],
                         [6, 8, 1.732051, -1.000, -1.000, 1.000, 0.00277, 10.357],
                         [7, 6, 2.000000, 0.000, 0.000, -2.000, -0.02929, 10.181],
                         [8, 24, 2.179449, 1.500, -1.500, 0.500, -0.03105, 9.436],
                         [9, 24, 2.236068, 2.000, -1.000, 0.000, 0.00730, 9.611],
                         [10, 24, 2.449490, -1.000, -1.000, -2.000, -0.07382, 7.839],
                         [11, 8, 2.598076, -1.500, 1.500, 1.500, 0.10794, 8.703],
                         [12, 24, 2.598076, -0.500, -0.500, 2.500, 0.01212, 8.994],
                         [13, 12, 2.828427, 0.000, -2.000, -2.000, -0.01835, 8.774],
                         [14, 48, 2.958040, 0.500, 1.500, 2.500, -0.01668, 7.973],
                         [15, 24, 3.000000, -1.000, 2.000, -2.000, 0.01017, 8.217],
                         [16, 6, 3.000000, 0.000, 0.000, 3.000, -0.02713, 8.054],
                         [17, 24, 3.162278, 3.000, 0.000, -1.000, -0.03053, 7.322],
                         [18, 24, 3.278719, 1.500, -2.500, -1.500, 0.05623, 8.671],
                         [19, 24, 3.316625, -1.000, -1.000, 3.000, -0.00426, 8.569],
                         [20, 8, 3.464102, 2.000, -2.000, -2.000, -0.16160, 7.276],
                         [21, 24, 3.570714, -0.500, -3.500, -0.500, -0.00345, 7.193],
                         [22, 24, 3.570714, 0.500, -2.500, -2.500, 0.00175, 7.235],
                         [23, 24, 3.605551, 0.000, 3.000, -2.000, -0.00264, 7.172],
                         [24, 48, 3.741657, 3.000, 1.000, 2.000, -0.00074, 7.137],
                         [25, 24, 3.840573, -2.500, -1.500, 2.500, 0.04824, 8.294],
                         [26, 48, 3.840573, -1.500, 0.500, 3.500, 0.01769, 9.144],
                         [27, 6, 4.000000, 0.000, -4.000, 0.000, -0.01751, 9.039]]).T
    elif mm_id == 'MMD-2':
        # Co-Co in hcp Co (MMD-2)
        data = [[2, 3, 1.000000, 0.500, -0.866, 0.000, 1.10039, 3.301],
                [3, 2, 1.000000, -1.000, 0.000, 0.000, 0.91699, 5.135],
                [4, 2, 1.622000, 0.000, 0.000, 1.622, 0.25451, 5.644],
                [5, 1, 1.732050, 0.000, -1.732, 0.000, 0.11645, 5.761],
                [6, 1, 1.732050, 0.000, 1.732, 0.000, 0.11645, 5.877],
                [7, 2, 1.732051, 1.500, -0.866, 0.000, 0.13974, 6.157],
                [8, 2, 1.732051, -1.500, 0.866, 0.000, 0.13974, 6.436],
                [9, 5, 1.905488, -0.500, -0.866, 1.622, -0.03067, 6.283],
                [10, 4, 1.905488, -1.000, 0.000, 1.622, -0.03067, 6.160],
                [11, 2, 1.999999, -1.000, -1.732, 0.000, -0.08057, 5.999],
                [12, 2, 2.000000, -2.000, 0.000, 0.000, -0.08057, 5.838],
                [13, 2, 2.372948, 0.000, -1.732, 1.622, 0.01007, 5.858],
                [14, 2, 2.372948, 0.000, 1.732, 1.622, 0.01007, 5.878],
                [15, 4, 2.372948, 1.500, -0.866, 1.622, 0.00805, 5.910],
                [16, 4, 2.372948, -1.500, 0.866, 1.622, 0.00805, 5.942],
                [17, 4, 2.575050, -1.000, -1.732, 1.622, 0.00834, 5.976],
                [18, 4, 2.575050, -2.000, 0.000, 1.622, 0.00834, 6.009],
                [19, 2, 2.645750, -0.500, -2.598, 0.000, -0.03211, 5.945],
                [20, 2, 2.645750, -0.500, 2.598, 0.000, -0.03211, 5.881],
                [21, 2, 2.645751, -2.000, -1.732, 0.000, -0.03211, 5.817],
                [22, 2, 2.645751, -2.000, 1.732, 0.000, -0.03211, 5.752],
                [23, 2, 2.645751, 2.500, 0.866, 0.000, -0.03211, 5.688],
                [24, 2, 2.645751, -2.500, -0.866, 0.000, -0.03211, 5.624],
                [25, 2, 2.999999, 1.500, -2.598, 0.000, 0.02416, 5.672],
                [26, 2, 3.000000, -3.000, 0.000, 0.000, 0.02416, 5.721],
                [27, 4, 3.103366, 0.500, -2.598, 1.622, 0.00107, 5.725]]
        data += [[1, 2, 0.995517, 0.000, 0.577, 0.811, 1.05863, 2.117],
                 [2, 4, 0.995517, 0.500, -0.289, 0.811, 0.84690, 5.505],
                 [3, 2, 1.411047, 0.000, -1.155, 0.811, 0.13637, 5.778],
                 [4, 4, 1.411047, 1.000, 0.577, -0.811, 0.13637, 6.323],
                 [5, 4, 1.729466, 0.500, 1.443, -0.811, 0.09739, 6.713],
                 [6, 4, 1.729466, -1.000, -1.155, 0.811, 0.09739, 7.102],
                 [7, 4, 1.729466, -1.500, -0.289, -0.811, 0.09739, 7.492],
                 [8, 4, 2.234066, 0.500, -2.021, -0.811, 0.00667, 7.518],
                 [9, 4, 2.234066, -1.500, 1.443, 0.811, 0.00667, 7.545],
                 [10, 4, 2.234067, -2.000, 0.577, 0.811, 0.00667, 7.572],
                 [11, 2, 2.447662, 0.000, 2.309, -0.811, 0.02183, 7.616],
                 [12, 4, 2.447663, -2.000, -1.155, -0.811, 0.02183, 7.703],
                 [13, 2, 2.500564, 0.000, 0.577, 2.433, 0.01138, 7.726],
                 [14, 4, 2.500564, 0.500, -0.289, -2.433, 0.00911, 7.762],
                 [15, 4, 2.644059, -1.000, 2.309, -0.811, -0.02169, 7.675],
                 [16, 4, 2.644059, 1.500, -2.021, 0.811, -0.02169, 7.589],
                 [17, 4, 2.644060, -2.500, -0.289, 0.811, -0.02169, 7.502],
                 [18, 2, 2.693106, 0.000, -1.155, -2.433, -0.03561, 7.431],
                 [19, 4, 2.693106, 1.000, 0.577, -2.433, -0.03561, 7.288],
                 [20, 4, 2.872772, -0.500, 1.443, 2.433, -0.01091, 7.244],
                 [21, 4, 2.872772, -1.000, -1.155, -2.433, -0.01091, 7.201],
                 [22, 4, 2.872773, 1.500, -0.289, -2.433, -0.01091, 7.157],
                 [23, 2, 2.998507, 0.000, -2.887, -0.811, 0.00600, 7.169],
                 [24, 4, 2.998508, 2.500, 1.443, -0.811, 0.00600, 7.193],
                 [25, 4, 3.160862, 1.000, -2.887, -0.811, -0.00759, 7.163]]
        data = np.array(data).T
    return data

@app.route('/info/<mm_id>/J_ij.png')
def plot_exchange_coupling_parameters_vs_d(mm_id):
    # matplotlib.pyplot as plt
    fig = plt.figure()

    # upper and lower bound
    xmin, xmax =  0, 10
    ymin, ymax = -5, 20
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # horizontal lines (J=0)
    plt.hlines(y=0, xmin=xmin, xmax=xmax, colors='gray')

    # scatter plot
    data = get_J_ij('MMD-1')
    alat = 2.8694
    plt.scatter(data[2]*alat, data[6]*Ry,
                marker='D', facecolors='none', edgecolor='blue', s=70,
                label='Fe-Fe (bcc Fe)')

    data = get_J_ij('MMD-2')
    alat = 2.5008
    plt.scatter(data[2]*alat, data[6]*Ry,
                marker='s', facecolors='none', edgecolor='orange', s=70,
                label='Co-Co (hcp Co)')

    # title, labels, font sizes
    fontsize = 16
    #plt.title("Magnetic exchange parameters", fontsize=fontsize+2)
    plt.xlabel("Distance between site $i$ and $j$ ($\AA$)", fontsize=fontsize)
    plt.ylabel("$J_{ij}$ (meV)", fontsize=fontsize)
    plt.tick_params( labelsize=fontsize-2 )
    plt.legend(loc="upper right", fontsize=fontsize)
    plt.tight_layout()

    return generate_png(fig)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # set font size
    fontsize = 16

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fontsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False,
                   labelsize=fontsize-data.shape[1]/4)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", va="bottom",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

@app.route('/info/<mm_id>/jmap.png')
def plot_exchange_coupling_parameters_heatmap(mm_id):
    # read data
    st = get_structure(mm_id)
    sst = get_symmetrized_structure_data(st)

    idx_irr = [ii[0] for ii in sst['equivalent_indices']]
    sites = [sst['wyckoff_sites'][i] for i in idx_irr]

    # random numbers
    values = np.array([ [random.uniform(-4, 20) for i in range(len(sites))]
                        for j in range(len(sites)) ])
    values = np.tril(values) + np.tril(values).T - np.diag(values.diagonal())

    # data
    data = {"key_x": sites,
            "key_y": sites,
            "list_val": values}

    # heatmap plot
    fig, ax = plt.subplots()
    im, cbar = heatmap(data["list_val"], 
                       data["key_y"], 
                       data["key_x"], 
                       ax=ax,
                       cmap="YlGn",
                       cbarlabel="$J_{ij}$ (meV)")

    # format
    if len(sites) <= 4:
        texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=16)
    elif len(sites) <= 8:
        texts = annotate_heatmap(im, valfmt="{x:.0f}", fontsize=14)
    else:
        texts = []

    #ax.set_title("Magnetic exchange parameters (virtual)", fontsize=16)
    fig.tight_layout()
    return generate_png(fig)

@app.route('/diagram')
def diagram():
    return make_response(render_template('diagram.html',
                                         pd_data=pd_selector()
    ))

@app.route('/about')
def about():
    return make_response(render_template('about.html'))

## clusters
#def get_cluster(mm_id):
#    return mongo.db.cluster.find_one({'entry_id':mm_id})
#
#@app.route('/cluster')
#def cluster():
#    num_default = 10
#    #total_count = mongo.db.cluster.find().count()
#    total_count = mongo.db.cluster.count_documents()
#
#    num_entries = {'showing': '{:,}'.format(num_default) + ' entries',
#                   'total_found': 'a total of ' + '{:,}'.format(total_count) + ' entries'}
#
#    # thumb_data
#    data = []
#    entries = mongo.db.cluster.find()[0:num_default]
#    for entry in entries:
#        # get structure data for this entry
#        mol = Molecule.from_dict(entry['structure'])
#        symbols = [str(el) for el in Counter(mol.species).keys()]
#        radial_coords = [np.inner(r, r) for r in np.array(mol.cart_coords)]
#
#        # ID, Formula, etc
#        tmp = {'mmid': entry['entry_id'],
#               'formula': {'link': '+'.join(symbols),
#                           'full':    html_formula(mol.composition.formula.replace(' ', '')).replace('<sub>1</sub>', ''),
#                           'reduced': html_formula(mol.composition.reduced_formula)},
#               'num_sites': mol.num_sites,
#               'diameter': fmt_d1%( 2.0 * math.sqrt( max(radial_coords) ) * abohr ),
#               'method': {'text': 'real-space DFT',
#                          'url': 'http://real-space.org'}}
#
#        # Magnetic moment
#        magmom = entry['data']['total_magnetic_moment']['value']
#        tmp['magmom'] = {'ave': fmt_d2%( magmom / mol.num_sites )}
#
#        # References
#        if entry['data']['notes']['references']['doi']:
#            tmp['ref'] = {'text': entry['data']['notes']['references']['paper'] + '.',
#                          'url':  entry['data']['notes']['references']['url']}
#        else:
#            tmp['ref'] = {'text': '',
#                          'url':  '.'}
#
#        data.append(tmp)
#    
#    return make_response(render_template('cluster.html',
#                                         num=num_entries,
#                                         data=data,
#                                         tmp={}))
#
#@app.route('/cluster/<mm_id>')
#def cluster_info(mm_id):
#    # entry
#    entry = get_cluster(mm_id)
#    mol = Molecule.from_dict(entry['structure'])
#
#    radial_coords = [np.inner(r, r) for r in np.array(mol.cart_coords)]
#
#    # material properties for this entry
#    cluster_data = {'mm_id': mm_id,
#                    'formula': {'full':    html_formula(mol.composition.formula.replace(' ', '')).replace('<sub>1</sub>', ''),
#                                'reduced': html_formula(mol.composition.reduced_formula)},
#                    'num_sites': mol.num_sites,
#                    'diameter': fmt_d1%( 2.0 * math.sqrt( max(radial_coords) ) * abohr ),}
#
#    # Magnetic properties
#    value = entry['data']['total_magnetic_moment']['value']
#    cluster_data['magmom'] = {'tot': fmt_d2%( value ),
#                              'ave': fmt_d2%( value / mol.num_sites )}
#
#    # Atomic positions and site-specific magnetic data
#    cluster_data['onsite'] = [{'index': i+1,
#                               'specie': site.specie.value,
#                               'r': fmt_d4%( math.sqrt(radial_coords[i]) * abohr ),
#                               'x': fmt_d6%( site.coords[0] * abohr ),
#                               'y': fmt_d6%( site.coords[1] * abohr ),
#                               'z': fmt_d6%( site.coords[2] * abohr ),
#                               'm': fmt_d2%( mol.site_properties['lmm'][i] )}
#                              for i, site in enumerate(mol)]
#
#    # Thermodynamic properties:
#    # Binding energy per atom
#
#    # References
#    key = 'references'
#    if entry['data']['notes'][key]['doi']:
#        cluster_data[key] = entry['data']['notes'][key]
#    else:
#        cluster_data[key] = []
#
#    return make_response(render_template('cluster_info.html',
#                                         cluster_data=cluster_data,
#                                         tmp=[]
#    ))
#
#@app.route('/cluster/<mm_id>/lmm.png')
#def plot_cluster_local_magnetic_moments(mm_id):
#    # matplotlib.pyplot as plt
#    fig = plt.figure()
#
#    # data
#    entry = get_cluster(mm_id)
#    mol = Molecule.from_dict(entry['structure'])
#
#    # Cluster size (diameter)
#    radial_coords = [np.inner(r, r) for r in np.array(mol.cart_coords)]
#    rmax = math.sqrt( max(radial_coords) ) * abohr
#
#    # ticks
#    if rmax < 5:
#        step = 1
#        rmax = 5
#    else:
#        step = int( rmax / 5 )
#        rmax = step*6
#    plt.xticks( np.arange(0, rmax+1, step) )
#    plt.yticks( np.arange(-1, 5, 1) )
#
#    # upper bound
#    xs = rmax * 0.05
#    xmin, xmax =  0-xs, rmax+xs
#    ymin, ymax = -0.5, 4.0
#    plt.xlim(xmin, xmax)
#    plt.ylim(ymin, ymax)
#
#    # horizontal lines (m=0, 1, 2, 3)
#    plt.hlines(y=0,         xmin=xmin, xmax=xmax, colors='gray')
#    plt.hlines(y=[1, 2, 3], xmin=xmin, xmax=xmax, colors='gray', linestyle='dotted')
#
#    # plot local magnetic moments
#    for el in mol.composition.elements:
#        plt.scatter([math.sqrt(radial_coords[i])*abohr for i, site in enumerate(mol) if site.specie.value == str(el)],
#                    [mol.site_properties['lmm'][i]     for i, site in enumerate(mol) if site.specie.value == str(el)],
#                    marker="D", s=70, alpha=0.7, edgecolor="white", label=str(el))
#
#    # DataFrame
#    df = pd.DataFrame( [(math.sqrt(radial_coords[i])*abohr,
#                         mol.site_properties['lmm'][i],
#                         site.specie.value)
#                        for i, site in enumerate(mol)],
#                       columns = ['r', 'm', 'el'] )
#
#    # title, labels, font sizes
#    fontsize = 16
#    plt.xlabel("Radial position ($\AA$)", fontsize=fontsize)
#    plt.ylabel("Local magnetic moment ($\mu_B$/atom)", fontsize=fontsize)
#    plt.tick_params( labelsize=fontsize-4 )
#    plt.legend(loc="upper left", fontsize=fontsize) # bbox_to_anchor=(1.05, 1)
#    plt.tight_layout()
#
#    return generate_png(fig)
#
# stats
@app.route('/stats')
def stats():
    stats = get_stats()
    total_count = get_total_count()

    crystal_systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
    compositions = ["elemental", "binary", "ternary", "quaternary"]
    base_elements = ['Mn','Fe','Co','Ni']

    stats_data = {}

    # Crystal_systems
    stats_data['crystal_systems'] = [dict(system  = key.capitalize(),
                                          counts  = '{:,}'.format(   int(stats['crystal_systems'][key])              ),
                                          percent = '{:.0%}'.format(     stats['crystal_systems'][key] / total_count ))
                                     for key in crystal_systems]

    # Compositions
    stats_data['compositions'] = [dict(system = key.capitalize(),
                                       types  = '{:,}'.format( int(stats['structures']['compound_types'][key]) ),
                                       total  = '{:,}'.format( int(stats['structures']['compound_total'][key]) ))
                                  for key in compositions]

    # Compounds with ...
    stats_data['compounds_with'] = [dict(compounds = 'Compounds containing {}'.format(key),
                                         counts    = '{:,}'.format( mongo.db.data.count_documents( {"composition."+Element(key).symbol: {'$gt':0},
                                                                                         "data.system.nelem":                {'$gt':1} } ) ))
                                    for key in base_elements]

    # Formation energy, Magnetic polarization
    keys = ['Formation energy', 'Magnetic polarization']
    data_list = [
        {'title': 'Formation energy',
         'phys': 'ehull' },
        {'title': 'Magnetic polarization',
         'phys': 'js' },
    ]

    return make_response(render_template('stats.html',
                                         total_count='{:,}'.format(total_count),
                                         stats_data=stats_data,
                                         data_list=data_list
    ))

def stats_data_df(name):
    if name == 'ehull':
        data = [[0.00,860,10],
                [10.00,80,19],
                [20.00,66,20],
                [30.00,38,41],
                [40.00,41,94],
                [50.00,36,61],
                [60.00,36,160],
                [70.00,30,114],
                [80.00,33,63],
                [90.00,24,44],
                [100.00,18,26],
                [110.00,15,14],
                [120.00,19,13],
                [130.00,21,14],
                [140.00,10,11],
                [150.00,13,9],
                [160.00,12,16],
                [170.00,13,5],
                [180.00,7,21],
                [190.00,7,21],
                [200.00,13,8],
                [210.00,11,6],
                [220.00,10,4],
                [230.00,11,0],
                [240.00,11,5],
                [250.00,3,8],
                [260.00,5,1],
                [270.00,9,0],
                [280.00,4,0],
                [290.00,5,0],
                [300.00,4,0],
                [310.00,3,0],
                [320.00,4,0],
                [330.00,6,0],
                [340.00,0,0],
                [350.00,2,2],
                [360.00,2,0],
                [370.00,5,0],
                [380.00,4,0],
                [390.00,5,0],
                [400.00,4,0],
                [410.00,4,0],
                [420.00,3,0],
                [430.00,2,0],
                [440.00,8,0],
                [450.00,6,0],
                [460.00,5,0],
                [470.00,17,0],
                [480.00,6,0],
                [490.00,3,0]]
    elif name == 'js':
        data = [[0.00,824,42],
                [0.10,100,0],
                [0.20,85,3],
                [0.30,52,2],
                [0.40,63,4],
                [0.50,57,5],
                [0.60,43,20],
                [0.70,49,24],
                [0.80,43,40],
                [0.90,44,50],
                [1.00,41,66],
                [1.10,23,78],
                [1.20,41,61],
                [1.30,33,41],
                [1.40,20,70],
                [1.50,19,46],
                [1.60,20,51],
                [1.70,21,50],
                [1.80,12,39],
                [1.90,12,26],
                [2.00,7,50],
                [2.10,4,42],
                [2.20,3,9],
                [2.30,9,0],
                [2.40,0,0],
                [2.50,1,0],
                [2.60,0,0],
                [2.70,0,0],
                [2.80,0,0],
                [2.90,0,0]]
    return pd.DataFrame(data, columns=['x', 'MP', 'AGA'])

@app.route('/stats/<name>.png')
def plot_stats(name):
    # matplotlib.pyplot as plt
    fig = plt.figure()
    fontsize = 16

    # data
    stats = get_stats()

    # keys
    crystal_systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
    compositions = ["binary", "elemental", "ternary", "quaternary"]
    base_elements = ['Mn','Fe','Co','Ni']

    if name == 'crystal_systems':
        # pie plot
        plt.rcParams['font.size'] = fontsize+4
        plt.pie([int(stats['crystal_systems'][key]) for key in crystal_systems],
                labels=crystal_systems,
                autopct="%.0f%%", pctdistance=0.75, startangle=45, counterclock=False)

    elif name == 'compositions':
        # pie plot
        plt.rcParams['font.size'] = fontsize+4
        plt.pie([int(stats['structures']['compound_total'][key]) for key in compositions],
                labels=compositions,
                autopct="%.0f%%", pctdistance=0.75, startangle=90, counterclock=False)

    elif name == 'compounds':
        labels = ['{}-based'.format(x) for x in base_elements]
        values = {system: [mongo.db.data.count_documents( {"composition."+Element(key).symbol: {'$gt':0},
                                                "data.system.nelem": i                        } )
                           for key in base_elements]
                  for i, system in enumerate(['binary', 'ternary', 'quaternary'], 2)}
        # bar plot
        plt.bar(labels, values['binary'], align="center", color = "c",
                label="binary")
        plt.bar(labels, values['ternary'], align="center", color = "b",
                label="ternary",
                bottom=values['binary'])
        plt.bar(labels, values['quaternary'], align="center", color = "m",
                label="quaternary",
                bottom=[x+y for x, y in zip(values['binary'], values['ternary'])])
        # labels
        plt.ylabel("Number of entries", fontsize=fontsize+4)
        plt.tick_params(labelsize=fontsize)
        plt.legend(loc="upper right", fontsize=fontsize-2)


    elif name == 'ehull' or name == 'js':
        if name == 'ehull':
            # upper and lower bound
            xmin, xmax =  0, 300
            plt.xlim(xmin, xmax)
            plt.xticks( np.arange(xmin, xmax*1.01, 50) )
            plt.xlabel("Formation energy above hull (meV/atom)", fontsize=fontsize)
            width = 10

        elif name == 'js':
            # upper and lower bound
            xmin, xmax =  0, 2.5
            plt.xlim(xmin, xmax)
            plt.xticks( np.arange(xmin, xmax*1.01, 0.5) )
            plt.xlabel("Magnetic polarization (T)", fontsize=fontsize)
            width = 0.1

        # upper and lower bound
        ymin, ymax =  0, 0.6
        plt.ylim(ymin, ymax)
        plt.yticks( np.arange(ymin, ymax*1.01, 0.1) )
        plt.ylabel("Count (normalized)", fontsize=fontsize)

        # plot
        df = stats_data_df(name)
        label = ['Materials project', 'AGA search']
        hatch = ['///', '\\\\\\']
        color = ['red', 'blue']
        for i, key in enumerate(['MP', 'AGA']):
            df['y'] = df[key] / df[key].sum()
            plt.bar(x=df['x'], height=df['y'], width=width,
                    label=label[i], hatch=hatch[i], facecolor='none', edgecolor=color[i])

        # title, labels, font sizes
        plt.tick_params( labelsize=fontsize-2 )
        plt.legend(loc="upper right", fontsize=fontsize)

    plt.tight_layout()
    return generate_png(fig)

# benchmark
def benchmark_data(key):
    columns = ['material', 'mm_id', 'experiment', 'theory'] # formula_html
    data_list = {'js': [['bcc Fe', 'MMD-1',    2.15, 2.18],
                        ['hcp Co', 'MMD-2',    1.81, 1.72],
                        ['YCo5',   'MMD-165',  1.06, 1.01],
                        ['MnAl',   'MMD-1014', 0.75, 1.02],
                        ['FeNi',   'MMD-1199', 1.59, 1.69],
                        ['Fe16N2', 'MMD-1229', 2.10, 2.21]],
                 'k1': [['hcp Co', 'MMD-2',    0.41, 0.22],
                        ['YCo5',   'MMD-165',  6.05, 2.68],
                        ['MnAl',   'MMD-1014', 1.70, 1.53],
                        ['FeNi',   'MMD-1199', 0.70, 0.30],
                        ['Fe16N2', 'MMD-1229', 1.00, 0.72],
                        ['CoPt',   'MMD-1446', 4.90, 4.34]],
                 'tc': [['bcc Fe', 'MMD-1',    1043,  953.2],
                        ['hcp Co', 'MMD-2',    1388, 1474.2],
                        ['fcc Ni', 'MMD-3',     627,  401.8],
                        ['YCo5',   'MMD-165',   987, 1155.9],
                        ['MnAl',   'MMD-1014',  650,  521.8],
                        ['FeNi',   'MMD-1199',  820,  953.3],
                        ['CoPt',   'MMD-1446',  840,  684.5]]}
    return pd.DataFrame(data_list[key], columns=columns)

@app.route('/benchmark')
def benchmark():
    keys = ['title', 'table', 'phys', 'unit']
    lists = [['Magnetic polarization',        benchmark_data('js').to_dict(orient='records'), 'js', 'T'],
             ['Magnetic anisotropy constant', benchmark_data('k1').to_dict(orient='records'), 'k1', 'MJ/m<sup>3</sup>'],
             ['Curie temperature',            benchmark_data('tc').to_dict(orient='records'), 'tc', 'K']]
    return make_response(render_template('benchmark.html',
                                         data_list = [dict(zip(keys, values)) for values in lists]
    ))

def func_linear_model(X, a):
    return a*X

@app.route('/benchmark/<name>.png')
def plot_benchmark(name):
    # matplotlib.pyplot as plt
    fig = plt.figure()

    # upper and lower bound, unit
    if name == 'js':
        xmin, xmax = 0, 2.5
        xd = 0.5
        unit = 'T'
    elif name == 'k1':
        xmin, xmax = 0, 8
        xd = 2
        unit = 'MJ/m3'
    else:
        xmin, xmax = 0, 1600
        xd = 400
        unit = 'K'

    ymin, ymax = xmin, xmax
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # title, labels, font sizes
    fontsize = 16
    plt.xlabel("Experimental value ({})".format(unit), fontsize=fontsize)
    plt.ylabel("Theoretical value ({})".format(unit),  fontsize=fontsize)

    # ticks
    xt = np.arange(xmin, xmax*1.01, xd)
    plt.xticks(xt)
    plt.yticks(xt)
    plt.tick_params( labelsize=fontsize-3 )

    # aspect ratio
    plt.gca().set_aspect('equal')

    # y=x
    xx = np.linspace(xmin, xmax, 10)
    plt.plot(xx, xx, '-', color='k')

    # fill (error)
    dk = 0.1
    ym, yp = xx * (1-dk), xx * (1+dk)
    plt.fill_between(xx, ym, yp, facecolor='lightgray', alpha=0.7,
                     label='{:.0%} error'.format(dk))
    dk = 0.2
    ym, yp = xx * (1-dk), xx * (1+dk)
    plt.plot(xx, ym, ':', color='gray', label='{:.0%} error'.format(dk))
    plt.plot(xx, yp, ':', color='gray')

    # scatter plot
    df = benchmark_data(name)
    plt.scatter(x=df['experiment'],
                y=df['theory'],
                marker='o', facecolors='none', edgecolor='blue', zorder=3, s=100,
                label='Test set')

    # R2 score
    plt.text((xmax-xmin)*0.05,
             (ymax-ymin)*0.5,
             '$R^2={:.2f}$'.format(r2_score(df['experiment'], df['theory'])),
             fontsize=fontsize)

    # Create linear regression object
    #regr = linear_model.LinearRegression()
    # Train the model using the training sets
    #regr.fit(X_train, y_train)

    # linear fit
    popt, pcov = curve_fit(func_linear_model, df['experiment'], df['theory'], p0=[1,])

    # Make predictions using the testing set
    plt.plot(xx, # regr.predict(xx),
             [func_linear_model(x, popt[0]) for x in xx],
             '--', color='blue', label='Linear fit')

    # legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,1,2,0]
    plt.legend([handles[idx] for idx in order],
               [labels[idx] for idx in order],
               loc='upper left', fontsize=fontsize, ncol=1)

    plt.tight_layout()
    return generate_png(fig)

# download
@app.route('/download',methods=['GET'])
def download():
    thumb_data = thumbnails_information(max_entries=-1)
    return str(pprint.pformat(thumb_data, indent=4))

@app.route('/download/<mm_id>/<fmt>')
def download_file(mm_id,fmt):
    save_file(mm_id, fmt)
    file_dir = os.path.join(basedir, app.config['DATA_FOLDER'])
    file_name = mm_id+'.'+fmt
    if os.path.isfile(os.path.join(file_dir, file_name)):
       return send_from_directory(file_dir, file_name, as_attachment=True)
    abort(404)
 
# upload file
@app.route('/test/upload')
def upload_test():
    return render_template('upload.html')

@app.route('/api/upload',methods=['POST'],strict_slashes=False)
def api_upload():
    file_dir=os.path.join(basedir,app.config['UPLOAD_FOLDER'])
    check_folder(file_dir)
    f=request.files['myfile']  # 从表单的file字段获取文件，myfile为该表单的name值
    print(f)
    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname=secure_filename(f.filename)
        ext = fname.rsplit('.',1)[1]  # 获取文件后缀
        unix_time = int(time.time())
        new_filename=str(unix_time)+'.'+ext  # 修改了上传的文件名
        f.save(os.path.join(file_dir,new_filename))  #保存文件到upload目录
        #token = base64.b64encode(new_filename)
        #print(new_filename)
        return jsonify({"errno":0,"errmsg":"success","token":new_filename})
    else:
        return jsonify({"errno":1001,"errmsg":"failed"})
