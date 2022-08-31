import os,shutil
from flask import render_template, make_response, request, Flask
from flask_restful import Api
from flask_pymongo import PyMongo
from bson.json_util import dumps

basedir = os.path.abspath(os.path.dirname(__file__))


#try:
#   from flask_cache import Cache
#except:
#srcfile=os.path.join(basedir,"jinja2ext.py")
#distfile="/app/.heroku/python/lib/python3.10/site-packages/flask_caching/jinja2ext.py"
#distfile="/home/vasp/.pyenv/versions/mpt_a3/lib/python3.6/site-packages/flask_cache/jinja2ext.py"
#print(srcfile)
#print(distfile)
#shutil.copyfile(srcfile,distfile)    
#from flask_cache import Cache
from flask_caching import Cache
#from werkzeug.utils import import_string

MONGO_URL = os.environ.get('MONGODB_URI')
MAPI_KEY = os.environ.get('MAPI_KEY')  # obtained from materials project
if not MONGO_URL:
    MONGO_URL = "mongodb://localhost:27017/restfulapi";


UPLOAD_FOLDER='upload'
#CACHE_TYPE = 'simple'
CACHE_TYPE = 'SimpleCache'
DATA_FOLDER = 'data'
print(basedir)
ALLOWED_EXTENSIONS = set(['json','yaml','vasp','cif','lammps'])


app = Flask(__name__)
cache = Cache(app,config={'CACHE_TYPE': CACHE_TYPE})
cache.init_app(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['MONGO_URI'] = MONGO_URL
mongo = PyMongo(app)
with open('log.txt','w+') as f:
  f.write(str(mongo)+'\n')
  f.write(str(mongo.cx)+'\n')
  f.write(str(mongo.db)+'\n')

def output_json(obj, code, headers=None):
    resp = make_response(dumps(obj), code)
    resp.headers.extend(headers or {})
    return resp

DEFAULT_REPRESENTATIONS = {'application/json': output_json}
api = Api(app)
#api = restful.Api(app)
api.representations = DEFAULT_REPRESENTATIONS

import flask_rest_service.resources
#import flask_rest_service.crystal_toolkit
