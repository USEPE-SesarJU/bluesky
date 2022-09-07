""" This plugin is written for BlueSky ATM Simulator as a machine learning efforts in USEPE projects"""
""" The version 3 has tactical scoring system for better separation. """
""" Main Changes
    - ON Command is removed 
    - BASIC command is added
    - TACTICAL command is added
    - DEBUG command is added
""" 
""" Some bugs are fixed in version 3."""
""" - Aircraft pair indexes starts from 1 for all.
    - White background added to the summaries on the graphs for better visualization.
    - Pairwise strategic analysis plot is fixed.
    - Comflict duration error is fixed.
    - Single Plots
        - Number of conflicts are added to local and general
        - Number of LoS are added to local and general
        - Single plot sumary detailes are updated
    - Single CSV files
        - Number of conflicts are added single csv for local and general
        - Number of LoS are added single csv for local and general
    - All summary boxes is made transparent for better visual
    - #simt is fixed from log files (removed)
"""
""" The version 2 manages the necessary data logging, separation analysis, outputting results and visualition of the experimental setup."""

__authors__  = "Serkan Güldal, Rina Komatsu"
__emails__   = "SrknGldl@hotmail.com, rinakomatsu2021@gmail.com"
__version__  = "3.0"
__status__   = "Production"
__date__     = "08/08/2022"

# Import the global bluesky objects. Uncomment the ones you need

# Create LOG ----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# List of the names of all the data loggers
import sys
import glob
from enum import unique
from tracemalloc import start
import seaborn as sns
import sklearn.metrics as metrics
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import itertools
import os
import bluesky as bs
from bluesky import settings
from bluesky.tools.datalog import allloggers
from bluesky.tools import datalog
from turtle import width
from prometheus_client import Summary
from bluesky import core, traf, stack  # , settings, navdb, sim, scr, tools
from bluesky.stack.stackbase import Stack

# List of the names of all the data loggers
loggers = ['USEPE_ML_Input', 'USEPE_ML_Conflict_Number']
ex_loggers = ['USEPE_ML_LoS_Number', 'USEPE_ML_Conflict_Duration', 'USEPE_ML_LoS_Duration', 'USEPE_ML_Conflict_Duration_Period', 'USEPE_ML_LoS_Duration_Period'] #EXTEND
strategic_logger = ['USEPE_ML_Strategical_Phase']
tactical_ex_loggers = ['USEPE_ML_TACTICAL_LoS_Number', 'USEPE_ML_TACTICAL_Conflict_Duration', 'USEPE_ML_TACTICAL_LoS_Duration', 'USEPE_ML_TACTICAL_Conflict_Duration_Period', 'USEPE_ML_TACTICAL_LoS_Duration_Period']
tactical_single_logger = ['USEPE_ML Tactical Separation Single All Features General for ', 'USEPE_ML Tactical Separation Single All Features Local for ']
debug_logger = ['USEPE_ML_DEBUG']

traffic_content = ['id', 'type', 'lat', 'lon', 'alt', 'distflown', 'hdg', 'trk', 'tas', 'gs', 'gsnorth', 'gseast', 'cas', 'M', 'selspd', 'aptas', 'selalt']

# Parameters used when logging
ml_input_header = \
    'Machine Learning Log for Input\n' + \
    'simt, id, type, lat, lon, alt, distflown, hdg, trk, tas, gs, gsnorth, gseast, cas, M, selspd, aptas, selalt'

conf_count_header = \
    'Number of total conflicts for every pair\n' + \
    'Start and end of [counted] all conflicts\n' + \
    'simt: simulation duration, Index: id of pair of Aircrafts, Aircraft 1, Aircraft 2, confNum: Number of conflicts between aircraft pairs\n' +\
    'end_simt, Index, Aircraft1, Aircraft2, confNum'
# Parameters used when logging (EXTEND)
los_count_header = \
    'LOSS NUM LOG\n' + \
    'Start and end of [counted] all loss of separation\n' + \
    'simt: simulation duration, Index: id of pair of Aircrafts, Aircraft 1, Aircraft 2, losNum: Number of LoS between aircraft pairs\n' +\
    'end_simt, Index, Aircraft1, Aircraft2, losNum'
conf_duration_header = \
    'CONF OF DURATION LOG (with DURATION LENGTH)\n' +\
    'simt: simulation duration, Index: id of pair of Aircrafts, Aircraft 1, Aircraft 2, Duration Time[s]: duration length within conflicted pair aircraft\n' +\
    'end_simt, Index, Aircraft1, Aircraft2, Duration Time[s]'
los_duration_header = \
    'LOSS OF DURATION LOG (with DURATION LENGTH)\n' +\
    'simt: simulation duration, Index: id of pair of Aircrafts, Aircraft 1, Aircraft 2, Duration Time[s]: duration length within los pair aircraft\n' +\
    'end_simt, Index, Aircraft1, Aircraft2, Duration Time[s]'
conf_duration_period_header = \
    'CONF OF DURATION LOG (with DURATION PERIOD)\n' +\
    'simt: simulation duration, Index: id of pair of Aircrafts, Aircraft 1, Aircraft 2, start duration Time[s]: start duration time within conflicted pair aircraft, end duration Time[s]: end duration time within conflicted pair aircraft, duration period Time[s]:  end duration Time[s] - start duration Time[s]\n' +\
    'end_simt, Index, Aircraft1, Aircraft2, start duration Time[s], end duration Time[s], duration period Time[s]'
los_duration_period_header = \
    'LOSS OF DURATION LOG (with DURATION PERIOD)\n' +\
    'simt: simulation duration, Index: id of pair of Aircrafts, Aircraft 1, Aircraft 2, start duration Time[s]: start duration time within los pair aircraft, end duration Time[s]: end duration time within los pair aircraft, duration period Time[s]:  end duration Time[s] - start duration Time[s]\n' +\
    'end_simt, Index, Aircraft1, Aircraft2, start duration Time[s], end duration Time[s], duration period Time[s]'
# Parameters used when logging (STRATEGIC)
strategic_header = \
    'Machine Learning Log for Input\n' + \
    'log_simt, simt, id, type, lat, lon, alt'

# Get this plugin dir (needed to search "scenario" dir)
PLUGIN_DIR = os.path.dirname(__file__)
SCENARIO_DIR = os.path.join(PLUGIN_DIR, '..', 'scenario')

# The data loggers
aircraft_ids = None
ml_inputlog = None
count_conflog = None
# The data loggers (EXTEND)
count_loslog = None
duration_conflog = None
duration_losflog = None
duration_period_conflog = None
duration_period_loslog = None
# The data loggers (STRATEGIC)
strategic_log = []
# The data loggers (TACTICAL)
tactical_log = None
tactical_conflog = None
# The data logger (DEBUG)
debug_log = None
# Update interval
updateInterval = 1.0

### Initialisation function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
# BASIC
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate the USEPE_ML entity
    usepe_ml = USEPE_ML()
    # Create the loggers
    #global aircraft_ids

    global count_conflog
    global ml_inputlog
    # Create log file (still empty)
    ml_inputlog = datalog.crelog(loggers[0], None, ml_input_header)
    count_conflog = datalog.crelog(loggers[1], None, conf_count_header)
    # Get aircraft ids
    #aircraft_ids = []
    #for id in traf.id:
    #    aircraft_ids.append(id)
    # Configuration parameters

    stackfunctions = {
        'USEPE_ML': [
            'USEPE_ML BASIC/EXTRA/EXTEND/STRATEGIC/TACTICAL/TACTICAL EXTEND/OFF/HELP,or HELP COMMAND',
            '[txt], [txt]',
            usepe_ml.usepe_ml,
            'enable all the available data loggers/write down all data loggers'
        ]
    }
    config = {
        'plugin_name':     'USEPE_ML',
        'plugin_type':     'sim',
        'update_interval': updateInterval,
        'update': usepe_ml.update
        }

    # init_plugin() should always return a configuration dict.
    return config, stackfunctions
# For ML EXTEND
def activate_extend_loggers():
    global count_loslog
    global duration_conflog
    global duration_losflog
    global duration_period_conflog
    global duration_period_loslog
    # Create log file (still empty)
    count_loslog = datalog.crelog(ex_loggers[0], None, los_count_header)
    duration_conflog = datalog.crelog(ex_loggers[1], None, conf_duration_header)
    duration_losflog = datalog.crelog(ex_loggers[2], None, los_duration_header)
    duration_period_conflog = datalog.crelog(ex_loggers[3], None, conf_duration_period_header)
    duration_period_loslog = datalog.crelog(ex_loggers[4], None, los_duration_period_header)
# For ML STRATEGIC
def activate_strategic_loggers(scn_names):
    global strategic_log
    #strategic_logger.append('USEPE_ML_STRATEGIC')
    strategic_log.append(datalog.crelog(strategic_logger[-1], None, strategic_header))

# lon, lat, name
def extract_shortcut_from_navdata(file_path='./data/navdata/fix.dat'):
    name2lon_dic = {}
    name2lat_dic = {}
    with open(file_path, "r") as f:
        data_list = f.readlines()[3:]
        for data in data_list[:-1]: # lon, lat, name
            data = data.split()
            name2lon_dic[data[-1]] = float(data[0])
            name2lat_dic[data[-1]] = float(data[1])

    return name2lon_dic, name2lat_dic

def extract_airport_from_navdata(file_path='./data/navdata/airports.dat'):
    # code,name,lat,lon,class,maxrunway, country code, elevation in sft
    airport_d = data_import(file_path, 0)
    code2lon_dic = {}
    code2lat_dic = {}
    for idx, d in enumerate(airport_d.iterrows()):
        code = airport_d.iloc[idx, 0]
        lon = airport_d.iloc[idx, 3]
        lat = airport_d.iloc[idx, 2]
        code2lon_dic[code] = lon
        code2lat_dic[code] = lat

    return code2lon_dic, code2lat_dic

# For ML TACTICAL
def activate_tactical_loggers(is_extend):
    # Parameters used when logging (TACTICAL)

    tactical_header = ml_input_header
    global tactical_log
    global tactical_conflog
    tactical_log = datalog.crelog(loggers[0], None, tactical_header)
    tactical_conflog = datalog.crelog(loggers[1], None, conf_count_header)

def activate_extend_tactical_loggers():
    global count_loslog
    global duration_conflog
    global duration_losflog
    global duration_period_conflog
    global duration_period_loslog
    # Create log file (still empty)
    count_loslog = datalog.crelog(tactical_ex_loggers[0], None, los_count_header)
    duration_conflog = datalog.crelog(tactical_ex_loggers[1], None, conf_duration_header)
    duration_losflog = datalog.crelog(tactical_ex_loggers[2], None, los_duration_header)
    duration_period_conflog = datalog.crelog(tactical_ex_loggers[3], None, conf_duration_period_header)
    duration_period_loslog = datalog.crelog(tactical_ex_loggers[4], None, los_duration_period_header)

class USEPE_ML(core.Entity):
    ''' Provides the needed funcionality for each log. '''
    # Init
    def __init__(self):
        super().__init__()
        # These lists stores the events from the previous step,
        # ensuring each event is logged only once and that we know when they have ended.
        self.prevconf = list()
        self.prevlos = list()

        self.is_basic = False       # for USEPE_ML BASIC
        self.is_extra = False       # for USEPE_ML EXTRA
        self.is_extend = False      # for USEPE_ML EXTEND
        self.is_strategic = False   # for USEPE_ML STRATEGIC
        self.scn_fname_list = None  # for USEPE_ML STRATEGIC
        self.is_tactical = False    # for USEPE_ML TACTICAL
        self.is_tactical_extend = False  # for USEPE_ML TACTICAL EXTEND
        self.is_debug = False # for USEPE_ML DEBUG

        self.conf_counter = {}
        self.los_counter = {}  # for USEPE_ML EXTRA
        self.duration_conf_start_dic = {}  # for USEPE_ML EXTEND
        self.duration_conf_end_dic = {}  # for USEPE_ML EXTEND
        self.duration_los_start_dic = {}  # for USEPE_ML EXTEND
        self.duration_los_end_dic = {}  # for USEPE_ML EXTEND

        self.pair_index_dic = {} # pair_name: idx
        self.pair_index = 1

        self.start_traf_num = len(traf.id)

        self.conf_num_in_sec = dict() # sec:[pair_names]
        self.los_num_in_sec = dict() # sec:[pair_names]

    # Use for Debug
    def debug(self):
        # Get cmd from scenario file
        scn_fname = Stack.scenname
        cmd_ic_load = True
        # Search scn files
        scn_path = glob.glob(SCENARIO_DIR + '/{}.scn'.format(scn_fname))
        if len(scn_path) == 0: # Case of .SCN
            scn_path = glob.glob(SCENARIO_DIR + '/{}.SCN'.format(scn_fname))
            if len(scn_path) == 0: # No cmd: IC *.scn
                cmd_ic_load = False
        if len(scn_path) != 0: # Extract matched scn file
            scn_path = scn_path[0]
        # Set header
        # Parameters used when logging (STRATEGIC)
        debug_header = \
            '=== Diagnostic information ===\n'
        # Init logger
        global debug_log
        debug_log = datalog.crelog(debug_logger[0], None, debug_header)
        stack.stack(f'{debug_logger[0]} BASIC')
        allloggers[debug_logger[0]].start()
        # Output to log
        # Read scenario file -------------------------------------------------
        if cmd_ic_load is True:
            with open(scn_path, "r") as fscen:
                for line in fscen:
                    debug_info = []
                    line = line.strip()
                    # Skip empty lines and comments
                    if len(line) < 12 or line[0] == "#":
                        continue
                    # Read cmd
                    icmdline = line.index(">")
                    tstamp = line[:icmdline]
                    # Get command
                    cmd_line = line[icmdline + 1:]
                    cmd_line = cmd_line.replace(',', ' ') # Except comma
                    cmd_line = cmd_line.strip() # Except first&end space
                    cmd_line = cmd_line.split()
                    cmd = cmd_line[0]
                    target_cmd = False
                    opt = None
                    # Extract [ASAS, RESO]
                    if cmd == 'ASAS':
                        target_cmd = True
                        try:
                            opt = cmd_line[1]
                        except: # Case without the option
                            opt = None
                    elif cmd == 'RESO':
                        target_cmd = True
                        try:
                            opt = cmd_line[1]
                        except:
                            opt = None
                    # Write to log
                    if target_cmd is True:
                        debug_info.append(cmd)
                        debug_info.append(opt)
        # Write to debug_log
        if len(debug_info) != 0:
            debug_log.log([debug_info])
        # --------------------------------------------------------------------
        # Get python library -------------------------------------------------
        # system state

        #Bluesky version ------
        bluesky_version = None
        with open("setup.py") as f:
            for line in f:
                if 'version=' in line:
                    bluesky_version = line[13:-3]
        #----------------------

        print("--------------SYSTEM STATE------------------------------")
        print("\t BlueSky version is {}.".format(bluesky_version))
        print("\t Python version is {}.".format(sys.version))
        debug_log.log(["BlueSky version = {}.".format(bluesky_version)])
        debug_log.log(["Python version is {}.".format(sys.version)])

        # library state
        print("--------------LIBRARY STATE--------------------------")
        lib_list = ["seaborn", "sklearn", "matplotlib", "numpy", "pandas"]
        for lib in lib_list:
            debug_info = []
            if lib in sys.modules:
                if lib == "seaborn":
                    version = sns.__version__
                    print("\t seaborn = {}".format(version))
                elif lib == "sklearn":
                    import sklearn
                    version = sklearn.__version__
                    print("\t sklearn = {}".format(version))
                elif lib == "matplotlib":
                    import matplotlib
                    version = matplotlib.__version__
                    print("\t matplotlib = {}".format(version))
                elif lib == "numpy":
                    version = np.__version__
                    print("\t numpy = {}".format(version))
                elif lib == "pandas":
                    version = pd.__version__
                    print("\t pandas = {}".format(version))

                debug_info.append(lib)
                debug_info.append(version)
            if len(debug_info) != 0:
                debug_log.log([debug_info])
        # --------------------------------------------------------------------
        # Get USEPE_ML cmd ---------------------------------------------------
        # cmd state
        print("--------------ACTIVATION STATE------------------------------")
        print("\t BASIC is {}".format(self.is_basic))
        debug_log.log([['USEPE_ML BASIC', self.is_basic]])
        print("\t EXTRA is {}".format(self.is_extra))
        debug_log.log([['USEPE_ML EXTRA', self.is_extra]])
        print("\t EXTEND is {}".format(self.is_extend))
        debug_log.log([['USEPE_ML EXTEND', self.is_extend]])
        print("\t STRATEGIC is {}".format(self.is_strategic))
        debug_log.log([['USEPE_ML STRATEGIC', self.is_strategic]])
        print("\t TACTICAL is {}".format(self.is_tactical))
        debug_log.log([['USEPE_ML TACTICAL', self.is_tactical]])
        print("\t TACTICAL EXTEND is {}".format(self.is_tactical_extend))
        debug_log.log([['USEPE_ML TACTICAL EXTEND', self.is_tactical_extend]])

        # Scenario File
        debug_log.log([" "])
        debug_log.log(["Beginning of The Scenario File --------"])
        scn_info=[]
        if cmd_ic_load is True:
            with open(scn_path, "r") as fscen:
                for line in fscen:
                    scn_info.append(line)
                debug_log.log([scn_info])
        debug_log.log(["End of The Scenario File---------------"])

        # log state
        print("--------------LOG STATE------------------------------")
        for log_name, v in allloggers.items():
            try:
                print("\t {} = I/O closed is {}".format(log_name, allloggers[log_name].file.flush()))
            except:
                continue
        # --------------------------------------------------------------------
        # Close log
        debug_fname = allloggers[debug_logger[0]].fname
        allloggers[debug_logger[0]].file.close()
        simt_remover(debug_fname) # Remove [# simt]
        # Remove time in column
        file_name = debug_fname
        lines = []
        # Read file
        with open(file_name, 'r') as fp:
            lines = fp.readlines()

        # Write file
        with open(file_name, 'w') as fp:
            for number, line in enumerate(lines):
                if number >= 2: # write cmd & opt (except time)
                    line = line.split(',', 1)
                    output_line = line[1:]
                    output_line = output_line[0]
                    output_line = output_line.replace(',', ' = ')
                else: # write header
                    output_line = line

                fp.write(output_line)
    # Reset
    def reset(self):
        global aircraft_ids
        # Get aircraft ids
        aircraft_ids = []
        for id in traf.id:
            aircraft_ids.append(id)
        self.prevconf = list()
        self.prevlos = list()

        self.conf_counter = {}
        self.los_counter = {}
        self.duration_conf_start_dic = {}
        self.duration_conf_end_dic = {}
        self.duration_los_start_dic = {}
        self.duration_los_end_dic = {}

        self.pair_index_dic = {}
        self.pair_index = 1

        self.start_traf_num = len(traf.id)

        self.conf_num_in_sec = dict()
        self.los_num_in_sec = dict()

    # Update
    def update(self):
        ''' Periodic function calling each logger function. '''
        if self.is_strategic is False:
            self.input_logger()
            self.conf_logger()
            # Activate los logger
            if self.is_extend is True:
                self.los_logger()
            # Acivate tactical sim
            if self.is_tactical is True:
                self.print_ontime_sep_score()
        else:
            if self.scn_fname_list is not None:
                self.output_wpt(self.scn_fname_list)
                self.is_strategic = False

    # Write ML Log
    def input_logger(self):
        # Get each aircraft info
        ml_input_dic = {}  # name: List[]
        simt = bs.sim.simt
        for idx, name in enumerate(traffic_content):
            value = getattr(traf, name)
            ml_input_dic[name] = value
        # Write
        for i in range(len(traf.id)):
            input_log = []
            for name in traffic_content:
                input_log.append(ml_input_dic[name][i])

            if (self.is_tactical is False) and (self.is_tactical_extend is False):
                ml_inputlog.log([input_log])
            else:
                tactical_log.log([input_log])

    # Strategic
    def print_ontime_sep_score(self):
        log = []
        traf_names = traf.id
        simt = bs.sim.simt # Get current time
        traf_content_dic = {} # name: List[] (ids)
        traf_name2id_dic = {} # name: id
        traf_pair2sep_dic = {} # pairname: sep_score

        currentconf = [sorted(pair) for pair in traf.cd.confpairs_unique] # Get conf [[aircraft1, aircraft2], []...]
        # Define calc sep contents
        separation_contents = None
        sep_category_name = None
        if self.is_extra:
            separation_contents = ['lat', 'lon', 'alt', 'distflown', 'hdg', 'trk', 'tas', 'gs', 'gsnorth', 'gseast', 'cas', 'M', 'selspd', 'aptas', 'selalt']
            sep_category_name = 'All features'
        else:
            separation_contents = ['lat', 'lon', 'alt']
            sep_category_name = 'Spatial features'
        # Get traf content vals 
        for idx, name in enumerate(traffic_content):
            value = getattr(traf, name)
            traf_content_dic[name] = value
        # Get traf name2id
        for idx, name in enumerate(traf_names):
            traf_name2id_dic[name] = idx
        # Get pairwise sep score
        if len(traf_names) > 1:
            for aircraft1, aircraft2 in itertools.combinations(traf_names, 2):
                # get id from name
                aircraft1_id = traf_name2id_dic[aircraft1]
                aircraft2_id = traf_name2id_dic[aircraft2]
                # get each sep content
                aircraft1_data = []
                aircraft2_data = []
                for name in separation_contents:
                    aircraft1_data.append(traf_content_dic[name][aircraft1_id])
                    aircraft2_data.append(traf_content_dic[name][aircraft2_id])
                aircraft1_data = np.array(aircraft1_data).reshape(1, -1)
                aircraft2_data = np.array(aircraft2_data).reshape(1, -1)
                # Get sep score
                sep_score = get_sep_score(aircraft1_data, aircraft2_data)
                traf_pair2sep_dic['{}&{}'.format(aircraft1, aircraft2)] = sep_score
    # Conf
    def conf_logger(self):
        ''' Sorts current conflicts and logs new and ended events. '''
        currentconf = list()

        # Go through all conflict pairs and sort the IDs for easier matching
        currentconf = [sorted(pair) for pair in traf.cd.confpairs_unique]

        # Create lists of all new and ended conflicts
        startconf = [currpair for currpair in currentconf if currpair not in self.prevconf]
        endconf = [prevpair for prevpair in self.prevconf if prevpair not in currentconf]
        # Initialize conflict number in second
        sec = float(bs.sim.simt)
        self.conf_num_in_sec[sec] = []
        # WRITE conf info -------------------------------------------------------------
        for idx, unique_pair in enumerate(traf.cd.confpairs_unique):
            # Log pair log
            a, b = unique_pair
            a, b = sorted([a, b])
            # Update index
            if "{}&{}".format(a, b) not in self.pair_index_dic.keys():
                self.pair_index_dic["{}&{}".format(a, b)] = self.pair_index
                self.pair_index += 1
            # Increment
            self.conf_num_in_sec[sec].append("{}&{}".format(a, b))
        # ------------------------------------------------------------------------------
        # STOCK conf duration ----------------------------------------------------------
        for unique_pair in startconf:
            a, b = unique_pair
            a, b = sorted([a, b])
            if '{}&{}'.format(a, b) not in self.duration_conf_start_dic.keys():
                self.duration_conf_start_dic['{}&{}'.format(a, b)] = [bs.sim.simt]
            else:
                self.duration_conf_start_dic['{}&{}'.format(a, b)].append(bs.sim.simt)
        # STOCK conf info (conf num) ---------------------------------------------------
        for unique_pair in endconf:
            a, b = unique_pair
            a, b = sorted([a, b])
            # register "conf count"
            if '{}&{}'.format(a, b) not in self.conf_counter.keys():
                self.conf_counter['{}&{}'.format(a, b)] = 1
            else:
                self.conf_counter['{}&{}'.format(a, b)] += 1
            # register "duration "end"
            if '{}&{}'.format(a, b) in self.duration_conf_start_dic.keys():
                # Duration counter by end - start
                if '{}&{}'.format(a, b) not in self.duration_conf_end_dic.keys():
                    self.duration_conf_end_dic['{}&{}'.format(a, b)] = [bs.sim.simt - updateInterval] # Only bs.sim.simt shows the time the pairname is vanished
                else:
                    self.duration_conf_end_dic['{}&{}'.format(a, b)].append(bs.sim.simt - updateInterval)

        # ------------------------------------------------------------------------------
        # Store the new conflict environment
        self.prevconf = currentconf
    # LoS
    def los_logger(self):
        ''' Sorts current LoS and logs new and ended events. '''
        currentlos = list()

        # Go through all loss of separation pairs and sort the IDs for easier matching
        currentlos = [sorted(pair) for pair in traf.cd.lospairs_unique]

        # Create lists of all new and ended LoS
        startlos = [currpair for currpair in currentlos if currpair not in self.prevlos]
        endlos = [prevpair for prevpair in self.prevlos if prevpair not in currentlos]
        # Initialize LoS number in second
        sec = bs.sim.simt
        self.los_num_in_sec[sec] = []
        # WRITE los info --------------------------------------------------------------
        for idx, unique_pair in enumerate(traf.cd.lospairs_unique):
            a, b = unique_pair
            a, b = sorted([a, b])
            # Update index
            if "{}&{}".format(a, b) not in self.pair_index_dic.keys():
                self.pair_index_dic["{}&{}".format(a, b)] = self.pair_index
                self.pair_index += 1
            # Increment
            self.los_num_in_sec[sec].append("{}&{}".format(a, b))
        # ------------------------------------------------------------------------------
        # STOCK los duration -----------------------------------------------------------
        # register "duration start"
        for unique_pair in startlos:
            a, b = unique_pair
            a, b = sorted([a, b])
            if '{}&{}'.format(a, b) not in self.duration_los_start_dic.keys():
                self.duration_los_start_dic['{}&{}'.format(a, b)] = [bs.sim.simt]
            else:
                self.duration_los_start_dic['{}&{}'.format(a, b)].append(bs.sim.simt)
        # STOCK LoS info (LoS num) ------------------------------------------------------
        for unique_pair in endlos:
            # register "LoS count"
            a, b = unique_pair
            a, b = sorted([a, b])
            if '{}&{}'.format(a, b) not in self.los_counter.keys():
                self.los_counter['{}&{}'.format(a, b)] = 1
            else:
                self.los_counter['{}&{}'.format(a, b)] += 1
            # register "duration start"
            if '{}&{}'.format(a, b) in self.duration_los_start_dic.keys():
                # Duration counter by end - start
                if '{}&{}'.format(a, b) not in self.duration_los_end_dic.keys():
                    # New duration
                    self.duration_los_end_dic['{}&{}'.format(a, b)] = [bs.sim.simt - updateInterval]
                else:
                    # Increment duration
                    self.duration_los_end_dic['{}&{}'.format(a, b)].append(bs.sim.simt - updateInterval)
        # ------------------------------------------------------------------------------
        # Store the new loss of separation environment
        self.prevlos = currentlos
    #Strategic analysis
    def output_wpt(self, scenario_path_list):
        import re
        # Get position from code
        placeName2lon_dic, placeName2lat_dic = extract_shortcut_from_navdata()
        airportCode2lon_dic, airportCode2lat_dic = extract_airport_from_navdata()
        # Init data
        log = []
        aircraftID2type = {}
        aircraftWP2place = {}
        for idx, scenario_path in enumerate(scenario_path_list):
            with open(scenario_path, "r") as fscen:
                for line in fscen:
                    line = line.strip()
                    # Skip emtpy lines and comments
                    if len(line) < 12 or line[0] == "#":
                        continue
                    # Read cmd
                    icmdline = line.index(">")
                    tstamp = line[:icmdline]
                    # Get time
                    ttxt = tstamp.strip().split(":")
                    ihr = int(ttxt[0]) * 3600.0
                    imin = int(ttxt[1]) * 60.0
                    xsec = float(ttxt[2])
                    cmdtime = ihr + imin + xsec
                    # Get command
                    cmd_line = line[icmdline + 1:]
                    # Get airtype
                    cmd_group = re.search(r'\S+', cmd_line)
                    cmd_line = cmd_group.string
                    
                    cmd_line = cmd_line.replace(',', ' ') # Except comma
                    cmd_line = cmd_line.strip() # Except first&end space
                    cmd_line = cmd_line.split()
                    
                    aircraft_id, aircraft_type = None, None
                    lat, lon, alt = None, None, None
                    if ('CRE' in cmd_line[0]) or ('Cre' in cmd_line[0]):
                        aircraft_id = str(cmd_line[1])
                        aircraft_type = cmd_line[2]
                        # Register to dic
                        if aircraft_id not in aircraftID2type.keys():
                            aircraftID2type[aircraft_id] = aircraft_type
                        # Get lon lat alt
                        try: 
                            # pattern -> CRE 0003 B744 -4.0 -1.5 60.3218243188 0 500
                            lat, lon, alt = float(cmd_line[3]), float(cmd_line[4]), float(cmd_line[6])
                        except: 
                            # pattern -> Cre KL204,E190, EHEH, RWY03, *,  0, 0
                            airport_name = cmd_line[3]
                            lon = airportCode2lon_dic[airport_name]
                            lat = airportCode2lat_dic[airport_name]
                            alt = cmd_line[-2]
                        log.append(
                            [cmdtime, aircraft_id, aircraft_type, lat, lon, alt])

                    elif ('DEFWPT' in cmd_line[0]):
                        # pattern -> DEFWPT Drone1_wp1 59.67005 9.64792 (set alt=0)
                        wp_name = cmd_line[1]
                        lat, lon, alt = float(cmd_line[2]), float(cmd_line[3]), float(0)
                        aircraftWP2place[wp_name] = [lat, lon, alt]
                    elif ('ADDWPT' in cmd_line[0]):
                        aircraft_id = str(cmd_line[1])
                        aircraft_type = aircraftID2type[aircraft_id]
                        # Get lon lat alt
                        try:
                            # pattern -> ADDWPT 0002 2.6815 0.5 5014.0 500
                            lat, lon, alt = float(cmd_line[2]), float(cmd_line[3]), float(cmd_line[4])
                        except:
                            wp_name = place_name = cmd_line[2]
                            lat, lon, alt = None, None, None
                            if place_name in placeName2lon_dic.keys():
                                # pattern -> ADDWPT KL206 VIRTU 6000 (set alt=6000)
                                # or
                                # pattern -> ADDWPT KL206 VIRTU (set alt=0)
                                lon = placeName2lon_dic[place_name]
                                lat = placeName2lat_dic[place_name]
                                if len(cmd_line) > 3:
                                    alt = cmd_line[-1]
                                else:
                                    alt = 0
                            elif wp_name in aircraftWP2place.keys():
                                # pattern -> ADDWPT Drone1 Drone1_wp1
                                lat, lon, alt = aircraftWP2place[wp_name]

                        log.append([cmdtime, aircraft_id, aircraft_type, lat, lon, alt])

        # Output to log
        strategic_log[-1].log(log)
        # Close log
        print(allloggers)
        allloggers[strategic_logger[-1]].file.close()

        strategic_log_fname = allloggers[strategic_logger[-1]].fname
        # Output csv & plot

        strategic_pair_csv(strategic_log_fname)
        strategic_pair_plot(strategic_log_fname)

        strategic_local_csv(strategic_log_fname)
        strategic_local_plot(strategic_log_fname)

        strategic_general_csv(strategic_log_fname)
        strategic_general_plot(strategic_log_fname)

        simt_remover(strategic_log_fname)
        
        # Reset log
        allloggers[strategic_logger[-1]].reset()
        # Force reset
        bs.sim.reset()

    def summarize_logger(self):
        # Write Stocked Info
        conf_count_log = []
        for (a, b) in itertools.combinations(aircraft_ids, 2):
            # Update index
            if "{}&{}".format(a, b) not in self.pair_index_dic.keys():
                self.pair_index_dic["{}&{}".format(a, b)] = self.pair_index
                self.pair_index += 1
            # WRITE CONTER LOG ---------------------------------------------------------------------------------------------------
            # get count
            conf_count = self.conf_counter["{}&{}".format(a, b)] if "{}&{}".format(a, b) in self.conf_counter.keys() else 0
            # append to log
            conf_count_log.append([self.pair_index_dic["{}&{}".format(a, b)], a, b, conf_count])
            # ---------------------------------------------------------------------------------------------------------------------
        if self.is_tactical is False:
            count_conflog.log(conf_count_log)
        else:
            tactical_conflog.log(conf_count_log)

    def summarize_with_extend_logger(self):
        # Write Stocked Info
        conf_count_log, los_count_log = [], []
        conf_duration_log, los_duration_log = [], []
        conf_duration_period_log, los_duration_period_log = [], []
        for (a, b) in itertools.combinations(aircraft_ids, 2):
            # Update index
            if "{}&{}".format(a, b) not in self.pair_index_dic.keys():
                self.pair_index_dic["{}&{}".format(a, b)] = self.pair_index
                self.pair_index += 1
            # WRITE CONTER LOG ---------------------------------------------------------------------------------------------------
            # get count
            conf_count = self.conf_counter["{}&{}".format(a, b)] if "{}&{}".format(a, b) in self.conf_counter.keys() else 0
            los_count = self.los_counter["{}&{}".format(a, b)] if "{}&{}".format(a, b) in self.los_counter.keys() else 0
            # append to log
            conf_count_log.append([self.pair_index_dic["{}&{}".format(a, b)], a, b, conf_count])
            los_count_log.append([self.pair_index_dic["{}&{}".format(a, b)], a, b, los_count])
            # ---------------------------------------------------------------------------------------------------------------------
            # WRITE DURATION LOG --------------------------------------------------------------------------------------------------
            # get duration (start & end)
            conf_start_times = self.duration_conf_start_dic['{}&{}'.format(
                a, b)] if "{}&{}".format(a, b) in self.duration_conf_start_dic.keys() else [0]
            conf_end_times = self.duration_conf_end_dic['{}&{}'.format(
                a, b)] if "{}&{}".format(a, b) in self.duration_conf_end_dic.keys() else [0]

            # append duration conf log
            conf_duration_length = 0
            for idx in range(len(conf_start_times)):
                s = conf_start_times[idx]
                e = conf_end_times[idx]
                conf_duration_length += e - s
                conf_duration_period_log.append([self.pair_index_dic["{}&{}".format(a, b)], a, b, s, e, e-s])
            conf_duration_log.append([self.pair_index_dic["{}&{}".format(a, b)], a, b, conf_duration_length])

            los_start_times = self.duration_los_start_dic['{}&{}'.format(a, b)] if "{}&{}".format(a, b) in self.duration_los_start_dic.keys() else [0]
            los_end_times = self.duration_los_end_dic['{}&{}'.format(a, b)] if "{}&{}".format(a, b) in self.duration_los_end_dic.keys() else [0]
            # append duration los log
            los_duration_length = 0
            for idx in range(len(los_start_times)):
                s = los_start_times[idx]
                e = los_end_times[idx]
                los_duration_length += e - s
                los_duration_period_log.append([self.pair_index_dic["{}&{}".format(a, b)], a, b, s, e, e-s])
            los_duration_log.append([self.pair_index_dic["{}&{}".format(a, b)], a, b, los_duration_length])
            # ---------------------------------------------------------------------------------------------------------------------

        count_conflog.log(conf_count_log)
        count_loslog.log(los_count_log)
        duration_conflog.log(conf_duration_log)
        duration_losflog.log(los_duration_log)
        duration_period_conflog.log(conf_duration_period_log)
        duration_period_loslog.log(los_duration_period_log)

    def usepe_ml(self, cmd="Help", cmd2=True):
        ''' USEPE_ML command for the plugin.
            Options:
            BASIC: Enable all the data loggers
            OFF: Disable all the data loggers
            EXTEND: Extend to all features
            EXTRA: Extra analysis are printed
            STRATEGIC: Strategic analysis for scenario files
            TACTICAL: Analysis during simulation
            USEPE: About the olugin
            HELP: Help for available commands            
             '''
        #----------------------------------------------------------------------------------------
        if cmd == 'BASIC':
            self.reset()
            
            if self.is_strategic is False:
                self.is_basic = True
                for x, log_name in enumerate(loggers):
                    stack.stack(f'{loggers[x]} BASIC')
                    allloggers[log_name].start()

                return True, f'USEPE_ML is enabled.'  # All data loggers for USEPE enabled. {str.join(", ", loggers)}'
            else:
                return True, f'Input USEPE_ML STRATEGIC/*.scn/*.scn ... to start STRATEGIC MODE'

        elif cmd == 'EXTRA':
            if allloggers[loggers[0]].isopen():
                self.is_extra = True
            else:
                self.is_extra = True
                self.reset()
                if self.is_extend is False:
                    for x, log_name in enumerate(loggers):
                        stack.stack(f'{loggers[x]} BASIC')
                        allloggers[log_name].start()
                else:
                    activate_extend_loggers()
                    for x, log_name in enumerate(loggers + ex_loggers):
                        if allloggers[log_name].isopen() is False:
                            if x < len(loggers):
                                stack.stack(f'{loggers[x]} BASIC')
                            else:
                                y = x - len(loggers)
                                stack.stack(f'{ex_loggers[y]} BASIC')
                            allloggers[log_name].start()

            return True, f'USEPE_ML with EXTRA analyses are enabled.'

        elif cmd == 'EXTEND':
            self.reset()
            self.is_extend = True
            activate_extend_loggers()
            for x, log_name in enumerate(loggers + ex_loggers):
                if allloggers[log_name].isopen() is False:
                    if x < len(loggers):
                        stack.stack(f'{loggers[x]} BASIC')
                    else:
                        y = x - len(loggers)
                        stack.stack(f'{ex_loggers[y]} BASIC')
                    allloggers[log_name].start()

            return True, f'USEPE_ML with EXTENDED analyses are enabled.'
        # -------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------
        # Read *.scn and output to calculate separation score ---------------------------------------
        elif cmd[:9] == 'STRATEGIC': # STRATEGIC/*.SCN
            import re
            self.is_strategic = True
            scn_names = cmd.split('/')[1:]
            # Get scn path
            self.scn_fname_list = []
            for scn_name in scn_names:
                scn_fname = None
                if scn_name[-3:] == 'scn':
                    scn_fname = scn_name
                else:
                    scn_fname = '{}.scn'.format(scn_name)
                scn_fname = os.path.join(settings.scenario_path, scn_fname)
                self.scn_fname_list.append(scn_fname)
                # Check exist
                if os.path.exists(scn_fname) is False:
                    return False, f'SCN file does NOT exist!'
            # Activate logger
            activate_strategic_loggers(self.scn_fname_list)
            for x, log_name in enumerate(strategic_logger):
                stack.stack(f'{strategic_logger[x]} BASIC')
                allloggers[log_name].start()
            return True, f'USEPE_ML STRATEGIC is enabled.'

        # --------------------------------------------------------------------------------------------
        # Activate tactical commmand -----------------------------------------------------------------
        elif cmd == 'TACTICAL': # TACTICAL or TACTICAL EXTEND
            if self.is_tactical is False:
                self.reset()
                self.is_tactical = True
            # Activate EXTEND
            if cmd2 == 'EXTEND':
                self.reset()
                self.is_tactical_extend = True
                
                # Activate extend automatically (use duration period for plot)
                if self.is_extend is False:
                    self.reset()
                    self.is_extend = True
            # Activate logger
            # Initialize loggers
            # for x, log_name in enumerate(loggers):
            #     stack.stack(f'{loggers[x]} ON')
            #     allloggers[log_name].start()
            #     print('{}:{}'.format(log_name, allloggers[log_name].file.flush()))
            # Tactical loggers
            activate_tactical_loggers(self.is_extend)
            for x, log_name in enumerate(loggers):
                stack.stack(f'{loggers[x]} BASIC')
                allloggers[log_name].start()
                print('{}:{}'.format(log_name, allloggers[log_name].file.flush()))
            # Add extras
            if self.is_extend is True:
                activate_extend_tactical_loggers()
                for x, log_name in enumerate(tactical_ex_loggers):
                    stack.stack(f'{tactical_ex_loggers[x]} BASIC')
                    allloggers[log_name].start()

            if self.is_extend is False:
                return True, f'USEPE_ML with TACTICAL is enabled.'
            else:
                return True, f'USEPE_ML with TACTICAL with EXTEND is enabled.'

        # --------------------------------------------------------------------------------------------
        # ON/OFF debug commmand ----------------------------------------------------------------------
        elif cmd == 'DEBUG':
            # ON
            if self.is_debug is False:
                self.is_debug = True
                return True, f'USEPE_ML with DEBUG is enabled.'
            # OFF
            else:
                self.is_debug = False
                return True, f'USEPE_ML with DEBUG is disabled.'
        # --------------------------------------------------------------------------------------------
        elif cmd == 'OFF':

            # DEBUG ======================
            if self.is_debug is True:
                self.debug()
                self.is_debug = False

            # BASIC ======================
            if self.is_basic is True and self.is_extra is False and self.is_extend is False:
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'
                    self.summarize_logger()
                    output_loggers = loggers

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()
                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    # Pairwise
                    output_csv_and_plot_with_count_log(log_fname, count_log_files=[conf_count_fname], mode_list=['conf'], is_extra=self.is_extra)

                    local_spatial_conf_csv(log_fname, conf_count_fname)
                    local_spatial_conf_plot(log_fname, conf_count_fname)

                    general_spatial_conf_csv(log_fname, conf_count_fname)
                    general_spatial_conf_plot(log_fname, conf_count_fname)

                self.is_basic     = False
                self.is_strategic = False

            # BASIC + EXTRA ==============
            if self.is_basic is True and self.is_extra is True and self.is_extend is False:
                # LOGGERS
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'
                    self.summarize_logger()
                    output_loggers = loggers

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()
                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    # Outputs

                    # Pairwise
                    output_csv_and_plot_with_count_log(log_fname, count_log_files=[conf_count_fname], mode_list=['conf'], is_extra=self.is_extra)

                    # Single
                    local_spatial_conf_csv(log_fname, conf_count_fname)
                    local_spatial_conf_plot(log_fname, conf_count_fname)

                    general_spatial_conf_csv(log_fname, conf_count_fname)
                    general_spatial_conf_plot(log_fname, conf_count_fname)

                    local_all_conf_csv(log_fname, conf_count_fname)
                    local_all_conf_plot(log_fname, conf_count_fname)

                    general_all_conf_csv(log_fname, conf_count_fname)
                    general_all_conf_plot(log_fname, conf_count_fname)

                self.is_basic = False
                self.is_extra = False

            # BASIC + EXTEND =============
            if self.is_basic is True and self.is_extra is False and self.is_extend is True:
                  # LOGGERS
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'
                    self.summarize_logger()
                    self.summarize_with_extend_logger()
                    output_loggers = loggers + ex_loggers

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()
  
                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    # Get los count
                    los_count_fname = allloggers[ex_loggers[0]].fname
                    simt_remover(los_count_fname)

                    # Get conf duration csv
                    conf_duration_fname = allloggers[ex_loggers[1]].fname
                    simt_remover(conf_duration_fname)

                    # Get los duration csv
                    los_duration_fname = allloggers[ex_loggers[2]].fname
                    simt_remover(los_duration_fname)

                    # Get conf duration plot
                    conf_duration_period_fname = allloggers[ex_loggers[3]].fname
                    simt_remover(conf_duration_period_fname)

                    # Get los duration plot
                    los_duration_period_fname = allloggers[ex_loggers[4]].fname                        
                    simt_remover(los_duration_period_fname)

                    # Outputs

                    # Pairwise
                    output_csv_and_plot_with_count_log(log_fname, count_log_files=[conf_count_fname, los_count_fname], mode_list=['conf','los'], is_extra=self.is_extra)

                    # Single
                    local_spatial_conf_csv(log_fname, conf_count_fname)
                    local_spatial_conf_plot(log_fname, conf_count_fname)

                    general_spatial_conf_csv(log_fname, conf_count_fname)
                    general_spatial_conf_plot(log_fname, conf_count_fname)

                    local_spatial_los_csv(log_fname, los_count_fname)
                    local_spatial_los_plot(log_fname, los_count_fname)

                    general_spatial_los_csv(log_fname, los_count_fname)
                    general_spatial_los_plot(log_fname, los_count_fname)

                    output_csv_and_plot_with_duration_log(log_fname, duration_log_files=[conf_duration_fname, los_duration_fname], duration_period_log_files=[conf_duration_period_fname, los_duration_period_fname],
                                                            mode_list=['conf', 'los'], is_extra=self.is_extra)

                self.is_basic = False
                self.is_extend = False

            # BASIC + EXTEND + EXTRA =====
            if self.is_basic is True and self.is_extra is True and self.is_extend is True:
                  # LOGGERS
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'
                    self.summarize_logger()
                    self.summarize_with_extend_logger()
                    output_loggers = loggers + ex_loggers

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()
  
                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    # Get los count
                    los_count_fname = allloggers[ex_loggers[0]].fname
                    simt_remover(los_count_fname)

                    # Get conf duration csv
                    conf_duration_fname = allloggers[ex_loggers[1]].fname
                    simt_remover(conf_duration_fname)

                    # Get los duration csv
                    los_duration_fname = allloggers[ex_loggers[2]].fname
                    simt_remover(los_duration_fname)

                    # Get conf duration plot
                    conf_duration_period_fname = allloggers[ex_loggers[3]].fname
                    simt_remover(conf_duration_period_fname)

                    # Get los duration plot
                    los_duration_period_fname = allloggers[ex_loggers[4]].fname                        
                    simt_remover(los_duration_period_fname)

                    # Outputs

                    # Pairwise
                    output_csv_and_plot_with_count_log(log_fname, count_log_files=[conf_count_fname, los_count_fname], mode_list=['conf', 'los'], is_extra=self.is_extra)

                    # Single
                    local_spatial_conf_csv(log_fname, conf_count_fname)
                    local_spatial_conf_plot(log_fname, conf_count_fname)

                    general_spatial_conf_csv(log_fname, conf_count_fname)
                    general_spatial_conf_plot(log_fname, conf_count_fname)

                    local_spatial_los_csv(log_fname, los_count_fname)
                    local_spatial_los_plot(log_fname, los_count_fname)

                    general_spatial_los_csv(log_fname, los_count_fname)
                    general_spatial_los_plot(log_fname, los_count_fname)

                    local_all_conf_csv(log_fname, conf_count_fname)
                    local_all_conf_plot(log_fname, conf_count_fname)

                    general_all_conf_csv(log_fname, conf_count_fname)
                    general_all_conf_plot(log_fname, conf_count_fname)

                    local_all_los_csv(log_fname, los_count_fname)
                    local_all_los_plot(log_fname, los_count_fname)

                    general_all_los_csv(log_fname, los_count_fname)
                    general_all_los_plot(log_fname, los_count_fname)

                    output_csv_and_plot_with_duration_log(log_fname, duration_log_files=[conf_duration_fname, los_duration_fname], duration_period_log_files=[conf_duration_period_fname, los_duration_period_fname],
                                                            mode_list=['conf', 'los'], is_extra=self.is_extra)

                self.is_basic  = False
                self.is_extend = False
                self.is_extra  = False

                return True, f'USEPE_ML with BASIC is disabled.'

            # TACTICAL ===================
            if self.is_tactical is True and self.is_extra is False and self.is_tactical_extend is False:
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'

                    self.summarize_logger()
                    output_loggers = loggers

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()
                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    tactical_pairwise_spatial_conf_csv(log_fname, self.conf_num_in_sec) 
                    tactical_local_spatial_conf_csv(log_fname, self.conf_num_in_sec)
                    tactical_general_spatial_conf_csv(log_fname, self.conf_num_in_sec)

                    output_tactical_log(log_file=log_fname,
                                        count_log_files=[conf_count_fname],
                                        count_timeseries_list=[self.conf_num_in_sec],
                                        mode_list=['conf'],
                                        is_extra=self.is_extra,
                                        is_extend=self.is_extend,
                                        duration_log_files = [None], 
                                        duration_period_log_files = [None], output_extend=self.is_tactical_extend)

                self.is_tactical = False
                return True, f'USEPE_ML with TACTICAL is disabled.'

            # TACTICAL + EXTRA ===========
            if self.is_tactical is True and self.is_extra is True and self.is_tactical_extend is False:
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'

                    self.summarize_logger()
                    output_loggers = loggers 

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()
                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    tactical_pairwise_spatial_conf_csv(log_fname, self.conf_num_in_sec) 
                    tactical_local_spatial_conf_csv(log_fname, self.conf_num_in_sec)
                    tactical_general_spatial_conf_csv(log_fname, self.conf_num_in_sec)

                    tactical_pairwise_all_conf_csv(log_fname, self.conf_num_in_sec)
                    tactical_local_all_csv(log_fname, self.conf_num_in_sec)
                    tactical_general_all_csv(log_fname, self.conf_num_in_sec)

                    output_tactical_log(log_file=log_fname,
                                        count_log_files=[conf_count_fname],
                                        count_timeseries_list=[self.conf_num_in_sec],
                                        mode_list=['conf'],
                                        is_extra=self.is_extra,
                                        is_extend=self.is_extend,
                                        duration_log_files=[None], 
                                        duration_period_log_files=[None], output_extend=self.is_tactical_extend)
                    
                self.is_tactical = False
                self.is_extra = False
                return True, f'USEPE_ML with TACTICAL is disabled.'

            # TACTICAL EXTEND ===========
            if self.is_tactical is True and self.is_extra is False and self.is_tactical_extend is True:
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'

                    self.summarize_logger()
                    self.summarize_with_extend_logger()
                    output_loggers = loggers + tactical_ex_loggers 

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()

                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    # Get los count 
                    los_count_fname = allloggers[tactical_ex_loggers[0]].fname
                    simt_remover(los_count_fname)

                    # Get conf duration csv
                    conf_duration_fname = allloggers[tactical_ex_loggers[1]].fname
                    simt_remover(conf_duration_fname)

                    # Get los duration csv
                    los_duration_fname = allloggers[tactical_ex_loggers[2]].fname
                    simt_remover(los_duration_fname)

                    # Get conf duration plot
                    conf_duration_period_fname = allloggers[tactical_ex_loggers[3]].fname
                    simt_remover(conf_duration_period_fname)

                    # Get los duration plot
                    los_duration_period_fname = allloggers[tactical_ex_loggers[4]].fname
                    simt_remover(los_duration_period_fname)

                    tactical_pairwise_spatial_conf_csv(log_fname, self.conf_num_in_sec) 
                    tactical_local_spatial_conf_csv(log_fname, self.conf_num_in_sec)
                    tactical_general_spatial_conf_csv(log_fname, self.conf_num_in_sec)

                    output_tactical_log(log_file=log_fname,
                                        count_log_files=[conf_count_fname],
                                        count_timeseries_list=[self.conf_num_in_sec],
                                        mode_list=['conf', 'los'],
                                        is_extra=self.is_extra,
                                        is_extend=self.is_extend,
                                        duration_log_files = [conf_duration_fname, los_duration_fname], 
                                        duration_period_log_files = [conf_duration_period_fname, los_duration_period_fname], output_extend=self.is_tactical_extend)

                self.is_tactical = False
                self.is_tactical_extend = False
                self.is_extend = False
                return True, f'USEPE_ML with TACTICAL is disabled.'

           # TACTICAL EXTEND + EXTRA ===========
            if self.is_tactical is True and self.is_extra is True and self.is_tactical_extend is True:
                if allloggers[loggers[0]].isopen():
                    if self.start_traf_num == 0:
                        return True, f'USEPE_ML is disabled.'

                    self.summarize_logger()
                    self.summarize_with_extend_logger()
                    output_loggers = loggers + tactical_ex_loggers 

                    # Close I/O stream
                    for log_name in output_loggers:
                        allloggers[log_name].file.close()

                    # Get src Log name
                    log_fname = allloggers[loggers[0]].fname
                    simt_remover(log_fname)

                    # Get conf count 
                    conf_count_fname = allloggers[loggers[1]].fname
                    simt_remover(conf_count_fname)

                    # Get los count 
                    los_count_fname = allloggers[tactical_ex_loggers[0]].fname
                    simt_remover(los_count_fname)

                    # Get conf duration csv
                    conf_duration_fname = allloggers[tactical_ex_loggers[1]].fname
                    simt_remover(conf_duration_fname)

                    # Get los duration csv
                    los_duration_fname = allloggers[tactical_ex_loggers[2]].fname
                    simt_remover(los_duration_fname)

                    # Get conf duration plot
                    conf_duration_period_fname = allloggers[tactical_ex_loggers[3]].fname
                    simt_remover(conf_duration_period_fname)

                    # Get los duration plot
                    los_duration_period_fname = allloggers[tactical_ex_loggers[4]].fname
                    simt_remover(los_duration_period_fname)


                    tactical_pairwise_spatial_conf_csv(log_fname, self.conf_num_in_sec) 
                    tactical_local_spatial_conf_csv(log_fname, self.conf_num_in_sec)
                    tactical_general_spatial_conf_csv(log_fname, self.conf_num_in_sec)

                    tactical_pairwise_all_conf_csv(log_fname, self.conf_num_in_sec)
                    tactical_local_all_csv(log_fname, self.conf_num_in_sec)
                    tactical_general_all_csv(log_fname, self.conf_num_in_sec)

                    output_tactical_log(log_file=log_fname,
                                        count_log_files=[conf_count_fname, los_count_fname],
                                        count_timeseries_list=[self.conf_num_in_sec, self.los_num_in_sec],
                                        mode_list=['conf', 'los'],
                                        is_extra=self.is_extra,
                                        is_extend=self.is_extend,
                                        duration_log_files = [conf_duration_fname, los_duration_fname], 
                                        duration_period_log_files = [conf_duration_period_fname, los_duration_period_fname], output_extend=self.is_tactical_extend)

                self.is_tactical = False
                self.is_tactical_extend = False
                self.is_extend = False
                self.is_extra = False
                return True, f'USEPE_ML with TACTICAL is disabled.'

            # Reset step ------------------------------------------------------
            self.reset()
            self.is_tactical = False
            self.is_tactical_extend = False
            self.is_extra = False
            self.is_extend = False
            self.is_strategic = False
            self.use_debug = False
        # -----------------------------------------------------------------------------------------------------------------------------
        elif cmd == "USEPE":
            return True, f"""
            **********************************************
            Jorge Bueno Gómez
            Gone from our sight, but never from our hearts
            Rest in Peace
            **********************************************
            This plugin is part of a project (namely USEPE: U-SPACE SEPARATION IN EUROPE) that has received funding from the SESAR Joint Undertaking under grant agreement No 890378 under European Union’s Horizon 2020 research and innovation programme.
            For more information: website: https://usepe.eu/ LinkedIn: https://www.linkedin.com/company/79835160/admin/
            ===========
            The USEPE_ML plugin aims to add machine learning capabilities to the available functionalities in USEPE implementations, BlueSky and its plugins.
            Although the main goal is to cover U-Space, it has flexibility to cover any simulations are done by BlueSky.
            Additionally, many analysis are printed for simulation activities.
            ===========
            Contacts for USEPE Project:
            Project Coordinator, Esther Nistal Cabanas, Enistal@isdefe.es
            Communication Leader, Aurilla Aurelie Arntzen, Aurilla.Aurelie.Arntzen@usn.no
            Communication Support, Manon Coyne, Mcoyne@polisnetwork.eu
            ===========
            Contacts for plugin developers:
            Serkan Güldal, SrknGldl@hotmail.com
            Rina Komatsu, rinakomatsu2021@gmail.com
            """

        elif cmd == "HELP" or cmd == "help" or cmd == "H" or cmd == "h":
            if cmd2 == "HELP" or cmd2 == "help" or cmd2 == "H" or cmd2 == "h":
                return True, f"""
                                HELP command provides information about the available commands with examples.
                                Usage : USEPE_ML HELP/Help/H/h
                                Where : User Interface (GUI)
                                Output: This command lists the available commands in USEPE_ML with brief explanations.

                                Usage : USEPE_ML HELP BASIC/OFF/EXTEND/EXTRA/STRATEGIC/TACTICAL/DEBUG/USEPE/HELP
                                Where : User Interface (GUI)
                                Output: The detailed information about the specified command
                                """
            
            elif cmd2 == "BASIC":
                return True, f"""
                                BASIC activates the USEPE_ML's basic functionalities.
                                Usage:  00:00:00.00> USEPE_ML BASIC
                                Where:  Scenario File
                                The List of Outputs
                                - Pairwise Separation Score with the Number of Conflicts ......: CSV and PNG
                                - Single General Separation Score with the Number of Conflicts.: CSV and PNG
                                - Single Local Separation Score with the Number of Conflicts...: CSV and PNG
                                """

            elif cmd2 == "OFF":
                return True, f"""
                                OFF deactivates the USEPE_ML plugin.
                                Usage : 00:00:00.00> USEPE_ML OFF
                                Where : Scenario File
                                Output: This terminates the activities of USEPE_ML.
                                ....... The activities' outputs are printed to CSV and PNG files.
                                """

            elif cmd2 == "EXTEND":
                return True, f"""
                                EXTEND activates broad analysis of the completed simulation.
                                Usage : 00:00:00.00> USEPE_ML EXTEND
                                Where : Scenario File
                                The List of Outputs
                                - Pairwise Separation Score with the Number of Conflicts.......: CSV and PNG
                                - Pairwise Separation Score with the Number of LoS.............: CSV and PNG
                                - Pairwise Separation Score with the Duration of Conflicts.....: CSV and PNG
                                - Pairwise Separation Score with the Duration of LoS...........: CSV and PNG
                                - Single General Separation Score with the Number of Conflicts.: CSV and PNG
                                - Single General Separation Score with the Number of LoS.......: CSV and PNG
                                - Single Local Separation Score with the Number of Conflicts...: CSV and PNG
                                - Single Local Separation Score with the Number of LoS.........: CSV and PNG
                                - Simulation Summary (Heatmap).................................: PNG
                                """

            elif cmd2 == "EXTRA":
                return True, f"""
                                EXTRA activates machine learning analysis beyond spatial features.
                                The list variable used in machine learning computations:
                                lat, lon, alt, distflown, hdg, trk, tas, gs, gsnorth, gseast, cas, M, selspd, aptas, selalt
                                Usage : 00:00:00.00> USEPE_ML EXTRA
                                Where : Scenario File
                                Output: Adds additional outputs to BASIC and EXTEND named All Features.
                                """

            elif cmd2 == "STRATEGIC":
                return True, f"""
                                STRATEGIC activates strategic analysis for scenario files.
                                Usage : 00:00:00.00> USEPE_ML STRATEGIC/scenario file 1/scenario file 2/...
                                Where : Scenario File
                                The List of Outputs:
                                - Pairwise Separation Score........: CSV and PNG
                                - Single General Separation Score .: CSV and PNG
                                - Single Local Separation Score ...: CSV and PNG
                                """

            elif cmd2 == "TACTICAL":
                return True, f"""
                                TACTICAL activates tactical analysis.
                                Usage : 00:00:00.00> USEPE_ML TACTICAL
                                Where : Scenario File
                                The List of Outputs:
                                - Pairwise Separation Score with the Number of Conflicts....: CSV and PNG
                                - Average Separation Score and Number of Aircrafts  vs Time.: CSV and PNG

                                TACTICAL EXTEND activates tactical analysis.
                                Usage : 00:00:00.00> USEPE_ML TACTICAL EXTEND
                                Where : Scenario File
                                The List of Outputs:
                                - Usage : 00:00:00.00> USEPE_ML TACTICAL
                                Where : Scenario File
                                The List of Outputs:
                                - Pairwise Separation Score with the Number of Conflicts...: CSV and PNG
                                - Pairwise Separation Score with the Number of LoS.........: CSV and PNG
                                - Pairwise Separation Score with the Duration of Conflicts.: CSV and PNG
                                - Pairwise Separation Score with the Duration of LoS.......: CSV and PNG
                                - Simulation Summary (Heatmap).............................: PNG
                                - Average Separation Score and Number of Aircrafts vs Time.: CSV and PNG
                                - .....
                                """

            elif cmd2 == "USEPE":
                return True, f"""
                                USEPE presents the information about the USEPE project and the USEPE_ML.
                                Usage : USEPE_ML USEPE
                                Where : User Interface (GUI)
                                The List of Outputs:
                                - The explanatory informatoin.: GUI 
                                """

            elif cmd2 == "DEBUG":
                return True, f"""
                                DEBUG colloct information about the installation of python libraries, BlueSky version, and scnario file.
                                Usage : USEPE_ML DEBUG
                                Where : User Interface (GUI)
                                The List of Outputs:
                                - Diagnostic of the installation and simulation.: LOG
                                """

            else:                
                return True,  f"""
                                HELP/help/H/h presents this information.
                                HELP COMMAND presents the detailed information about COMMAND.
                                BASIC activates the USEPE_ML plugin with basic functionalities.
                                OFF finishes all the activated functions and prints out the results.
                                EXTEND activates broad analysis of the completed simulation.
                                EXTRA activates machine learning analysis beyond spatial features.
                                STRATEGIC activates strategic analysis for scenario files.
                                TACTICAL activates tactical analysis summary.
                                TACTICAL EXTEND activates tactical analysis summary and detailed separation information of each aircraft.
                                DEBUG generates diagnostics of the installation and simulation setup.
                                USEPE presents about the plugin.
                                """

        else:
            return True, f"Invalid argument! \n For available commands: USEPE_ML HELP"

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Remove # simt ------ input: file name and the line number of #simt
def simt_remover(file_name):
    lines = []
    # Read file
    with open(file_name, 'r') as fp:
        lines = fp.readlines()

    # Write file, everything but "# simt"
    with open(file_name, 'w') as fp:
        for line in lines:
            if "# simt," in line:  # It is for header
                fp.write(line)
            elif "# simt:" in line:  # It is for definition
                fp.write(line)
            elif "# simt" not in line:  # It is wrong "# simt"
                fp.write(line)

# Output result function -----------------------------------------------------------------

# Number of conflicts per aircraft
def single_conf_num(count_fname):
    if count_fname is False:
        single_conf_count_dic = False
    else:
        count_d = data_import(count_fname, 3)
        count_dic = output_pair_dic(count_d, ' confNum')

        single_conf_count_dic = {}

        for i in count_dic:
            aircraft_a, aircraft_b = i.split('&')
            if aircraft_a not in single_conf_count_dic:
                single_conf_count_dic[aircraft_a] = 0
            if aircraft_b not in single_conf_count_dic:
                single_conf_count_dic[aircraft_b] = 0

            single_conf_count_dic[aircraft_a] = single_conf_count_dic[aircraft_a] + count_dic[i]
            single_conf_count_dic[aircraft_b] = single_conf_count_dic[aircraft_b] + count_dic[i]
    return single_conf_count_dic

# Number of LoS (Loss of Separation) per aircraft
def single_los_num(count_fname):

    count_d = data_import(count_fname, 3)
    count_dic = output_pair_dic(count_d, ' losNum')
    single_los_count_dic = {}

    for i in count_dic:
        aircraft_a, aircraft_b = i.split('&')

        if aircraft_a not in single_los_count_dic:
            single_los_count_dic[aircraft_a] = 0
        if aircraft_b not in single_los_count_dic:
            single_los_count_dic[aircraft_b] = 0

        single_los_count_dic[aircraft_a] = single_los_count_dic[aircraft_a] + count_dic[i]
        single_los_count_dic[aircraft_b] = single_los_count_dic[aircraft_b] + count_dic[i]
    return single_los_count_dic

# This generates the list of ids from count log files
def id_list(log_fname):
    d = data_import(log_fname, 1)
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    return unique_ids

# This generates the list of secs from log files
def sec_list(log_fname):
    d = data_import(log_fname, 1)
    secs = list(d['# simt'])[1:]
    return secs

def aircraft_num_in_sec_list(log_fname):
    '''This collects the aircraft IDs for every second. Output: dictionry[sec][id]'''
    d = data_import(log_fname, 1)
    # Extract IDs
    ids = list(d[' id'])
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    secs = list(d['# simt'])
    # Get aircrafts in second
    aircrafts_in_sec = dict() # sec: [ids]
    for idx, (id, sec) in enumerate(zip(ids, secs)):
        sec = float(sec)
        if sec not in aircrafts_in_sec.keys():
            aircrafts_in_sec[sec] = []
        if id not in aircrafts_in_sec[sec]:
            aircrafts_in_sec[sec].append(id)
    return aircrafts_in_sec

# Separation score for spatial features
def sep_pair_spatial(log_fname):
    ids = id_list(log_fname)
    d = data_import(log_fname, 1)
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {} # {pair_name0 : sep_score0, pair_name1 : sep_score1, ...}
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    
    for idx, pair in enumerate(itertools.combinations(ids, 2)):
        aircraft_a, aircraft_b = pair
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
            # Get separation score in seconds
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = {}
            sec_separation_dic[sec][key] = get_sep_score(aircraft_a_data[j].reshape(1, -1), aircraft_b_data[j].reshape(1, -1))
        # Get separation score
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score

    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))

    return separation_dic

def sec_sep_pair_spatial(log_fname):
    ids = id_list(log_fname)
    d = data_import(log_fname, 1)
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {} # {pair_name0 : sep_score0, pair_name1 : sep_score1, ...}
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    
    for idx, pair in enumerate(itertools.combinations(ids, 2)):
        aircraft_a, aircraft_b = pair
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
            # Get separation score in seconds
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = {}
            sec_separation_dic[sec][key] = get_sep_score(aircraft_a_data[j].reshape(1, -1), aircraft_b_data[j].reshape(1, -1))
        # Get separation score
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score

    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))

    return separation_dic, sec_separation_dic

# Separation score for all features
def sep_pair_all(log_fname):
    ids = id_list(log_fname)
    d = data_import(log_fname, 1)
    separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
    separation_dic = {} # {pair_name0 : sep_score0, pair_name1 : sep_score1, ...}
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    
    for idx, pair in enumerate(itertools.combinations(ids, 2)):
        aircraft_a, aircraft_b = pair
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
            # Get separation score in sec
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = {}
            sec_separation_dic[sec][key] = get_sep_score(aircraft_a_data[j].reshape(1, -1), aircraft_b_data[j].reshape(1, -1))
        # Get separation score
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score

    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))

    return separation_dic

# Separation score for all features
def sec_sep_pair_all(log_fname):
    ids = id_list(log_fname)
    d = data_import(log_fname, 1)
    separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
    separation_dic = {} # {pair_name0 : sep_score0, pair_name1 : sep_score1, ...}
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    
    for idx, pair in enumerate(itertools.combinations(ids, 2)):
        aircraft_a, aircraft_b = pair
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
            # Get separation score in sec
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = {}
            sec_separation_dic[sec][key] = get_sep_score(aircraft_a_data[j].reshape(1, -1), aircraft_b_data[j].reshape(1, -1))
        # Get separation score
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score

    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))

    return separation_dic, sec_separation_dic

# Single local seaparation score for spatial features
def sep_local_spatial(log_fname):
    ids = id_list(log_fname)
    sep = sep_pair_spatial(log_fname)
    sep_min = list(sep.values())[-1]
    local_scores = {}

    for id in ids:
        scores = {}
        for pair_name in sep.keys():            
            if id in pair_name:
                scores[pair_name]=sep[pair_name]

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)) # sorted list of pair scores

        i=1
        while i <= len(scores) and list(scores.values())[-i] <= 10*sep_min:
            i += 1

        nearby_values = list(scores.values())[-i:] # The lowest scored pairs for specified aircraft.
        nearby_values = np.array(nearby_values)
        local_scores[id] = np.average(nearby_values)

    local_scores = dict(sorted(local_scores.items(), key=lambda x: x[1], reverse=True))

    return local_scores

# Single local seaparation score for spatial features
def sec_sep_local_spatial(log_fname):
    ids = id_list(log_fname)
    sep, sec_sep = sec_sep_pair_spatial(log_fname)

    sec_scores = {} # {sec:{id:score}}
    for sec in sec_sep.keys():
        sec_scores[sec] = dict()
        for id in ids:
            local_value = float("inf")
            for pair_name in sec_sep[sec].keys():
                if id in pair_name:
                    if local_value > sec_sep[sec][pair_name]:
                        local_value = sec_sep[sec][pair_name]
            sec_scores[sec][id] = local_value

    return sec_scores

# Single local seaparation score for all features
def sep_local_all(log_fname):
    ids = id_list(log_fname)
    sep = sep_pair_all(log_fname)
    sep_min = list(sep.values())[-1]
    local_scores = {}

    for id in ids:
        scores = {}
        for pair_name in sep.keys():            
            if id in pair_name:
                scores[pair_name]=sep[pair_name]

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)) # sorted list of pair scores

        i = 1
        while i <= len(scores) and list(scores.values())[-i] <= 3*sep_min:            
            i += 1

        nearby_values = list(scores.values())[-i:] # The lowest scored pairs for specified aircraft, up to 5.
        nearby_values = np.array(nearby_values)
        local_scores[id] = np.average(nearby_values)

    local_scores = dict(sorted(local_scores.items(), key=lambda x: x[1], reverse=True))

    return local_scores

# Single local seaparation score for all features
def sec_sep_local_all(log_fname):
    ids = id_list(log_fname)
    sep, sec_sep = sec_sep_pair_all(log_fname)

    sec_scores = {} # {sec:{id:score}}
    for sec in sec_sep.keys():
        sec_scores[sec] = dict()
        for id in ids:
            local_value = float("inf")
            for pair_name in sec_sep[sec].keys():
                if id in pair_name:
                    if local_value > sec_sep[sec][pair_name]:
                        local_value = sec_sep[sec][pair_name]
            sec_scores[sec][id] = local_value

    return sec_scores

# Single general seaparation score for spatial features
def sep_general_spatial(log_fname):
    ids = id_list(log_fname)
    sep = sep_pair_spatial(log_fname)

    scores = {}
    sep_median = np.median(list(sep.values()))
    general_scores = {}

    for id in ids:
        scores = {}
        for pair_name in sep.keys():
            if id in pair_name:
                scores[pair_name]=sep[pair_name]

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)) # sorted list of pair scores

        i = 1
        while i <= len(scores) and list(scores.values())[-i] <= sep_median:            
            i += 1

        nearby_values = list(scores.values())[-i:] # The lowest scored pairs for specified aircraft, up to 5.
        nearby_values = np.array(nearby_values)
        general_scores[id] = np.average(nearby_values)

    # Sort
    general_scores = dict(sorted(general_scores.items(), key=lambda x: x[1], reverse=True))

    return general_scores

# Single general seaparation score for spatial features
def sec_sep_general_spatial(log_fname):
    ids = id_list(log_fname)
    sep, sec_sep = sec_sep_pair_spatial(log_fname)

    sec_scores = {} # {sec:{id:score}
    for sec in sec_sep.keys():
        sec_scores[sec] = dict()
        for id in ids:
            sum = 0
            for pair_name in sec_sep[sec].keys():
                if id in pair_name:
                    sum += sec_sep[sec][pair_name]/len(ids)

            sec_scores[sec][id] = sum

    return sec_scores

# Single general seaparation score for all features
def sep_general_all(log_fname):
    ids = id_list(log_fname)
    sep = sep_pair_all(log_fname)
    scores = {}
    sep_median = np.median(list(sep.values()))
    general_scores = {}

    for id in ids:
        scores = {}
        for pair_name in sep.keys():
            if id in pair_name:
                scores[pair_name]=sep[pair_name]

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)) # sorted list of pair scores

        i = 1
        while i <= len(scores) and list(scores.values())[-i] <= sep_median:            
            i += 1

        nearby_values = list(scores.values())[-i:] # The lowest scored pairs for specified aircraft, up to 5.
        nearby_values = np.array(nearby_values)
        general_scores[id] = np.average(nearby_values)

    # Sort
    general_scores = dict(sorted(general_scores.items(), key=lambda x: x[1], reverse=True))
    return general_scores

# Single general seaparation score for all features
def sec_sep_general_all(log_fname):
    ids = id_list(log_fname)
    sep, sec_sep = sec_sep_pair_all(log_fname)

    sec_scores = {} # {sec:{id:score}
    for sec in sec_sep.keys():
        sec_scores[sec] = dict()
        for id in ids:
            sum = 0
            for pair_name in sec_sep[sec].keys():
                if id in pair_name:
                    sum += sec_sep[sec][pair_name]/len(ids)

            sec_scores[sec][id] = sum

    return sec_scores

def output_csv_and_plot_with_count_log(log_file, count_log_files, mode_list=['conf'], is_extra=False):
    output_single_score = True
    # Set use_entire_content
    if is_extra is False:
        use_entire_content_list = [False]
    else:
        use_entire_content_list = [False, True]

    for i, (mode, count_log) in enumerate(zip(mode_list, count_log_files)):
        for j, use_entire_content in enumerate(use_entire_content_list):
            # if j > 0 or i > 0:
            #     output_single_score = False
            # Output csv & plot
            output_csv_from_count_dic(log_file, count_log, mode=mode, use_entire_content=use_entire_content, output_single_score=output_single_score)
            output_plots_from_count_dic(log_file, count_log, mode=mode, use_entire_content=use_entire_content)
    # Output
    if len(count_log_files) == 2:
        for use_entire_content in use_entire_content_list:
            output_heatmap_from_conf_and_los_dic(log_file, count_log_files[0], count_log_files[1], use_entire_content=use_entire_content)

def output_csv_and_plot_with_duration_log(log_file, duration_log_files, duration_period_log_files,
                                          mode_list=['conf', 'los'], is_extra=False):
    # Set use_entire_content
    if is_extra is False:
        use_entire_content_list = [False]
    else:
        use_entire_content_list = [False, True]

    for mode, duration_log, duration_period_log in zip(mode_list, duration_log_files, duration_period_log_files):
        for use_entire_content in use_entire_content_list:
            output_csv_from_duration_dic(log_file, duration_log, mode=mode, use_entire_content=use_entire_content)
            output_plots_from_duration(log_file, duration_period_log, mode=mode, use_entire_content=use_entire_content)

def output_tactical_log(log_file,
                        count_log_files,
                        count_timeseries_list,
                        mode_list=['conf'],
                        is_extra=False,
                        is_extend=False,
                        duration_log_files=[None],
                        duration_period_log_files=[None], 
                        output_extend=False):
                        
    output_single_score = is_extend
    # Set use_entire_content
    if is_extra is False:
        use_entire_content_list = [False]

    else:
        use_entire_content_list = [False, True]

    for i, (mode, count_log, count_timeseries, duration_log, duration_period_log) in enumerate(zip(mode_list, count_log_files, count_timeseries_list, duration_log_files, duration_period_log_files)):
        for j, use_entire_content in enumerate(use_entire_content_list):
           # output_tactical_csv(log_file, count_log, mode, count_timeseries, duration_fname=duration_period_log, use_entire_content=use_entire_content, output_single_score=output_single_score)
            output_tactical_plot(log_file, count_log, mode, count_timeseries, duration_period_log, use_entire_content=use_entire_content, output_extend=output_extend)
# ----------------------------------------------------------------------------------------
# Data Control Content -------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Data importer function
def data_import(file_name, header_line):
    # Opening the file
    file = open(file_name)
    # Creates the dataframe
    d = pd.read_csv(file, header=header_line, dtype=object)
    return(d)

# Exporter function for ML result
def data_export(file):
    file_name = (file[:-4] + '.csv')
    return(file_name)

# Generate aircraft pair name dic
def output_pair_dic(data, col_name):
    output_dic = {}
    for (aircraft_a, aircraft_b, counter) in zip(data[' Aircraft1'], data[' Aircraft2'], data[col_name]): 
        if isinstance(aircraft_a, str):
            if '[s]' not in col_name:
                output_dic['{}&{}'.format(aircraft_a, aircraft_b)] = int(counter)
            else:
                if col_name == ' Duration Time[s]':
                    output_dic['{}&{}'.format(aircraft_a, aircraft_b)] = float(counter)
                else:
                    # Get start & end duration
                    if '{}&{}'.format(aircraft_a, aircraft_b) not in output_dic.keys():
                        output_dic['{}&{}'.format(aircraft_a, aircraft_b)] = [float(counter)]
                    else:
                        output_dic['{}&{}'.format(aircraft_a, aircraft_b)].append(float(counter))
        elif math.isnan(aircraft_a):
            continue
    return output_dic
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Calc sep Score Content -----------------------------------------------------------------
# ----------------------------------------------------------------------------------------
METRICS = ['Euclidean distance']   # more metrics can be added
# Output Sep Score
def get_sep_score(X, Y):
    output_dic = {}
    for idx, m in enumerate(METRICS):
        # Metric method is obtained from
        # https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
        sep = 0
        if idx == 0:
            sep = metrics.pairwise.euclidean_distances(X, Y)
        output_dic[m] = np.average(sep)

    return output_dic['Euclidean distance']

# Output Min-Max Norm
def get_norm_MinMax_sep_score(sep_dic):
    score_max = max(list(sep_dic.values()))
    score_min = min(list(sep_dic.values()))
    for k, v in sep_dic.items():
        if score_max != 0:
            sep_dic[k] = v / score_max
        else:
            sep_dic[k] = 0
    return sep_dic
# Output General Sep Score
def get_single_general_aircraft_sep_score(sep_dic, unique_ids): # sep_dic= {a&b:score}
    output_dic = dict()
    for id in unique_ids:
        sum = 0
        for pair_name in sep_dic.keys():
            if id in pair_name:
                # print("{} extracted in {}:{}".format(id, pair_name, sep_dic[pair_name]))
                sum += sep_dic[pair_name]/len(unique_ids)
        output_dic[id] = sum
    # Sort
    output_dic = dict(sorted(output_dic.items(), key=lambda x: x[1], reverse=True))
    return output_dic
# Output Local Score
def get_single_local_aircraft_sep_score(sep_dic, unique_ids): # sep_dic= {a&b:score}
    output_dic = dict()
    for id in unique_ids:
        local_value = float("inf")
        for pair_name in sep_dic.keys():
            if id in pair_name:
                # print("{} extracted in {}:{}".format(id, pair_name, sep_dic[pair_name]))
                if local_value > sep_dic[pair_name]:
                    local_value = sep_dic[pair_name]
        output_dic[id] = local_value
    # Sort
    output_dic = dict(sorted(output_dic.items(), key=lambda x: x[1], reverse=True))
    return output_dic

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Output CSV sep result ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def count_dic_to_csv(file_name, conf_dic, sep_dic, mode='conf'):
    if mode == 'conf':
        mode_name = 'Conflict'
    elif mode == 'los':
        mode_name = 'LoS'
    # Define column of csv
    csv_column = ['Index', 'Aircraft1', 'Aircraft2',
                  'Number of {}'.format(mode_name), 'Number of {} (Normalized)'.format(mode_name),
                  'Separation Score', 'Separation Score (Normalized)']
    # Norm min-max
    norm_sep_dic = get_norm_MinMax_sep_score(sep_dic.copy())
    norm_conf_dic = get_norm_MinMax_sep_score(conf_dic.copy())

    csv_log = []
    for idx, pair_aircraft_name in enumerate(sep_dic.keys()):
        # Get score
        sep_score = sep_dic[pair_aircraft_name]
        norm_sep_score = norm_sep_dic[pair_aircraft_name]
        # Get conf num
        conf_count = conf_dic[pair_aircraft_name]
        norm_conf_count = norm_conf_dic[pair_aircraft_name]
        # Get aircraft names
        aircraft_a, aircraft_b = pair_aircraft_name.split('&')
        # Convert 0000 -> ="0000" for csv output
        if (aircraft_a[0] == str(0)):
            aircraft_a = '="'+aircraft_a+'"'
        if (aircraft_b[0] == str(0)):
            aircraft_b = '="'+aircraft_b + '"'
        # Append
        csv_log.append([idx+1, aircraft_a, aircraft_b,
                        conf_count, norm_conf_count,
                        sep_score, norm_sep_score])

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

def pair_aircraft_dic_to_csv(file_name, sep_dic, conf_dic=None, mode=None):
    csv_column = None
    if conf_dic is None:
        # Define column of csv
        csv_column = ['Index', 'Aircraft1', 'Aircraft2', 'Separation Score', 'Separation Score (Normalized)']
    else:
        mode_name = None
        if mode == 'conf':
            mode_name = 'Conflict'
        elif mode == 'los':
            mode_name = 'LoS'
        csv_column = ['Index', 'Aircraft1', 'Aircraft2', 'Number of {}'.format(mode_name), 'Number of {} (Normalized)'.format(mode_name), 'Separation Score', 'Separation Score (Normalized)']
    # Norm min-max
    norm_sep_dic = get_norm_MinMax_sep_score(sep_dic.copy())
    norm_conf_dic = None
    if conf_dic is not None:
        norm_conf_dic = get_norm_MinMax_sep_score(conf_dic.copy())
    csv_log = []
    for idx, pair_aircraft_name in enumerate(sep_dic.keys()):
        # Get score
        sep_score = sep_dic[pair_aircraft_name]
        norm_sep_score = norm_sep_dic[pair_aircraft_name]
        # Get conf norm
        conf_num, norm_conf_num = None, None
        if conf_dic is not None:
            conf_num = conf_dic[pair_aircraft_name]
            norm_conf_num = norm_conf_dic[pair_aircraft_name]
        # Get aircraft names
        aircraft_a, aircraft_b = pair_aircraft_name.split('&')
        # Convert 0000 -> ="0000" for csv output
        # if (aircraft_a[0] == str(0)):
        #    aircraft_a = '="'+aircraft_a+'"'
        # if (aircraft_b[0] == str(0)):
        #    aircraft_b = '="'+aircraft_b+ '"'
        # Append
        if conf_dic is None:
            csv_log.append([idx+1, aircraft_a, aircraft_b, sep_score, norm_sep_score])
        else:
            csv_log.append([idx+1, aircraft_a, aircraft_b, conf_num, norm_conf_num, sep_score, norm_sep_score])


    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

# [For STRATEGIC] Generate CSV: single_general_aircraft_sep_score_dic_to_csv (Calc from Entire Time)
def single_general_aircraft_sep_score_dic_to_csv(file_name, sep_dic, unique_ids):
    # Get each aircraft separation score in local
    lower_sep_dic = dict(sorted(sep_dic.items(), key=lambda x: x[1], reverse=True))
    single_general_aircraft_score_dic = get_single_general_aircraft_sep_score(sep_dic, unique_ids)
    norm_single_general_aircraft_score_dic = get_norm_MinMax_sep_score(single_general_aircraft_score_dic.copy())

    wrote_id_list = []
    csv_log = []
    for idx, pair_name in enumerate(lower_sep_dic.keys()):
        drone_a, drone_b = pair_name.split('&')
        if drone_a not in wrote_id_list:
            wrote_id_list.append(drone_a)
        if drone_b not in wrote_id_list:
            wrote_id_list.append(drone_b)

    for i, id in enumerate(wrote_id_list):
        score = single_general_aircraft_score_dic[id]
        norm_score = norm_single_general_aircraft_score_dic[id]

        csv_column = ['Index', 'Aircraft ID', 'General Separation Score', 'General Separation Score (Normalized)']
        csv_log.append([i+1, id, score, norm_score])

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('General Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(wrote_id_list)+1)
        df.to_csv(file_name, index=False, encoding="cp932")
        
# [For STRATEGIC] Generate CSV: single_local_aircraft_sep_score_dic_to_csv (Calc from Entire Time)
def single_local_aircraft_sep_score_dic_to_csv(file_name, sep_dic, unique_ids):
    # Get each aircraft separation score in local
    lower_sep_dic = dict(sorted(sep_dic.items(), key=lambda x: x[1], reverse=True))
    single_local_aircraft_score_dic = get_single_local_aircraft_sep_score(sep_dic, unique_ids)
    norm_single_local_aircraft_score_dic = get_norm_MinMax_sep_score(single_local_aircraft_score_dic.copy())

    wrote_id_list = []
    csv_log = []
    for idx, pair_name in enumerate(lower_sep_dic.keys()):
        drone_a, drone_b = pair_name.split('&')
        if drone_a not in wrote_id_list:
            wrote_id_list.append(drone_a)
        if drone_b not in wrote_id_list:
            wrote_id_list.append(drone_b)

    for i, id in enumerate(wrote_id_list):
        score = single_local_aircraft_score_dic[id]
        norm_score = norm_single_local_aircraft_score_dic[id]
        csv_column = ['Index', 'Aircraft ID', 'Local Separation Score', 'Local Separation Score (Normalized)']
        csv_log.append([i+1, id, score, norm_score])

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('Local Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(wrote_id_list)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

def tactical_local_spatial_conf_csv(log_fname, count_in_sec):

    # Load log
    log_name = os.path.basename(log_fname)[:-4]

    print("Tactical Analysis, Single Local Separation with Number of Conflicts for {}.csv is started.".format(log_name))

    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])

    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get separation score
    sep_score_in_sec = sec_sep_local_spatial(log_fname) # sep score -> id: score, sep_score_in_sec -> sec:{id: score}

    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))

    csv_log = []
    csv_column = ['Time', "Aircraft ID", 'Local Separation Score', 'Local Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"]

    for sec in secs:
        aircraft_num = len(aircraft_num_in_sec[sec])
        aircraft_list = aircraft_num_in_sec[sec]
        # Get separation score in sec
        sep_score_list = list(sep_score_in_sec[sec].values())
        sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
        sec_max_sep = max(sep_score_list)
        sec_avg_sep = sum(sep_score_list) / aircraft_num
        # Get conflict num in sec
        count_num_total = len(count_in_sec[sec])
        norm_count_num_total = count_num_total / max_conf_num if max_conf_num != 0 else 0

        # Number of conflicts
        conf = {}
        for id in aircraft_list:
            conf[id] = 0
            for pair in count_in_sec[sec]:
                if id in pair:
                    conf[id] += 1

        # Normalized conflict
        norm_conf = {}
        for id in aircraft_list:
            norm_conf[id] = 0
            if  max(conf.values()) != 0:
                norm_conf[id] = conf[id] / max(conf.values())

        # Separation score
        for id in aircraft_list:
            sep_score_aircraft = sep_score_in_sec[sec][id]
            norm_sep_score_aircraft = sep_score_aircraft / sec_max_sep

            # Append information to output
            csv_log.append([sec, id, sep_score_aircraft, norm_sep_score_aircraft, conf[id], norm_conf[id]])

        # Summary for the specified second
        csv_log.append(['===============================================================','','','','',''])
        csv_log.append(['Time', "Number of Aircraft", 'Average Separation Score', 'Average Separation Score (Normalized)','Number of Conflicts', 'Number of Conflicts (Normalized)'])
        csv_log.append([sec, aircraft_num, sec_avg_sep, sec_avg_sep/sec_max_sep, count_num_total, norm_count_num_total])
        csv_log.append(['===============================================================','','','','',''])

        # Header for the next second
        csv_log.append(['Time', 'Aircraft ID', 'Local Separation Score', 'Local Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"])

    output_csv_dir = os.path.dirname(log_fname)

    file_name = os.path.join(output_csv_dir,"USEPE_ML Tactical Single Spatial Features Local Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Tactical Analysis, Single Local Separation with Number of Conflicts for {}.csv is completed.".format( log_name))
    print("Tactical Analysis, Output file: {} \n".format(file_name))

def tactical_local_all_csv(log_fname, count_in_sec):

    # Load log
    log_name = os.path.basename(log_fname)[:-4]

    print("Tactical Analysis, Single Local Separation with Number of Conflicts for {}.csv is started.".format(log_name))

    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])

    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get separation score
    sep_score_in_sec = sec_sep_local_all(log_fname) # sep score -> id: score, sep_score_in_sec -> sec:{id: score}

    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))

    csv_log = []
    csv_column = ['Time', "Aircraft ID", 'Local Separation Score', 'Local Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"]

    for sec in secs:
        aircraft_num = len(aircraft_num_in_sec[sec])
        aircraft_list = aircraft_num_in_sec[sec]
        # Get separation score in sec
        sep_score_list = list(sep_score_in_sec[sec].values())
        sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
        sec_max_sep = max(sep_score_list)
        sec_avg_sep = sum(sep_score_list) / aircraft_num
        # Get conflict num in sec
        count_num_total = len(count_in_sec[sec])
        norm_count_num_total = count_num_total / max_conf_num if max_conf_num != 0 else 0

        # Number of conflicts
        conf = {}
        for id in aircraft_list:
            conf[id] = 0
            for pair in count_in_sec[sec]:
                if id in pair:
                    conf[id] += 1

        # Normalized conflict
        norm_conf = {}
        for id in aircraft_list:
            norm_conf[id] = 0
            if  max(conf.values()) != 0:
                norm_conf[id] = conf[id] / max(conf.values())

        # Separation score
        for id in aircraft_list:
            sep_score_aircraft = sep_score_in_sec[sec][id]
            norm_sep_score_aircraft = sep_score_aircraft / sec_max_sep

            # Append information to output
            csv_log.append([sec, id, sep_score_aircraft, norm_sep_score_aircraft, conf[id], norm_conf[id]])

        # Summary for the specified second
        csv_log.append(['===============================================================','','','','',''])
        csv_log.append(['Time', "Number of Aircraft", 'Average Separation Score', 'Average Separation Score (Normalized)','Number of Conflicts', 'Number of Conflicts (Normalized)'])
        csv_log.append([sec, aircraft_num, sec_avg_sep, sec_avg_sep/sec_max_sep, count_num_total, norm_count_num_total])
        csv_log.append(['===============================================================','','','','',''])

        # Header for the next second
        csv_log.append(['Time', 'Aircraft ID', 'Local Separation Score', 'Local Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"])

    output_csv_dir = os.path.dirname(log_fname)

    file_name = os.path.join(output_csv_dir,"USEPE_ML Tactical Single All Features Local Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Tactical Analysis, Single Local Separation with Number of Conflicts for {}.csv is completed.".format( log_name))
    print("Tactical Analysis, Output file: {} \n".format(file_name))

def tactical_general_spatial_conf_csv(log_fname, count_in_sec):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]

    print("Tactical Analysis, Single General Separation with Number of Conflicts for {}.csv is started.".format(log_name))

    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])

    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get separation score
    sep_score_in_sec = sec_sep_general_spatial(log_fname) # sep score -> id: score, sep_score_in_sec -> sec:{id: score}

    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))

    csv_log = []
    csv_column = ['Time', "Aircraft ID", 'General Separation Score', 'General Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"]

    for sec in secs:
        aircraft_num = len(aircraft_num_in_sec[sec])
        aircraft_list = aircraft_num_in_sec[sec]
        # Get separation score in sec
        sep_score_list = list(sep_score_in_sec[sec].values())
        sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
        sec_max_sep = max(sep_score_list)
        sec_avg_sep = sum(sep_score_list) / aircraft_num
        # Get conflict num in sec
        count_num_total = len(count_in_sec[sec])
        norm_count_num_total = count_num_total / max_conf_num if max_conf_num != 0 else 0

        # Number of conflicts
        conf = {}
        for id in aircraft_list:
            conf[id] = 0
            for pair in count_in_sec[sec]:
                if id in pair:
                    conf[id] += 1

        # Normalized conflict
        norm_conf = {}
        for id in aircraft_list:
            norm_conf[id] = 0
            if  max(conf.values()) != 0:
                norm_conf[id] = conf[id] / max(conf.values())

        # Separation score
        for id in aircraft_list:
            sep_score_aircraft = sep_score_in_sec[sec][id]
            norm_sep_score_aircraft = sep_score_aircraft / sec_max_sep

            # Append information to output
            csv_log.append([sec, id, sep_score_aircraft, norm_sep_score_aircraft, conf[id], norm_conf[id]])

        # Summary for the specified second
        csv_log.append(['===============================================================','','','','',''])
        csv_log.append(['Time', "Number of Aircraft", 'Average Separation Score', 'Average Separation Score (Normalized)','Number of Conflicts', 'Number of Conflicts (Normalized)'])
        csv_log.append([sec, aircraft_num, sec_avg_sep, sec_avg_sep/sec_max_sep, count_num_total, norm_count_num_total])
        csv_log.append(['===============================================================','','','','',''])

        # Header for the next second
        csv_log.append(['Time', 'Aircraft ID', 'General Separation Score', 'General Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"])

    output_csv_dir = os.path.dirname(log_fname)

    file_name = os.path.join(output_csv_dir,"USEPE_ML Tactical Single Spatial Features General Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Tactical Analysis, Single General Separation with Number of Conflicts for {}.csv is completed.".format( log_name))
    print("Tactical Analysis, Output file: {} \n".format(file_name))

def tactical_general_all_csv(log_fname, count_in_sec):

    # Load log
    log_name = os.path.basename(log_fname)[:-4]

    print("Tactical Analysis, Single General Separation with Number of Conflicts for {}.csv is started.".format(log_name))

    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])

    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get separation score
    sep_score_in_sec = sec_sep_general_all(log_fname) # sep score -> id: score, sep_score_in_sec -> sec:{id: score}

    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))

    csv_log = []
    csv_column = ['Time', "Aircraft ID", 'General Separation Score', 'General Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"]

    for sec in secs:
        aircraft_num = len(aircraft_num_in_sec[sec])
        aircraft_list = aircraft_num_in_sec[sec]
        # Get separation score in sec
        sep_score_list = list(sep_score_in_sec[sec].values())
        sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
        sec_max_sep = max(sep_score_list)
        sec_avg_sep = sum(sep_score_list) / aircraft_num
        # Get conflict num in sec
        count_num_total = len(count_in_sec[sec])
        norm_count_num_total = count_num_total / max_conf_num if max_conf_num != 0 else 0

        # Number of conflicts
        conf = {}
        for id in aircraft_list:
            conf[id] = 0
            for pair in count_in_sec[sec]:
                if id in pair:
                    conf[id] += 1

        # Normalized conflict
        norm_conf = {}
        for id in aircraft_list:
            norm_conf[id] = 0
            if  max(conf.values()) != 0:
                norm_conf[id] = conf[id] / max(conf.values())

        # Separation score
        for id in aircraft_list:
            sep_score_aircraft = sep_score_in_sec[sec][id]
            norm_sep_score_aircraft = sep_score_aircraft / sec_max_sep

            # Append information to output
            csv_log.append([sec, id, sep_score_aircraft, norm_sep_score_aircraft, conf[id], norm_conf[id]])

        # Summary for the specified second
        csv_log.append(['===============================================================','','','','',''])
        csv_log.append(['Time', "Number of Aircraft", 'Average Separation Score', 'Average Separation Score (Normalized)','Number of Conflicts', 'Number of Conflicts (Normalized)'])
        csv_log.append([sec, aircraft_num, sec_avg_sep, sec_avg_sep/sec_max_sep, count_num_total, norm_count_num_total])
        csv_log.append(['===============================================================','','','','',''])

        # Header for the next second
        csv_log.append(['Time', 'Aircraft ID', 'General Separation Score', 'General Separation Score (Normalized)', 'Number of Conflicts', "Number of Conflicts (Normalized)"])

    output_csv_dir = os.path.dirname(log_fname)

    file_name = os.path.join(output_csv_dir,"USEPE_ML Tactical Single All Features General Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Tactical Analysis, Single General Separation with Number of Conflicts for {}.csv is completed.".format( log_name))
    print("Tactical Analysis, Output file: {} \n".format(file_name))

# Export Single Local & General Separation Score to CSV file
def single_separation_score_to_csv(log_fname, count_fname, count_in_sec,
                                  count_category_name, sep_category_name, calc_category_name, use_tactical=False):
    print("Tactical Analysis, Single {} Separation, {}, Number of {}, to CSV started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    if count_category_name == 'Conflict':
        count_dic = output_pair_dic(count_d, ' confNum')
    elif count_category_name == 'LoS':
        count_dic = output_pair_dic(count_d, ' losNum')
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])

    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get separation score
    if sep_category_name == 'Spatial Features' and calc_category_name == 'Local':
        sep_score_in_sec = sec_sep_local_spatial(log_fname) # sep score -> id: score, sep_score_in_sec -> sec:{id: score}
    elif sep_category_name == 'All Features' and calc_category_name == 'Local':
        sep_score_in_sec = sec_sep_local_all(log_fname)
    elif sep_category_name == 'Spatial Features' and calc_category_name == 'General':
        sep_score_in_sec = sec_sep_general_spatial(log_fname)
    elif sep_category_name == 'All Features' and calc_category_name == 'General':
        sep_score_in_sec = sec_sep_general_all(log_fname)

    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))

    csv_log = []
    csv_column = ['Time', "Number of Aircraft in time", "Aircraft ID",
                  '{} Separation Score'.format(calc_category_name), '{} Separation Score (Normalized)'.format(calc_category_name),
                  'Number of {}'.format(count_category_name), "Number of {} (Normalized)".format(count_category_name)]

    for sec in secs:
        aircraft_num = len(aircraft_num_in_sec[sec])
        aircraft_list = aircraft_num_in_sec[sec]
        # Get separation score in sec
        sep_score_list = list(sep_score_in_sec[sec].values())
        sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
        sec_max_sep = max(sep_score_list)
        sec_avg_sep = sum(sep_score_list) / aircraft_num
        # Get conflict num in sec
        count_num_total = len(count_in_sec[sec])
        norm_count_num_total = count_num_total / max_conf_num if max_conf_num != 0 else 0

        for aircraft in aircraft_list:
            sep_score_aircraft = sep_score_in_sec[sec][aircraft]
            norm_sep_score_aircraft = sep_score_aircraft / sec_max_sep
            csv_log.append([
                sec, aircraft_num, aircraft, 
                sep_score_aircraft, norm_sep_score_aircraft, 
                '', ''
                ])

        # Write average of separation score and count num
        csv_log.append([
                sec, aircraft_num, 'Average', 
                sec_avg_sep, sec_avg_sep/sec_max_sep, 
                count_num_total, norm_count_num_total
                ])

    output_csv_dir = os.path.dirname(log_fname)
    if use_tactical is False:
        file_name = os.path.join(output_csv_dir,"USEPE_ML Time -vs- Single {} {} Separation with {} for {}.csv".format(sep_category_name, calc_category_name, count_category_name, log_name))
    else:
        file_name = os.path.join(output_csv_dir,"USEPE_ML Tactical Single {} {} Separation with {} for {}.csv".format(sep_category_name, calc_category_name, count_category_name, log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Tactical Analysis, Single {} Separation, {}, Number of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
    print("Tactical Analysis, Output file: {}".format(file_name))
        
# Export Single local separation score with number of conflicts to CSV file 
def local_spatial_conf_csv(log_fname, conf_count_fname):
    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of Conflicts, to CSV started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

    local = sep_local_spatial(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of Conflicts", "Number of Conflicts (Normalized)", 'Local Separation Score', 'Local Separation Score (Normalized)']
    for i, id in enumerate(local.keys()):      
        csv_log.append([i+1, id, conf[id], conf_norm[id], local[id], local_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single Spatial Features Local Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('Local Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of Conflicts, to CSV completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# Export Single local separation score with number of LoS to CSV file 
def local_spatial_los_csv(log_fname, los_count_fname):
    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of LoS, to CSV started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())

    local = sep_local_spatial(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of LoS", "Number of LoS (Normalized)", 'Local Separation Score', 'Local Separation Score (Normalized)']
    for i, id in enumerate(ids):      
        csv_log.append([i+1, id, los[id], los_norm[id], local[id], local_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single Spatial Features Local Separation with LoS for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('Local Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of LoS, to CSV completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# Export Single local separation score by all features with number of conflicts to CSV file 
def local_all_conf_csv(log_fname, conf_count_fname):
    print("Simulation Analysis, Single Local Separation, All Features, Number of Conflicts, to CSV started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

    local = sep_local_all(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of Conflicts", "Number of Conflicts (Normalized)", 'Local Separation Score', 'Local Separation Score (Normalized)']
    for i, id in enumerate(ids):      
        csv_log.append([i+1, id, conf[id], conf_norm[id], local[id], local_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single All Features Local Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('Local Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single Local Separation, All Features, Number of Conflicts, to CSV completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# Export Single local separation score by all features with number of LoS to CSV file 
def local_all_los_csv(log_fname, los_count_fname):
    print("Simulation Analysis, Single Local Separation, All Features, Number of LoS, to CSV started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())

    local = sep_local_all(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of LoS", "Number of LoS (Normalized)", 'Local Separation Score', 'Local Separation Score (Normalized)']
    for i, id in enumerate(ids):      
        csv_log.append([i+1, id, los[id], los_norm[id], local[id], local_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single All Features Local Separation with LoS for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('Local Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single Local Separation, All Features, Number of LoS, to CSV completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# Export single general separation score by spatial with number of conflict to CSV file 
def general_spatial_conf_csv(log_fname, conf_count_fname):
    print("Simulation Analysis, Single General Separation, Spatial Features, Number of Conflicts, to CSV started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

    general = sep_general_spatial(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of Conflicts", "Number of Conflicts (Normalized)", 'General Separation Score', 'General Separation Score (Normalized)']
    for i, id in enumerate(ids):      
        csv_log.append([i+1, id, conf[id], conf_norm[id], general[id], general_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single Spatial Features General Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('General Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single General Separation, Spatial Features, Number of Conflicts, to CSV completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# Export single general separation score with number of LoS to CSV file 
def general_spatial_los_csv(log_fname, los_count_fname):
    print("Simulation Analysis, Single General Separation, Spatial Features, Number of LoS, to CSV started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())

    general = sep_general_spatial(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of LoS", "Number of LoS (Normalized)", 'General Separation Score', 'General Separation Score (Normalized)']
    for i, id in enumerate(ids):      
        csv_log.append([i+1, id, los[id], los_norm[id], general[id], general_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single Spatial Features General Separation with LoS for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('General Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single General Separation, Spatial Features, Number of LoS, to CSV completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# Export single general separation score by all features with number of conflict to CSV file 
def general_all_conf_csv(log_fname, conf_count_fname):
    print("Simulation Analysis, Single General Separation, All Features, Number of Conflicts, to CSV started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

    general = sep_general_all(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of Conflicts", "Number of Conflicts (Normalized)", 'General Separation Score', 'General Separation Score (Normalized)']
    for i, id in enumerate(ids):      
        csv_log.append([i+1, id, conf[id], conf_norm[id], general[id], general_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single All Features General Separation with Conflict for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('General Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single General Separation, All Features, Number of Conflicts, to CSV completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# Export single general separation score by all features with number of conflict to CSV file 
def general_all_los_csv(log_fname, los_count_fname):
    print("Simulation Analysis, Single General Separation, All Features, Number of LoS, started.")

    log_name = os.path.basename(log_fname)[:-4]
    output_csv_dir = os.path.dirname(log_fname)

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())

    general = sep_general_all(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    ids = id_list(log_fname)

    csv_log = []
    csv_column = ['Index', 'Aircraft ID', "Number of LoS", "Number of LoS (Normalized)", 'General Separation Score', 'General Separation Score (Normalized)']
    for i, id in enumerate(ids):      
        csv_log.append([i+1, id, los[id], los_norm[id], general[id], general_norm[id]])
    file_name = os.path.join(output_csv_dir,"USEPE_ML Single All Features General Separation with LoS for {}.csv".format(log_name))

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.sort_values('General Separation Score', ascending=False, inplace=True)
        df.iloc[:, 0] = np.arange(1, len(ids)+1)
        df.to_csv(file_name, index=False, encoding="cp932")

    print("Simulation Analysis, Single General Separation, All Features, Number of LoS, completed.")
    print("Simulation Analysis, Output file: {}".format(file_name))

# [For EXTEND] Generate CSV: duration_dic_to_csv
def duration_dic_to_csv(file_name, duration_dic, sep_dic, mode='conf'):
    if mode == 'conf':
        mode_name = 'Conflict'
    elif mode == 'los':
        mode_name = 'LoS'
    # Define column of csv
    csv_column = ['Index', 'Aircraft1', 'Aircraft2', '{} Duration'.format(mode_name), '{} Duration (Normalized)'.format(mode_name), 'Separation Score', 'Separation Score (Normalized)']
    # Norm min-max
    norm_sep_dic = get_norm_MinMax_sep_score(sep_dic.copy())
    norm_duration_dic = get_norm_MinMax_sep_score(duration_dic.copy())

    csv_log = []
    for idx, pair_aircraft_name in enumerate(sep_dic.keys()):
        # Get score
        sep_score = sep_dic[pair_aircraft_name]
        norm_sep_score = norm_sep_dic[pair_aircraft_name]
        # Get duration
        duration_period = duration_dic[pair_aircraft_name]
        norm_duration_period = norm_duration_dic[pair_aircraft_name]
        # Get aircraft names
        aircraft_a, aircraft_b = pair_aircraft_name.split('&')
        # Convert 0000 -> ="0000" for csv output
        if (aircraft_a[0] == str(0)):
            aircraft_a = '="'+aircraft_a+'"'
        if (aircraft_b[0] == str(0)):
            aircraft_b = '="'+aircraft_b + '"'
        # Append
        csv_log.append([idx+1, aircraft_a, aircraft_b, duration_period, norm_duration_period, sep_score, norm_sep_score])

    with open(file_name, 'w') as f:
        df = pd.DataFrame(csv_log, columns=csv_column)
        df.to_csv(file_name, index=False, encoding="cp932")

# Generate CSV Func ---------------------------------------------------------------

# Exports STRATEGIC phase pair separation score
def strategic_pair_csv(log_fname):
    print("Strategic Phase, Pairwise analysis to CSV is started.")

    scn_name = '{}.scn'.format(stack.get_scenname())

    # Load log file
    d = data_import(log_fname, 1)

    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    
    # Define separation distance 
    # extract contents
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance 
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_csv_dir = os.path.dirname(log_fname)
    # Output csv
    log_name = os.path.basename(log_fname)[:-4]
    csv_fname = None
    csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Strategical Phase, Pairwise Separation Score for {}.csv'.format(log_name))
    pair_aircraft_dic_to_csv(csv_fname, separation_dic)

    print("Strategic Phase, Pairwise analysis to CSV is completed.")
    print("Strategic Phase, Output file: {}".format(csv_fname))

# Exports STRATEGIC phase local separation score
def strategic_local_csv(log_fname):
    print("Strategic Phase, Single Local analysis to CSV is started.")

    scn_name = '{}.scn'.format(stack.get_scenname())

    # Load log file
    d = data_import(log_fname, 1)

    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    
    # Define separation distance 
    # extract contents
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance 
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_csv_dir = os.path.dirname(log_fname)
    # Output csv
    log_name = os.path.basename(log_fname)[:-4]
    csv_fname = None
    csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Strategical Phase, Single Local Separation Score for {}.csv'.format(log_name))
    single_local_aircraft_sep_score_dic_to_csv(csv_fname, separation_dic, unique_ids)

    print("Strategic Phase, Single Local analysis to CSV is completed.")
    print("Strategic Phase, Output file: {}".format(csv_fname))

# Exports STRATEGIC phase local separation score
def strategic_general_csv(log_fname):
    print("Strategic Phase, Single General analysis to CSV is started.")

    scn_name = '{}.scn'.format(stack.get_scenname())

    # Load log file
    d = data_import(log_fname, 1)

    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    
    # Define separation distance 
    # extract contents
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance 
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_csv_dir = os.path.dirname(log_fname)
    # Output csv
    log_name = os.path.basename(log_fname)[:-4]
    csv_fname = None
    csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Strategical Phase, Single General Separation Score for {}.csv '.format(log_name))
    single_general_aircraft_sep_score_dic_to_csv(csv_fname, separation_dic, unique_ids)

    print("Strategic Phase, Single General analysis to CSV is completed.")
    print("Strategic Phase, Output file: {}".format(csv_fname))

# Exports BASIC
def output_csv_from_count_dic(log_fname, count_fname, mode='conf', use_entire_content=True, output_single_score=False):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)
    # Load log file
    d = data_import(log_fname, 1)
    # load conf_pair
    count_d = data_import(count_fname, 3)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    # Conf count ------------------------------------------------------------
    # Generate conflict
    if mode == 'conf':
        count_dic = output_pair_dic(count_d, ' confNum')
        count_category_name = 'Conflict'
    elif mode == 'los':
        count_dic = output_pair_dic(count_d, ' losNum')
        count_category_name = 'LoS'
    # Sort by conflict num
    count_dic = dict(sorted(count_dic.items(), key=lambda x: x[1]))
    # --------------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for key in count_dic:
        aircraft_a, aircraft_b = key.split('&')
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_csv_dir = os.path.dirname(log_fname)

    # Output csv
    csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Pairwise {} {} for {}'.format(sep_category_name, count_category_name, log_name))
    csv_fname = data_export(csv_fname)
    count_dic_to_csv(csv_fname, count_dic, separation_dic, mode)

def output_csv_from_duration_dic(log_fname, duration_fname, mode='conf', use_entire_content=True):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)
    # Load log file
    d = data_import(log_fname, 1)
    # load duration
    duration_d = data_import(duration_fname, 2)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    if mode == 'conf':
        count_category_name = 'Conflict'
    elif mode == 'los':
        count_category_name = 'LoS'
    # Get duration period ---------------------------------------------------
    duration_dic = output_pair_dic(duration_d, ' Duration Time[s]')
    # Sort by duration length
    duration_dic = dict(sorted(duration_dic.items(), key=lambda x: x[1]))
    # --------------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for key in duration_dic:
        aircraft_a, aircraft_b = key.split('&')
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_csv_dir = os.path.dirname(log_fname)
    # Output csv
    csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Pairwise {} {} Duration for {}'.format(sep_category_name, count_category_name, log_name))
    csv_fname = data_export(csv_fname)
    duration_dic_to_csv(csv_fname, duration_dic, separation_dic, mode)

# [For ML TACTICAL]

def tactical_pairwise_spatial_conf_csv(log_fname, count_timeseries):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]

    print("Tactical Analysis, Pairwise Separation, Spatial, Number of Conflicts, to CSV started for {}.".format(log_name))

    # Load log file
    d = data_import(log_fname, 1)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # Get seconds
    secs = sorted(list(d['# simt'])[1:], key=lambda x: float(x))
    # Get aircrafts in seconds
    aircrafts_in_sec = aircraft_num_in_sec_list(log_fname) # sec: [ids]

    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    separation_contents = [' lat', ' lon', ' alt']
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    # ----------------------------------------------------------------------------
    for idx_i, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output CSV with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for idx_j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[idx_j][k] = data_a[content]
                aircraft_b_data[idx_j][k] = data_b[content]
            # get sep score in time=sec
            euc_score = get_sep_score(aircraft_a_data.reshape(1, -1), aircraft_b_data.reshape(1, -1))
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = dict()
            sec_separation_dic[sec][key] = euc_score
    # -----------------------------------------------------------------------------------------
    # Set output csv dir
    output_csv_dir = os.path.dirname(log_fname)
    # Pairwise ----------------------------------------
    conf_csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Tactical Pairwise Spatial Features with Conflict for {}'.format(log_name))
    conf_csv_fname = data_export(conf_csv_fname)
    pairwise_csv_log = []
    pairwise_csv_column = ['Time', 'Aircraft1', 'Aircraft2', 'Separation Score', 'Separation Score (Normalized)', 'Conflicted']

    max_conf_num = max([len(count_timeseries[sec]) for sec in count_timeseries.keys()]) # Get maximum conflict number
    
    for idx, sec in enumerate(sec_separation_dic.keys()):
        aircraft_num = len(aircrafts_in_sec[sec])
        # get average
        sum_sec_sep = sum(sec_separation_dic[sec].values())
        avg_sec_sep =sum_sec_sep / len(sec_separation_dic[sec].keys())
        max_sec_sep = max(sec_separation_dic[sec].values())
        # Get total conflict number in second
        total_conf_num = len(count_timeseries[sec])
        norm_total_conf_num = total_conf_num/max_conf_num if max_conf_num !=0 else 0
        # get norm
        norm_sec_separation_dic = get_norm_MinMax_sep_score(sec_separation_dic[sec].copy())

        # Write to log
        for pair_name in sec_separation_dic[sec].keys():
            aircraft_a, aircraft_b = pair_name.split('&')

            sec_score = sec_separation_dic[sec][pair_name]
            norm_sec_score = norm_sec_separation_dic[pair_name]

            # If there is a conflict 1 otherwise 0 at sec
            if pair_name in count_timeseries[sec]:
                sec_conf = 1
            else:
                sec_conf = 0

            pairwise_csv_log.append([sec, aircraft_a, aircraft_b, sec_score, norm_sec_score, sec_conf])
        
        # Summary for the specified second
        pairwise_csv_log.append(['===============================================================','','','','',''])
        pairwise_csv_log.append(['Time', "Number of Aircraft", 'Average Separation Score', 'Average Normalized Separation Score','Number of Conflicts', 'Number of Conflicts (Normalized)'])
        pairwise_csv_log.append([sec, aircraft_num, avg_sec_sep, avg_sec_sep/max_sec_sep, total_conf_num, norm_total_conf_num])
        pairwise_csv_log.append(['===============================================================','','','','',''])

        # Header for the next second
        pairwise_csv_log.append(['Time', 'Aircraft1', 'Aircraft2', 'Separation Score', 'Separation Score (Normalized)', 'Conflicted'])

    with open(conf_csv_fname, 'w') as f:
        df = pd.DataFrame(pairwise_csv_log, columns=pairwise_csv_column)
        df.to_csv(conf_csv_fname, index=False, encoding="cp932")

    print("Tactical Analysis, Pairwise Separation, Spatial, Number of Conflicts to CSV completed for {}.".format(log_name))
    print("Tactical Analysis, Output file: {}.png \n".format(conf_csv_fname))

def tactical_pairwise_all_conf_csv(log_fname, count_timeseries):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]

    print("Tactical Analysis, Pairwise Separation, All, Number of Conflicts, to CSV started for {}.".format(log_name))

    # Load log file
    d = data_import(log_fname, 1)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # Get seconds
    secs = sorted(list(d['# simt'])[1:], key=lambda x: float(x))
    # Get aircrafts in seconds
    aircrafts_in_sec = aircraft_num_in_sec_list(log_fname) # sec: [ids]

    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    # ----------------------------------------------------------------------------
    for idx_i, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output CSV with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for idx_j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[idx_j][k] = data_a[content]
                aircraft_b_data[idx_j][k] = data_b[content]
            # get sep score in time=sec
            euc_score = get_sep_score(aircraft_a_data.reshape(1, -1), aircraft_b_data.reshape(1, -1))
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = dict()
            sec_separation_dic[sec][key] = euc_score
    # -----------------------------------------------------------------------------------------
    # Set output csv dir
    output_csv_dir = os.path.dirname(log_fname)
    # Pairwise ----------------------------------------
    conf_csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Tactical Pairwise All Features with Conflict for {}'.format(log_name))
    conf_csv_fname = data_export(conf_csv_fname)
    pairwise_csv_log = []
    pairwise_csv_column = ['Time', 'Aircraft1', 'Aircraft2', 'Separation Score', 'Separation Score (Normalized)', 'Conflicted']

    max_conf_num = max([len(count_timeseries[sec]) for sec in count_timeseries.keys()]) # Get maximum conflict number
    
    for idx, sec in enumerate(sec_separation_dic.keys()):
        aircraft_num = len(aircrafts_in_sec[sec])
        # get average
        sum_sec_sep = sum(sec_separation_dic[sec].values())
        avg_sec_sep =sum_sec_sep / len(sec_separation_dic[sec].keys())
        max_sec_sep = max(sec_separation_dic[sec].values())
        # Get total conflict number in second
        total_conf_num = len(count_timeseries[sec])
        norm_total_conf_num = total_conf_num/max_conf_num if max_conf_num !=0 else 0
        # get norm
        norm_sec_separation_dic = get_norm_MinMax_sep_score(sec_separation_dic[sec].copy())

        # Write to log
        for pair_name in sec_separation_dic[sec].keys():
            aircraft_a, aircraft_b = pair_name.split('&')

            sec_score = sec_separation_dic[sec][pair_name]
            norm_sec_score = norm_sec_separation_dic[pair_name]

            # If there is a conflict 1 otherwise 0 at sec
            if pair_name in count_timeseries[sec]:
                sec_conf = 1
            else:
                sec_conf = 0

            pairwise_csv_log.append([sec, aircraft_a, aircraft_b, sec_score, norm_sec_score, sec_conf])
        
        # Summary for the specified second
        pairwise_csv_log.append(['===============================================================','','','','',''])
        pairwise_csv_log.append(['Time', "Number of Aircraft", 'Average Separation Score', 'Average Normalized Separation Score','Number of Conflicts', 'Number of Conflicts (Normalized)'])
        pairwise_csv_log.append([sec, aircraft_num, avg_sec_sep, avg_sec_sep/max_sec_sep, total_conf_num, norm_total_conf_num])
        pairwise_csv_log.append(['===============================================================','','','','',''])

        # Header for the next second
        pairwise_csv_log.append(['Time', 'Aircraft1', 'Aircraft2', 'Separation Score', 'Separation Score (Normalized)', 'Conflicted'])

    with open(conf_csv_fname, 'w') as f:
        df = pd.DataFrame(pairwise_csv_log, columns=pairwise_csv_column)
        df.to_csv(conf_csv_fname, index=False, encoding="cp932")

    print("Tactical Analysis, Pairwise Separation, All, Number of Conflicts to CSV completed for {}.".format(log_name))
    print("Tactical Analysis, Output file: {}.png \n".format(conf_csv_fname))

def output_tactical_csv(log_fname, count_fname, mode, count_timeseries, duration_fname=None, use_entire_content=False, output_single_score=False):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]
    # Load log file
    d = data_import(log_fname, 1)
    # load conf_pair
    count_d = data_import(count_fname, 3)
    count_dic = None
    count_category_name = None
    if mode == 'conf':
        count_dic = output_pair_dic(count_d, ' confNum')
        count_category_name = 'Conflict'
    elif mode == 'los':
        count_dic = output_pair_dic(count_d, ' losNum')
        count_category_name = 'LoS'
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # Get seconds
    secs = sorted(list(d['# simt'])[1:], key=lambda x: float(x))
    # Get aircrafts in seconds
    aircrafts_in_sec = aircraft_num_in_sec_list(log_fname) # sec: [ids]
    # Load duration
    duration_d = None
    duration_start_dic, duration_end_dic = None, None
    if duration_fname is not None:
        duration_d = data_import(duration_fname, 2)
        duration_start_dic = output_pair_dic(duration_d, ' start duration Time[s]')
        duration_end_dic = output_pair_dic(duration_d, ' end duration Time[s]')

    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    # ----------------------------------------------------------------------------
    for idx_i, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output CSV with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for idx_j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[idx_j][k] = data_a[content]
                aircraft_b_data[idx_j][k] = data_b[content]
            # get sep score in time=sec
            euc_score = get_sep_score(aircraft_a_data.reshape(1, -1), aircraft_b_data.reshape(1, -1))
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = dict()
            sec_separation_dic[sec][key] = euc_score
    # -----------------------------------------------------------------------------------------
    # Set output csv dir
    output_csv_dir = os.path.dirname(log_fname)
    # Pairwise ----------------------------------------
    conf_csv_fname = os.path.join(output_csv_dir, 'USEPE_ML Tactical Pairwise {} with {} for {}'.format(sep_category_name, count_category_name, log_name))
    conf_csv_fname = data_export(conf_csv_fname)
    pairwise_csv_log = []
    pairwise_csv_column = ['Time', "Number of Aircraft", 'Aircraft1', 'Aircraft2', 'Separation Score', 'Separation Score (Normalized)', 'Number of {}'.format(count_category_name), 'Number of {} (Normalized)'.format(count_category_name)]

    max_conf_num = max([len(count_timeseries[sec]) for sec in count_timeseries.keys()]) # Get maximum conflict number
    
    for idx, sec in enumerate(sec_separation_dic.keys()):
        aircraft_num = len(aircrafts_in_sec[sec])
        # get average
        sum_sec_sep = sum(sec_separation_dic[sec].values())
        avg_sec_sep =sum_sec_sep / len(sec_separation_dic[sec].keys())
        max_sec_sep = max(sec_separation_dic[sec].values())
        # Get total conflict number in second
        total_conf_num = len(count_timeseries[sec])
        norm_total_conf_num = total_conf_num/max_conf_num if max_conf_num !=0 else 0
        # get norm
        norm_sec_separation_dic = get_norm_MinMax_sep_score(sec_separation_dic[sec].copy())
        # Write to log
        for pair_name in sec_separation_dic[sec].keys():
            aircraft_a, aircraft_b = pair_name.split('&')

            sec_score = sec_separation_dic[sec][pair_name]
            norm_sec_score = norm_sec_separation_dic[pair_name]

            pairwise_csv_log.append([sec, aircraft_num, aircraft_a, aircraft_b, sec_score, norm_sec_score, '', ''])
        pairwise_csv_log.append([sec, aircraft_num, '', 'Average', avg_sec_sep, avg_sec_sep/max_sec_sep, total_conf_num, norm_total_conf_num])

    with open(conf_csv_fname, 'w') as f:
        df = pd.DataFrame(pairwise_csv_log, columns=pairwise_csv_column)
        df.to_csv(conf_csv_fname, index=False, encoding="cp932")

    # Single -------------------------------------------
    # single_separation_score_to_csv(log_fname, count_fname, count_timeseries, count_category_name, sep_category_name, calc_category_name='Local', use_tactical=True)
    # single_separation_score_to_csv(log_fname, count_fname, count_timeseries, count_category_name, sep_category_name, calc_category_name='General', use_tactical=True)

    if duration_d is not None:
        csv_duration_fname = os.path.join(output_csv_dir, 'USEPE_ML Tactical Pairwise {} Separation with {} Duration for {}'.format(sep_category_name, count_category_name, log_name))
        csv_duration_fname = data_export(csv_duration_fname)
        pairwise_csv_log = []
        pairwise_csv_column = ['Index', 'Time', 'Aircraft1', 'Aircraft2', 'Separation Score', 'Separation Score (Normalized)', 'Number of {}'.format(count_category_name), 'Number of {} (Normalized)'.format(count_category_name)]
        
        stack_count_dic = dict() # pairname: stacked conf num
        for i in range(len(duration_d)):
            d_arry = duration_d.iloc[i]
            # Get aircraft pair & duration period
            aircraft_a, aircraft_b = d_arry[' Aircraft1'], d_arry[' Aircraft2']
            key = '{}&{}'.format(aircraft_a, aircraft_b)
            start_duration_sec = int(float(d_arry[' start duration Time[s]']))
            end_duration_sec = int(float(d_arry[' end duration Time[s]']))

            if key not in stack_count_dic:
                stack_count_dic[key] = 1
            else:
                stack_count_dic[key] += 1

            for sec in range(start_duration_sec, end_duration_sec):
                # Get sep & max sep (use for normalize)
                sec = float(sec)
                sec_sep = sec_separation_dic[sec][key]
                max_sec_sep = max(sec_separation_dic[sec].values())
                norm_sec_sep = sec_sep/max_sec_sep if max_sec_sep != 0 else 0
                # Get stacked conf num in
                stack_conf_num = stack_count_dic[key]
                norm_stack_conf_num = stack_conf_num/max_conf_num if max_conf_num != 0 else 0

                pairwise_csv_log.append([i+1, float(sec), aircraft_a, aircraft_b, sec_sep, norm_sec_sep, stack_conf_num, norm_stack_conf_num])

            with open(csv_duration_fname, 'w') as f:
                df = pd.DataFrame(pairwise_csv_log, columns=pairwise_csv_column)
                df.to_csv(csv_duration_fname, index=False, encoding="cp932")

# ----------------------------------------------------------------------------------------
# Plot Content ---------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Plot: Single Separation Score with the Number of Aircraft & Conflicts vs time
def single_separation_score_plot(log_fname, count_fname, count_in_sec, count_category_name, sep_category_name, calc_category_name, use_tactical=False):

    # Load log
    log_name = os.path.basename(log_fname)[:-4]

    print("Tactical Analysis, Single {} Separation, {}, Number of {}, to PNG started for {}.".format(sep_category_name, calc_category_name, count_category_name, log_name))

    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)

    if count_category_name == 'Conflict':
        count_dic = output_pair_dic(count_d, ' confNum')
    elif count_category_name == 'LoS':
        count_dic = output_pair_dic(count_d, ' losNum')
    max_conf_time = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_time = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_time = sum_conf_time / len(ids)
    norm_avg_conf_time = avg_conf_time / max_conf_time if max_conf_time != 0 else 0

    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get separation score
    if sep_category_name == 'Spatial Features' and calc_category_name == 'Local':
        sep_score_in_sec = sec_sep_local_spatial(log_fname) # sep score -> id: score, sep_score_in_sec -> sec:{id: score}
    elif sep_category_name == 'All Features' and calc_category_name == 'Local':
        sep_score_in_sec = sec_sep_local_all(log_fname)
    elif sep_category_name == 'Spatial Features' and calc_category_name == 'General':
        sep_score_in_sec = sec_sep_general_spatial(log_fname)
    elif sep_category_name == 'All Features' and calc_category_name == 'General':
        sep_score_in_sec = sec_sep_general_all(log_fname)

    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))

    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    ax.set_title('Tactical Analysis, {} Separation Score ({})'.format(calc_category_name, sep_category_name), loc='center', fontsize=15, color='#154B5F')
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    X = []
    Y0, Y1 = [], []
    avg_sim_list = []

    for sec in secs:
        aircraft_num = len(aircraft_num_in_sec[sec])
        pairname_list = count_in_sec[sec]
        # Get separation score in sec
        sep_score_list = list(sep_score_in_sec[sec].values())
        sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
        sec_max_sep = max(sep_score_list)
        sec_avg_sep = sum(sep_score_list) / aircraft_num
        # Get conflict time in sec
        count_time_total = len(count_in_sec[sec])
        norm_count_time_total = count_time_total / max_conf_time if max_conf_time != 0 else 0

        #for pairname in pairname_list:
        #    aircraft_a, aircraft_b = pairname.split('&')
        #    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
        #    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep

        #    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
        #    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep

        X.append(sec)
        Y0.append(count_time_total / max_conf_time if max_conf_time != 0 else 0)
        Y1.append(sec_avg_sep / sec_max_sep)
        avg_sim_list.append(sec_avg_sep)
    X = np.array(X)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    avg_sim = sum(avg_sim_list) / len(avg_sim_list)
    norm_avg_sim = avg_sim / max(avg_sim_list)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of {}s (Normalized)".format(count_category_name), color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="{} Separation Score (Normalized)".format(calc_category_name), color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Time [s]', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')       
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (len(X)*0.02, 0.78), len(X)*0.75, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Total time of conflicts:'],
            ['Number of conflicted time/aircraft:'],
            ['Number of normalized conflicted time/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(len(ids))],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(sum_conf_time)],
            [' {:.2f}'.format(avg_conf_time)],
            [' {:.2f}'.format(norm_avg_conf_time)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')

    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    if use_tactical is False:
        file_name = os.path.join(output_graph_dir, 'USEPE_ML Time -vs- Single {} {} Separation with {} for {}'.format(sep_category_name, calc_category_name, count_category_name, single_output_name))
    else:
        file_name = os.path.join(output_graph_dir, 'USEPE_ML Tactical Single {} {} Separation with {} for {}'.format(sep_category_name, calc_category_name, count_category_name, single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Tactical Analysis, Single {} Separation, {}, Number of {}, to PNG completed for {}.".format(sep_category_name, calc_category_name, count_category_name, log_name))
    print("Tactical Analysis, Output file: {}.png \n".format(file_name))

# CSV & Plot: Single Separation Score with the Conflict Duration vs time
def single_separation_score_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list,
                                                       count_category_name, sep_category_name, calc_category_name, use_tactical=False):
    print("Tactical Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    if count_category_name == 'Conflict':
        count_dic = output_pair_dic(count_d, ' confNum')
    elif count_category_name == 'LoS':
        count_dic = output_pair_dic(count_d, ' losNum')
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0

    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get separation score
    if sep_category_name == 'Spatial Features' and calc_category_name == 'Local':
        sep_score_in_sec = sec_sep_local_spatial(log_fname) # sep score -> id: score, sep_score_in_sec -> sec:{id: score}
    elif sep_category_name == 'All Features' and calc_category_name == 'Local':
        sep_score_in_sec = sec_sep_local_all(log_fname)
    elif sep_category_name == 'Spatial Features' and calc_category_name == 'General':
        sep_score_in_sec = sec_sep_general_spatial(log_fname)
    elif sep_category_name == 'All Features' and calc_category_name == 'General':
        sep_score_in_sec = sec_sep_general_all(log_fname)

    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if use_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {} {} {} {}'.format(sep_category_name, calc_category_name, count_category_name, log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Tactical Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Tactical Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Number of Aircraft in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Tactical Analysis, Average {} Separation Score ({})'.format(calc_category_name, sep_category_name), loc='center', fontsize=15, color='#154B5F')
            ax1.set_xlabel('Time [s]', fontsize=11, color='#154B5F')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#154B5F')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-o', color='#999999', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-^', color='#20708D', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            # ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax1.tick_params(axis='x', colors='#154B5F')
            ax1.tick_params(axis='y', colors='#154B5F')
            #-----------
            ax2.spines['bottom'].set_color('#154B5F')
            ax2.spines['top'].set_color('#154B5F')
            ax2.spines['left'].set_color('#154B5F')
            ax2.spines['right'].set_color('#154B5F')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, '{} {} {} {} {}'.format(pairname, sep_category_name, calc_category_name, count_category_name, log_name))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Tactical Analysis,  ==== Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Tactical Analysis, Output file: {}".format(csv_file_name))
        print("Tactical Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

# Plots the local separation score with number of conflicts
def local_spatial_conf_plot(log_fname, conf_count_fname):
    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of Conflicts, to PNG started.")

    ids = id_list(log_fname)

    local = sep_local_spatial(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(local.values())) / aircraft
    norm_avg_sim = sum(list(local_norm.values())) / aircraft
    tot_conf_single = sum(list(conf.values()))
    ave_conf_single = sum(list(conf.values())) / aircraft
    ave_norm_conf_single = sum(list(conf_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(local)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(local.keys()):
        conf = conf_norm[aircraft_name]
        sep = local_norm[aircraft_name]
        Y0.append(conf)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of Conflicts (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="Local Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')       
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single Local Separation (Spatial Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total conflicts:'],
            ['Number of conflicts/aircraft:'],
            ['Number of normalized conflicts/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_conf_single)],
            [' {:.2f}'.format(ave_conf_single)],
            [' {:.2f}'.format(ave_norm_conf_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single Spatial Features Local Separation with Conflict for {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of Conflicts, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# Plots the local separation score with number of LoS
def local_spatial_los_plot(log_fname, los_count_fname):
    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of LoS, to PNG started.")

    ids = id_list(log_fname)

    local = sep_local_spatial(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())
    
   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(local.values())) / aircraft
    norm_avg_sim = sum(list(local_norm.values())) / aircraft
    tot_los_single = sum(list(los.values()))
    ave_los_single = sum(list(los.values())) / aircraft
    ave_norm_los_single = sum(list(los_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(local)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(local.keys()):
        los = los_norm[aircraft_name]
        sep = local_norm[aircraft_name]
        Y0.append(los)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of LoS (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="Local Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')       
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single Local Separation (Spatial Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total LoS:'],
            ['Number of LoS/aircraft:'],
            ['Number of normalized LoS/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_los_single)],
            [' {:.2f}'.format(ave_los_single)],
            [' {:.2f}'.format(ave_norm_los_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single Spatial Features Local Separation with LoS for  {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single Local Separation, Spatial Features, Number of LoS, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# Plots the general separation score with number of conflicts
def general_spatial_conf_plot(log_fname, conf_count_fname):
    print("Simulation Analysis, Single General Separation, Spatial Features, Number of Conflicts, to PNG started.")

    ids = id_list(log_fname)

    general = sep_general_spatial(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(general.values())) / aircraft
    norm_avg_sim = sum(list(general_norm.values())) / aircraft
    tot_conf_single = sum(list(conf.values()))
    ave_conf_single = sum(list(conf.values())) / aircraft
    ave_norm_conf_single = sum(list(conf_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(general)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(general.keys()):
        conf = conf_norm[aircraft_name]
        sep = general_norm[aircraft_name]
        Y0.append(conf)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of Conflicts (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="General Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')       
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single General Separation (Spatial Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total conflicts:'],
            ['Number of conflicts/aircraft:'],
            ['Number of normalized conflicts/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_conf_single)],
            [' {:.2f}'.format(ave_conf_single)],
            [' {:.2f}'.format(ave_norm_conf_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single Spatial Features General Separation with Conflict for {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single General Separation, Spatial Features, Number of Conflicts, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# Plots the general separation score with number of LoS
def general_spatial_los_plot(log_fname, los_count_fname):
    print("Simulation Analysis, Single General Separation, Spatial Features, Number of LoS, to PNG started.")

    ids = id_list(log_fname)

    general = sep_general_spatial(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())

   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(general.values())) / aircraft
    norm_avg_sim = sum(list(general_norm.values())) / aircraft
    tot_los_single = sum(list(los.values()))
    ave_los_single = sum(list(los.values())) / aircraft
    ave_norm_los_single = sum(list(los_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(general)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(general.keys()):
        los = los_norm[aircraft_name]
        sep = general_norm[aircraft_name]
        Y0.append(los)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of LoS (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="General Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')         
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single General Separation (Spatial Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total LoS:'],
            ['Number of LoS/aircraft:'],
            ['Number of normalized LoS/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_los_single)],
            [' {:.2f}'.format(ave_los_single)],
            [' {:.2f}'.format(ave_norm_los_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single Spatial Features General Separation with LoS for {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single General Separation, Spatial Features, Number of LoS, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# Plots the local separation score with number of conflicts
def local_all_conf_plot(log_fname, conf_count_fname):
    print("Simulation Analysis, Single Local Separation, All Features, Number of Conflicts, to PNG started.")

    ids = id_list(log_fname)

    local = sep_local_all(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(local.values())) / aircraft
    norm_avg_sim = sum(list(local_norm.values())) / aircraft
    tot_conf_single = sum(list(conf.values()))
    ave_conf_single = sum(list(conf.values())) / aircraft
    ave_norm_conf_single = sum(list(conf_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(local)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(local.keys()):
        conf = conf_norm[aircraft_name]
        sep = local_norm[aircraft_name]
        Y0.append(conf)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of Conflicts (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="Local Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')         
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single Local Separation (All Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total conflicts:'],
            ['Number of conflicts/aircraft:'],
            ['Number of normalized conflicts/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_conf_single)],
            [' {:.2f}'.format(ave_conf_single)],
            [' {:.2f}'.format(ave_norm_conf_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single All Features Local Separation with Conflict for {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single Local Separation, All Features, Number of Conflicts, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# Plots the local separation score with number of LoS
def local_all_los_plot(log_fname, los_count_fname):
    print("Simulation Analysis, Single Local Separation, All Features, Number of LoS, to PNG started.")

    ids = id_list(log_fname)

    local = sep_local_all(log_fname)
    local_norm = get_norm_MinMax_sep_score(local.copy())

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())
    
   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(local.values())) / aircraft
    norm_avg_sim = sum(list(local_norm.values())) / aircraft
    tot_los_single = sum(list(los.values()))
    ave_los_single = sum(list(los.values())) / aircraft
    ave_norm_los_single = sum(list(los_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(local)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(local.keys()):
        los = los_norm[aircraft_name]
        sep = local_norm[aircraft_name]
        Y0.append(los)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of LoS (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="Local Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')         
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single Local Separation (All Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total LoS:'],
            ['Number of LoS/aircraft:'],
            ['Number of normalized LoS/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_los_single)],
            [' {:.2f}'.format(ave_los_single)],
            [' {:.2f}'.format(ave_norm_los_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single All Features Local Separation with LoS for  {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single Local Separation, All Features, Number of LoS, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# Plots the general separation score with number of conflicts
def general_all_conf_plot(log_fname, conf_count_fname):
    print("Simulation Analysis, Single General Separation, All Features, Number of Conflicts, to PNG started.")

    ids = id_list(log_fname)

    general = sep_general_all(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    conf = single_conf_num(conf_count_fname)
    conf_norm = get_norm_MinMax_sep_score(conf.copy())

   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(general.values())) / aircraft
    norm_avg_sim = sum(list(general_norm.values())) / aircraft
    tot_conf_single = sum(list(conf.values()))
    ave_conf_single = sum(list(conf.values())) / aircraft
    ave_norm_conf_single = sum(list(conf_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(general)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(general.keys()):
        conf = conf_norm[aircraft_name]
        sep = general_norm[aircraft_name]
        Y0.append(conf)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of Conflicts (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="General Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')         
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single General Separation (All Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total conflicts:'],
            ['Number of conflicts/aircraft:'],
            ['Number of normalized conflicts/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_conf_single)],
            [' {:.2f}'.format(ave_conf_single)],
            [' {:.2f}'.format(ave_norm_conf_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single All Features General Separation with Conflict for {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single General Separation, All Features, Number of Conflicts, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# Plots the general separation score with number of LoS
def general_all_los_plot(log_fname, los_count_fname):
    print("Simulation Analysis, Single General Separation, All Features, Number of LoS, to PNG started.")

    ids = id_list(log_fname)

    general = sep_general_all(log_fname)
    general_norm = get_norm_MinMax_sep_score(general.copy())

    los = single_los_num(los_count_fname)
    los_norm = get_norm_MinMax_sep_score(los.copy())

   # norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(ids)
    avg_sim = sum(list(general.values())) / aircraft
    norm_avg_sim = sum(list(general_norm.values())) / aircraft
    tot_los_single = sum(list(los.values()))
    ave_los_single = sum(list(los.values())) / aircraft
    ave_norm_los_single = sum(list(los_norm.values())) / aircraft
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(general)+1))
    ax.xaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    plt.xlim([0, 1.02*len(X)])
    Y0, Y1 = [], []
    for idx, aircraft_name in enumerate(general.keys()):
        los = los_norm[aircraft_name]
        sep = general_norm[aircraft_name]
        Y0.append(los)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label="Number of LoS (Normalized)", color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label="General Separation Score (Normalized)", color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircrafts', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')         
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    legend = ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")         

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Single General Separation (All Features)', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
        (aircraft*0.13, 0.78), aircraft*0.61, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Number of total LoS:'],
            ['Number of LoS/aircraft:'],
            ['Number of normalized LoS/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(tot_los_single)],
            [' {:.2f}'.format(ave_los_single)],
            [' {:.2f}'.format(ave_norm_los_single)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    # -------------------------------------------------------------------------
    output_graph_dir = os.path.dirname(log_fname)
    log_name = os.path.basename(log_fname)[:-4]
    single_output_name = log_name
    file_name = os.path.join(output_graph_dir, 'USEPE_ML Single All Features General Separation with LoS for {}'.format(single_output_name))
    plt.savefig(file_name)
    plt.close()

    print("Simulation Analysis, Single General Separation, All Features, Number of LoS, to PNG completed.")
    print("Simulation Analysis, Output file: {}.png".format(file_name))

# CSV & Plot: Single Separation Score with the Conflict Duration vs time
def local_spatial_conf_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'Conflict'
    sep_category_name = 'Spatial Features'
    calc_category_name = 'Local'
    # Get separation score
    sep_score_in_sec = sec_sep_local_spatial(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

def local_spatial_los_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'LoS'
    sep_category_name = 'Spatial Features'
    calc_category_name = 'Local'
    # Get separation score
    sep_score_in_sec = sec_sep_local_spatial(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

def general_spatial_conf_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'Conflict'
    sep_category_name = 'Spatial Features'
    calc_category_name = 'General'
    # Get separation score
    sep_score_in_sec = sec_sep_general_spatial(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

def general_spatial_los_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'LoS'
    sep_category_name = 'Spatial Features'
    calc_category_name = 'General'
    # Get separation score
    sep_score_in_sec = sec_sep_general_spatial(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

def local_all_conf_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'Conflict'
    sep_category_name = 'All Features'
    calc_category_name = 'Local'
    # Get separation score
    sep_score_in_sec = sec_sep_local_all(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

def local_all_los_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'LoS'
    sep_category_name = 'All Features'
    calc_category_name = 'Local'
    # Get separation score
    sep_score_in_sec = sec_sep_local_all(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

def general_all_conf_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'Conflict'
    sep_category_name = 'All Features'
    calc_category_name = 'General'
    # Get separation score
    sep_score_in_sec = sec_sep_general_all(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))

def general_all_los_csv_and_plot_with_duration(log_fname, count_fname, count_in_sec, duration_list, used_tactical=False):
    # Load log
    log_name = os.path.basename(log_fname)[:-4]
    ids = id_list(log_fname)
    # Get count dic
    count_d = data_import(count_fname, 3)
    # Get aircraft num in sec
    aircraft_num_in_sec = aircraft_num_in_sec_list(log_fname)
    # Get count category name & sep category name & calc category name
    count_category_name = 'LoS'
    sep_category_name = 'All Features'
    calc_category_name = 'General'
    # Get separation score
    sep_score_in_sec = sec_sep_general_all(log_fname)
    # Get sec
    secs = list(sep_score_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    # Get max conf num in sec
    max_conf_num = max([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    sum_conf_num = sum([len(count_in_sec[sec]) for sec in count_in_sec.keys()])
    avg_conf_num = sum_conf_num / len(ids)
    norm_avg_conf_num = avg_conf_num / max_conf_num if max_conf_num != 0 else 0
    # Get duration info
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    duration_pairname_list = duration_start_list.keys()
    print("Simulation Analysis, Single {} Separation, {}, Duration of {}, to CSV & PNG started.".format(calc_category_name, sep_category_name, calc_category_name))
    # Create dir 
    if len(duration_pairname_list) != 0:
        log_name = os.path.basename(log_fname)[:-4]
        if used_tactical is False:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        else:
            outputs_dir = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single Separation Scores & Durations for {}'.format(log_name))
            csv_file_name = os.path.join(os.path.dirname(log_fname), 'USEPE_ML TACTICAL Time -vs- Single {} {} Separation with {} & Duration for {}.csv'.format(sep_category_name, calc_category_name, count_category_name, log_name))
        
        if os.path.exists(outputs_dir) is False:
            os.mkdir(outputs_dir)
        output_graph_dir = outputs_dir
        
        # Initialize csv variables
        csv_log = []
        csv_column = ['Index', 'Time', "Aircraft Num in Time", "Aircraft1", "Aircraft2",
                      '{} Separation Score of Aircraft1'.format(calc_category_name), '{} Separation Score of Aircraft1 (Normalized)'.format(calc_category_name),
                      '{} Separation Score of Aircraft2'.format(calc_category_name), '{} Separation Score of Aircraft2 (Normalized)'.format(calc_category_name)]

        for idx, pairname in enumerate(duration_pairname_list):
            aircraft_a, aircraft_b = pairname.split('&')
            start_duration, end_duration = duration_start_list[pairname], duration_end_list[pairname]
            # CSV -----------------------------------------------------------------------------------
            for s, e in zip(start_duration, end_duration):
                for sec in range(int(s), int(e)):
                    sec = float(sec)
                    # Get aircraft num in sec
                    aircraft_num = len(aircraft_num_in_sec[sec])
                    # Get separation score in sec
                    sep_score_list = list(sep_score_in_sec[sec].values())
                    sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                    sec_max_sep = max(sep_score_list)
                    sec_avg_sep = sum(sep_score_list) / aircraft_num
                    # Sep scores of each aircraft
                    sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a]
                    sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b]
                    norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep
                    norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep
                    # Append to csv_log (each separation score)
                    csv_log.append([idx, sec, aircraft_num, aircraft_a, aircraft_b,
                                   sep_score_aircraft_a, norm_sep_score_aircraft_a, sep_score_aircraft_b, norm_sep_score_aircraft_b])
            # Plot ------------------------------------------------------------------------------------
            # Initialize plot variables
            # Set figure
            fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
            ax2 = ax1.twinx()
            ax1.set_title('Average {} Separation Score and Number of {} for {} with Duration'.format(calc_category_name, count_category_name, sep_category_name), loc='center', fontsize=15)
            ax1.set_xlabel('Time [s]', fontsize=11, color='black')
            ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#329B8A')
            X = np.array(secs)

            Y1 = [] # Normalized Average Separation score
            Y1_1 = [] # Normalized Separation score of Aircraft1
            Y1_2 = [] # Normalized Separation score of Aircraft2
            for sec in secs:
                aircraft_num = len(aircraft_num_in_sec[sec])
                # Get separation score in sec
                sep_score_list = list(sep_score_in_sec[sec].values())
                sep_score_list = [sep_score_list[i] if sep_score_list[i] != float('inf') else 0 for i in range(len(sep_score_list))]
                sec_max_sep = max(sep_score_list)
                sec_avg_sep = sum(sep_score_list) / aircraft_num
                norm_sec_avg_sep = sec_avg_sep / sec_max_sep if sec_max_sep != 0 else 0
                Y1.append(norm_sec_avg_sep)
                # Sep scores of each aircraft
                sep_score_aircraft_a = sep_score_in_sec[sec][aircraft_a] if aircraft_a in sep_score_in_sec[sec].keys() else None
                sep_score_aircraft_b = sep_score_in_sec[sec][aircraft_b] if aircraft_b in sep_score_in_sec[sec].keys() else None
                norm_sep_score_aircraft_a = sep_score_aircraft_a / sec_max_sep if sep_score_aircraft_a is not None else None
                norm_sep_score_aircraft_b = sep_score_aircraft_b / sec_max_sep if sep_score_aircraft_b is not None else None
                Y1_1.append(norm_sep_score_aircraft_a)
                Y1_2.append(norm_sep_score_aircraft_b)
            # Plot separation scores
            Y1 = np.array(Y1)
            Y1_1 = np.array(Y1_1)
            Y1_2 = np.array(Y1_2)
            lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average {} Separation Score (Normalized)'.format(calc_category_name))
            lns1_1 = ax1.plot(X, Y1_1, '-s', color='red', label='{} {} Separation Score (Normalized)'.format(aircraft_a, calc_category_name))
            lns1_2 = ax1.plot(X, Y1_2, '-s', color='blue', label='{} {} Separation Score (Normalized)'.format(aircraft_b, calc_category_name))

            # Plot duration
            Y_duration = np.zeros((len(start_duration)))
            # Make plot
            ax2.text(start_duration[0], Y_duration[0], pairname, color='orange')
            lns2 = ax2.plot(start_duration, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
            ax2.plot(end_duration, Y_duration, '-D', color='orange')
            # Make line
            start_pos = [[s, y] for s, y in zip(start_duration, Y_duration)]
            end_pos = [[e, y] for e, y in zip(end_duration, Y_duration)]
            colors = ['orange' for i in range(len(start_duration))]
            lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
            lc = LineCollection(lines, colors=colors, linewidth=3)
            ax2.add_collection(lc)
            ax2.axes.yaxis.set_visible(False)
            # Set color of axis
            ax2.spines['left'].set_color('#329B8A')
            ax2.tick_params(axis='y', colors='#329B8A')
            # Write Plot
            lns = lns1 + lns1_1 + lns1_2 + lns2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='lower left')
            file_name = os.path.join(output_graph_dir, 'Time -vs- Single Plots for {}'.format(pairname))
            plt.savefig(file_name)
            plt.close()


        # Write CSV
        with open(csv_file_name, 'w') as f:
            df = pd.DataFrame(csv_log, columns=csv_column)
            df.to_csv(csv_file_name, index=False, encoding="cp932")

        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to CSV completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))
        print("Simulation Analysis, Output file: {}".format(csv_file_name))
        print("Simulation Analysis, Pairwise {} Separation, {}, Duration of {}, to PNGs completed.".format(sep_category_name, calc_category_name, count_category_name, log_name))


#########################
# STRATEGIC Plots

# Strategic pair plots
def strategic_pair_plot(log_fname):

    print("Strategic Phase, Pairwise analysis to PNG is started.") 

    # Load log file
    d = data_import(log_fname, 1)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output Plot with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_graph_dir = os.path.dirname(log_fname)

    log_name = os.path.basename(log_fname)[:-4]

    file_name = os.path.join(output_graph_dir, 'USEPE_ML Strategical Phase, Pairwise Separation Score for {}'.format(log_name))

    # Normalize separation and conflict
    norm_sep_dic = get_norm_MinMax_sep_score(separation_dic.copy())

    # Get analysis summary
    aircraft = len(unique_ids)
    pair_number = int((aircraft * (aircraft-1)) / 2)
    avg_sim = sum(list(separation_dic.values())) / pair_number
    norm_avg_sim = sum(list(norm_sep_dic.values())) / pair_number

    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(separation_dic)+1))
    plt.xlim([0, 1.02*len(X)])
    Y1 = []
    for idx, aircraft_name in enumerate(separation_dic.keys()):
        sep = norm_sep_dic[aircraft_name]
        Y1.append(sep)
    Y1 = np.array(Y1)
    Y1_name='Separation Score (Normalized)'
    # Set title
    ax.plot(X, Y1, '-s', label=Y1_name, color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircraft Pairs', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')       
    # y label
    ax.set_ylabel('Separation Score (Normalized)', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Strategical Phase, Pairwise Separation Score', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
    ((pair_number)*0.16, 0.88), (pair_number)*0.58, 0.13,
    edgecolor = 'gray',
    facecolor = 'white',
    alpha = 0.9,
    fill=True,
    ))

    rect.set_zorder(9)

    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of pairs:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
        ],
        bbox=[-0.20, 0.87, 0.9, 0.11], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(pair_number)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)]
        ],
        bbox=[0.60, 0.87, 0.1, 0.11], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)

    # -------------------------------------------------------------------------
    plt.savefig(file_name)
    plt.close()

    print("Strategic Phase, Pairwise analysis to PNG is completed")
    print("Strategic Phase, Output file: {}.png".format(file_name))

# Strategic local plots
def strategic_local_plot(log_fname):

    print("Strategic Phase, Single local analysis to PNG is started.") 

    # Load log file
    d = data_import(log_fname, 1)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output Plot with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_graph_dir = os.path.dirname(log_fname)

    log_name = os.path.basename(log_fname)[:-4]
       
    local_sep_dic = get_single_local_aircraft_sep_score(separation_dic.copy(), unique_ids)

    file_name = os.path.join(output_graph_dir, 'USEPE_ML Strategical Phase, Single Local Separation Score for {}'.format(log_name))   

    # Normalize separation and conflict
    norm_sep_dic = get_norm_MinMax_sep_score(local_sep_dic.copy())

    # Get analysis summary
    aircraft = len(unique_ids)
    avg_sim = sum(list(local_sep_dic.values())) / aircraft
    norm_avg_sim = sum(list(norm_sep_dic.values())) / aircraft

    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(local_sep_dic)+1))
    plt.xlim([0, 1.02*len(X)])
    Y1 = []
    for idx, aircraft_name in enumerate(local_sep_dic.keys()):
        sep = norm_sep_dic[aircraft_name]
        Y1.append(sep)
    Y1 = np.array(Y1)
    Y1_name='Separation Score (Normalized)'
    # Set title
    ax.plot(X, Y1, '-s', label=Y1_name, color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircraft', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')         
    # y label
    ax.set_ylabel('Separation Score (Normalized)', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Strategical Phase, Single Local Separation Score', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
    ((aircraft)*0.16, 0.88), (aircraft)*0.58, 0.13,
    edgecolor = 'gray',
    facecolor = 'white',
    alpha = 0.9,
    fill=True,
    ))

    rect.set_zorder(9)

    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of Aircrafts:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
        ],
        bbox=[-0.20, 0.87, 0.9, 0.11], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)]
        ],
        bbox=[0.60, 0.87, 0.1, 0.11], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)

    # -------------------------------------------------------------------------
    plt.savefig(file_name)
    plt.close()

    print("Strategic Phase, Single local analysis to PNG is completed")
    print("Strategic Phase, Output file: {}.png".format(file_name))

# Strategic general plots
def strategic_general_plot(log_fname):

    print("Strategic Phase, Single general analysis to PNG is started.") 

    # Load log file
    d = data_import(log_fname, 1)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    separation_contents = [' lat', ' lon', ' alt']
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output Plot with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_graph_dir = os.path.dirname(log_fname)

    log_name = os.path.basename(log_fname)[:-4]

    general_sep_dic = get_single_general_aircraft_sep_score(separation_dic.copy(), unique_ids)

    file_name = os.path.join(output_graph_dir, 'USEPE_ML Strategical Phase, Single General Separation Score for {}'.format(log_name))   

    # Normalize separation and conflict
    norm_sep_dic = get_norm_MinMax_sep_score(general_sep_dic.copy())

    # Get analysis summary
    aircraft = len(unique_ids)
    avg_sim = sum(list(general_sep_dic.values())) / aircraft
    norm_avg_sim = sum(list(norm_sep_dic.values())) / aircraft

    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(general_sep_dic)+1))
    plt.xlim([0, 1.02*len(X)])
    Y1 = []
    for idx, aircraft_name in enumerate(general_sep_dic.keys()):
        sep = norm_sep_dic[aircraft_name]
        Y1.append(sep)
    Y1 = np.array(Y1)
    Y1_name='Separation Score (Normalized)'
    # Set title
    ax.plot(X, Y1, '-s', label=Y1_name, color='#329B8A', zorder=8)  # 50 155 138
    # set axis label names: x and y -----------------------------------------
    # x label
    ax.set_xlabel('Aircraft', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')       
    # y label
    ax.set_ylabel('Separation Score (Normalized)', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')

    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Put title

    # Set plot title
    ax.set_title('Strategical Phase, Single General Separation Score', loc='center', fontsize=15, color='#154B5F')

    # Summary information
    # Put background
    rect = ax.add_patch(Rectangle(
    ((aircraft)*0.16, 0.88), (aircraft)*0.58, 0.13,
    edgecolor = 'gray',
    facecolor = 'white',
    alpha = 0.9,
    fill=True,
    ))

    rect.set_zorder(9)

    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of Aircrafts:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
        ],
        bbox=[-0.20, 0.87, 0.9, 0.11], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)]
        ],
        bbox=[0.60, 0.87, 0.1, 0.11], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)

    # -------------------------------------------------------------------------
    plt.savefig(file_name)
    plt.close()

    print("Strategic Phase, Single general analysis to PNG is completed")
    print("Strategic Phase, Output file: {}.png".format(file_name))

##########################

# Number of Conflicts and Separation scores are presented for comparison
def num_and_sep_plot(file_name, aircraft_ids, count_dic, sep_dic, sep_category_name, Y0_name, Y1_name, mode='conf'):
    if mode == 'conf':
        text_0 = 'Number of total conflicts:'
        text_1 = 'Number of conflicts/pair:'
        text_2 = 'Number of normalized conflicts/pair:'
    elif mode == 'los':
        text_0 = 'Number of total LoS:'
        text_1 = 'Number of LoS/pair:'
        text_2 = 'Number of normalized LoS/pair:'
    # Normalize sep
    norm_sep_dic = get_norm_MinMax_sep_score(sep_dic.copy())
    norm_count_dic = get_norm_MinMax_sep_score(count_dic.copy())
    # Get analysis summary
    aircraft = len(aircraft_ids)
    pair_number = int((aircraft * (aircraft-1)) / 2)
    avg_sim = sum(list(sep_dic.values())) / pair_number
    norm_avg_sim = sum(list(norm_sep_dic.values())) / pair_number
    sep_num = sum(list(count_dic.values()))
    avg_sep_num = sep_num / pair_number
    avg_norm_sep_num = sum(list(norm_count_dic.values())) / pair_number
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    X = np.array(np.arange(1, len(sep_dic)+1))
    plt.xlim([0, 1.01*(len(sep_dic)+1)])
    Y0, Y1 = [], []
    for idx, pair_aircraft_name in enumerate(sep_dic.keys()):
        count = norm_count_dic[pair_aircraft_name]
        sep = norm_sep_dic[pair_aircraft_name]
        Y0.append(count)
        Y1.append(sep)
    Y0 = np.array(Y0)
    Y1 = np.array(Y1)
    # Set title
    ax.plot(X, Y0, '-o', label=Y0_name, color='#999999', zorder=8)
    ax.plot(X, Y1, '-s', label=Y1_name, color='#329B8A', zorder=8) # 50 155 138
    # set label name
    ax.set_xlabel('Aircraft Pairs', fontsize=11, color='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')         
    # y label
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')
    ax.tick_params(axis='y', colors='#154B5F')
    ax.legend(bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")
    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')
    # Put texts ---------------------------------------------------------------
    # Set sep category in text
    if sep_category_name == 'All Features':
        text_sep_category = 'All Features'
    else:
        text_sep_category = 'Spatial Features'
    # Put title
    if mode == 'conf':
        ax.set_title('Pairwise Separation Summary for Conflict ({})'.format(text_sep_category), loc='center', fontsize=15, color='#154B5F')
    elif mode == 'los':
        ax.set_title('Pairwise Separation Summary for LoS ({})'.format(text_sep_category), loc='center', fontsize=15, color='#154B5F')
    # Put background
    if mode == 'conf':
        rect = ax.add_patch(Rectangle(
            ((pair_number+1)*0.13, 0.76), (pair_number+1)*0.6, 0.25,
            edgecolor = 'gray',
            facecolor = 'white',
            alpha = 0.9,
            fill=True,
            ))
    elif mode == 'los':
        rect = ax.add_patch(Rectangle(
            ((pair_number+1)*0.16, 0.76), (pair_number+1)*0.57, 0.25,
            edgecolor = 'gray',
            facecolor = 'white',
            alpha = 0.9,
            fill=True,
            ))
    rect.set_zorder(9)
    # Set text output list
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['The number of pairs:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            [text_0],
            [text_1],
            [text_2]
        ],
        bbox = [-0.20, 0.76, 0.9, 0.22], edges='open', cellLoc='right')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {}'.format(pair_number)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(sep_num)],
            [' {:.2f}'.format(avg_sep_num)],
            [' {:.2f}'.format(avg_norm_sep_num)]
        ],
        bbox=[0.60, 0.76, 0.1, 0.22], edges='open', cellLoc='left')

    analysis_summary.set_fontsize(13)
    analysis_summary.set_zorder(10)
    # -------------------------------------------------------------------------
    # plt.tight_layout(pad=0.1)
    plt.savefig(file_name)
    plt.close()

def duration_and_sep_plot(file_name, aircraft_ids, duration_start_dic, duration_end_dic, sep_dic, sep_category_name, mode):
    if mode == 'conf':
        text_0 = 'Total conflict duration [s]:'
        text_1 = 'Average conflict duration:'
    elif mode == 'los':
        text_0 = 'Total LoS duration [s]:'
        text_1 = 'Average LoS duration:'
    # Normalize sep
    norm_sep_dic = get_norm_MinMax_sep_score(sep_dic.copy())
    # Get analysis summary
    aircraft = len(aircraft_ids)
    pair_number = int((aircraft * (aircraft-1)) / 2)
    avg_sim = sum(list(sep_dic.values())) / aircraft
    norm_avg_sim = sum(list(norm_sep_dic.values())) / aircraft
    # Initialize sum duration
    sum_duration = 0
    # Set figure
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
    #fig = plt.figure(figsize=(10, 8), dpi=1000)
    #ax1 = fig.subplots()
    ax2 = ax1.twinx()
    sep_Y = []
    # Plot duration period
    max_end_duration = 0
    for idx, pair_drone_name in enumerate(sep_dic.keys()):
        start_time = duration_start_dic[pair_drone_name]
        end_time = duration_end_dic[pair_drone_name]
        for s, e in zip(start_time, end_time):
            sum_duration += e - s
        # Plot duration ---------------------------------------
        X = [idx + 1, idx + 1] #Pair index
        # Make plot
        duration = list(zip(start_time, end_time))
        for d in duration:
            ax1.plot(X, d, '-D', color='#FCB544', linewidth=2)
        # -------------------------------------------------------
        # Update max_duration -----------------------------------
        if e > max_end_duration:
            max_end_duration = e
        # Get Separation Score -----------------------------------------
        sep_Y.append(norm_sep_dic[pair_drone_name])
    # Get average of sim
    avg_duration = sum_duration / aircraft
    # Plot Separation Score ----------------------------------------
    X = np.array(np.arange(1, len(sep_dic)+1))
    plt.xlim([0, 1.02*(len(sep_dic)+1)])
    Y = np.array(sep_Y)
    ax2.plot(X, Y, '-s', color='#329B8A', zorder=8)
    # -------------------------------------------------------

    # Put background to summary on graph
    rect = ax2.add_patch(Rectangle(
        ((pair_number+1)*0.16, 0.82), (pair_number+1)*0.6, 0.18,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True
        ))
    rect.set_zorder(9)

    # Define text
    if sep_category_name == 'All Features':
        text_sep_category = 'All Features'
    else:
        text_sep_category = 'Spatial Features'

    if mode == 'conf':
        title = 'Pairwise Separation Summary for Conflict Duration ({})'.format(text_sep_category)
        y_label = 'Conflict Duration [s]'
    elif mode == 'los':
        title = 'Pairwise Separation Summary for LoS Duration ({})'.format(text_sep_category)
        y_label = 'LoS duration [s]'

    # Set title&label
    ax1.set_title(title, loc='center', fontsize=15, color='#154B5F')
    analysis_summary_labels = ax2.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            [text_0],
            [text_1]
        ],
        bbox=[-0.20, 0.82, 0.9, 0.15], edges='open', cellLoc='right')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)

    analysis_summary = ax2.table(
        cellText=[
            [' {}'.format(aircraft)],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {:.2f}'.format(sum_duration)],
            [' {:.2f}'.format(avg_duration)]
        ],
        bbox=[0.60, 0.82, 0.1, 0.15], edges='open', cellLoc='left')
    analysis_summary.set_fontsize(13)
    analysis_summary.set_zorder(10)

    ax1.set_xlabel('Aircraft Pairs', fontsize=11, color='#154B5F')
    ax1.tick_params(axis='x', colors='#154B5F')
    ax1.set_ylabel(y_label, color='#154B5F', fontsize=11)
    ax1.tick_params(axis='y', colors='#154B5F')


    ax2.set_ylabel('Normalized Separation Score', color='#154B5F')
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(axis='y', colors='#154B5F')
    ax2.spines['bottom'].set_color('#154B5F')
    ax2.spines['top'].set_color('#154B5F')
    ax2.spines['left'].set_color('#154B5F')
    ax2.spines['right'].set_color('#154B5F')

    plt.savefig(file_name)
    plt.close()

# Plot: Pairwise Separation Score with the Number of Aircraft & Conflicts vs time
def timeseries_sep_and_aircrafts_plot(file_name, sep_scores_in_sec, aircraft_num_in_sec, conf_num_in_sec, count_category_name, sep_category_name, log_fname):
    avg_sep_in_sec, max_sep_in_sec = sep_scores_in_sec

    print("Tactical Analysis, Pairwise Separation, {} to PNG started.".format(sep_category_name))
    # Set figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)

    ax.set_title('Tactical Analysis, Pairwise Separation Score ({})'.format(sep_category_name), loc='center', fontsize=15, color='#154B5F')
    ax.set_xlabel('Time [s]', fontsize=11, color='#154B5F')
    ax.set_ylabel('Normalized Values', fontsize=11, color='#154B5F')

    # Set X label
    secs = list(avg_sep_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    X = np.array(secs)
    
    # Normalize sep & Set Y1 label
    norm_avg_sep_timeseries = np.array(list(avg_sep_in_sec.values())) / np.array(list(max_sep_in_sec.values()))
    Y1 = np.nan_to_num(norm_avg_sep_timeseries) # Change nan to 0
    # Set Y2, Number of Conflict/LoS (Normalized)
    max_conf_num = max([len(conf_num_in_sec[sec]) for sec in conf_num_in_sec.keys()])

    Y2 = []
    for sec in secs:
        # Get conflict num in sec
        count_num_total = len(conf_num_in_sec[sec])
        norm_count_num_total = count_num_total / max_conf_num if max_conf_num != 0 else 0
        Y2.append(norm_count_num_total)
    Y2 = np.array(Y2)

    # Set Y3, Aircraft number
    # Get aircrafts in sec
    aircraft_in_sec = aircraft_num_in_sec_list(log_fname) # sec: [ids]
    max_aircraft_num = max([len(aircraft_in_sec[sec]) for sec in aircraft_in_sec.keys()])

    Y3 = []
    avg_sep_list = []
    for sec in secs:
        total_aircraft = len(aircraft_in_sec[sec])
        norm_total_aircraft = total_aircraft / max_aircraft_num if max_aircraft_num != 0 else 0
        Y3.append(norm_total_aircraft)
    Y3 = np.array(Y3)

    lns1 = ax.plot(X, Y1, '-s', color='#329B8A', label='Separation Score (Normalized)')
    lns2 = ax.plot(X, Y2, '-o', color='#999999', label='Number of {}s (Normalized)'.format(count_category_name))
    lns3 = ax.plot(X, Y3, '-^', color='#20708D', label='Number of Aircrafts (Normalized)')
    
    lns = lns1 + lns2 + lns3
    labels = [l.get_label() for l in lns]
    legend = ax.legend(lns, labels, bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white")
    legend.set_zorder(13)

    # Summary information
    ids = id_list(log_fname)
    avg_sim = sum(avg_sep_in_sec.values()) / len(avg_sep_in_sec.values())
    norm_avg_sim = sum(norm_avg_sep_timeseries) / len(norm_avg_sep_timeseries)
    sum_conf_time = sum([len(conf_num_in_sec[sec]) for sec in conf_num_in_sec.keys()])
    avg_conf_time = sum_conf_time / len(ids)
    norm_avg_conf_time = avg_conf_time / max_conf_num if max_conf_num != 0 else 0
    # Put background
    rect = ax.add_patch(Rectangle(
        (len(X)*0.02, 0.78), len(X)*0.75, 0.23,
        edgecolor = 'gray',
        facecolor = 'white',
        alpha = 0.9,
        fill=True,
        ))

    rect.set_zorder(9)
    # xy, width, height, angle
    analysis_summary_labels = ax.table(
        cellText=[
            ['The number of aircraft:'],
            ['Average separation score:'],
            ['Average normalized separation score:'],
            ['Total time of conflicts:'],
            ['Number of conflicted time/aircraft:'],
            ['Number of normalized conflicted time/aircraft:']
        ],
        bbox = [-0.20, 0.78, 0.9, 0.2], edges='open', cellLoc='right')

    analysis_summary = ax.table(
        cellText=[
            [' {}'.format(len(ids))],
            [' {:.2f}'.format(avg_sim)],
            [' {:.2f}'.format(norm_avg_sim)],
            [' {}'.format(sum_conf_time)],
            [' {:.2f}'.format(avg_conf_time)],
            [' {:.2f}'.format(norm_avg_conf_time)]
        ],
            bbox = [0.60, 0.78, 0.1, 0.2], edges='open', cellLoc='left')

    analysis_summary_labels.set_fontsize(13)
    analysis_summary.set_fontsize(13)
    analysis_summary_labels.set_zorder(10)
    analysis_summary.set_zorder(10)
    legend.set_zorder(10)
    #--------------
    ax.tick_params(axis='y', colors='#154B5F')
    ax.tick_params(axis='x', colors='#154B5F')
    # Set limit in Y1
    ax.set_ylim(-0.02, 1.02)
    #-----------
    ax.spines['bottom'].set_color('#154B5F')
    ax.spines['top'].set_color('#154B5F')
    ax.spines['left'].set_color('#154B5F')
    ax.spines['right'].set_color('#154B5F')

    plt.savefig(file_name)
    plt.close()

    print("Tactical Analysis, Pairwise {} Separation, Number of {}, to PNG completed.".format(sep_category_name, count_category_name))
    print("Tactical Analysis, Output file: {}.png \n".format(file_name))

 # Plot: Pairwise Separation Score with the Duration of Conflicts (or LoS) vs time
def timeseries_sep_and_aircrafts_plot_with_duration(file_name, sep_scores_in_sec, duration_list, count_category_name, sep_category_name):
    avg_sep_in_sec, max_sep_in_sec = sep_scores_in_sec

    # Set figure
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
    ax2 = ax1.twinx()
    ax1.set_title('Tactical Analysis, Pairwise Separation Score ({})'.format(sep_category_name), loc='center', fontsize=15, color='#154B5F')
    ax1.set_xlabel('Time [s]', fontsize=11, color='#154B5F')
    ax1.tick_params(axis='x', colors='#154B5F')  
    ax1.set_ylabel('Normalized Separation Score', fontsize=11, color='#154B5F')
    ax1.set_ylim(-0.02, 1.02)
    ax2.set_ylabel('Aircraft Pairs', fontsize=11, color='#154B5F')
    # Set X label
    secs = list(avg_sep_in_sec.keys())
    secs = sorted(secs, key=lambda x: float(x))
    X = np.array(secs)
    # Normalize sep & Set Y1 label
    norm_avg_sep_timeseries = np.array(list(avg_sep_in_sec.values())) / np.array(list(max_sep_in_sec.values()))
    Y1 = np.nan_to_num(norm_avg_sep_timeseries) # Change nan to 0
    # Plot
    ax1.tick_params(axis='y', colors='#154B5F')
    ax2.tick_params(axis='y', colors='#154B5F')

    ax2.yaxis.get_major_locator().set_params(integer=True) # Makes X axis ticks integer values, Aircrafts
    # Plot duration
    # duration_list -> [pairname:{s1, s2, ...}
    duration_start_list, duration_end_list = duration_list
    index_pairname_duration = dict()
    lns2 = None
    for idx, pairname in enumerate(duration_start_list.keys()):
        start_list = duration_start_list[pairname]
        end_list = duration_end_list[pairname]
        # Get index
        if pairname not in index_pairname_duration.keys():
            index_pairname_duration[pairname] = idx
        # Plot duration
        Y_duration = np.ones((len(start_list))) * index_pairname_duration[pairname]
        # Make plot
        # ax2.text(start_list[0], Y_duration[0], pairname, color='orange')
        lns2 = ax2.plot(start_list, Y_duration, '-D', color='orange', label='{} Duration'.format(count_category_name))
        ax2.plot(end_list, Y_duration, '-D', color='orange')
        # Make line
        start_pos = [[s, y] for s, y in zip(start_list, Y_duration)]
        end_pos = [[e, y] for e, y in zip(end_list, Y_duration)]
        colors = ['orange' for i in range(len(start_list))]
        lines = [[sp, ep] for sp, ep in zip(start_pos, end_pos)]
        lc = LineCollection(lines, colors=colors, linewidth=3)
        ax2.add_collection(lc)
        # -------------------------------------------------------------------------
    ax2.spines['bottom'].set_color('#154B5F')
    ax2.spines['top'].set_color('#154B5F')
    ax2.spines['left'].set_color('#154B5F')
    ax2.spines['right'].set_color('#154B5F')

    lns1 = ax1.plot(X, Y1, '-s', color='#329B8A', label='Average Separation Score for every second (Normalized)', zorder=10)

    lns = lns1 + lns2
    labels = [l.get_label() for l in lns]
    legend = ax2.legend(lns, labels, bbox_to_anchor=(0.01, 0.15), loc='upper left', borderaxespad=1, fontsize=11, facecolor="white") 

    ax1.set_facecolor("none")
    ax2.set_facecolor("none")

    ax1.set_zorder(10)
    ax2.set_zorder(7)

    plt.savefig(file_name)
    plt.close()

# Export heatmap
def heatmap_plot(file_name, aircraft_ids, conf_count_dic, los_count_dic, sep_dic):

    if "Spatial" in file_name:
        print("Simulation Analysis, Heatmap, Spatial Features, is started.")
    else:
        print("Simulation Analysis, Heatmap, All Features, is started.")

    text_size = 8
    fig = plt.figure(figsize=(12, 6), dpi=1000)
    # Plots are "Number of Conflicts", "Number of LoS", and "Normalized Separation Scores"
    axes = fig.subplots(nrows=1, ncols=3)
    # Get Normalized dic
    norm_sep_dic = get_norm_MinMax_sep_score(sep_dic.copy())
    # Create pd Dataframe
    df_column = ['Aircraft1', 'Aircraft2', 'Number of Conflict', 'Number of LoS', 'Separation Score']
    df_log = []
    for i, pair_aircraft_name in enumerate(sep_dic.keys()):
        aircraft1, aircraft2 = pair_aircraft_name.split('&')
        # Get sim score
        norm_score = norm_sep_dic[pair_aircraft_name]
        # Get num
        conf_num = conf_count_dic[pair_aircraft_name]
        los_num = los_count_dic[pair_aircraft_name]
        # Append data
        df_log.append([aircraft1, aircraft2, conf_num, los_num, norm_score])
    df = pd.DataFrame(df_log, columns=df_column)
    # Plot Conflicts heat map
    conf_pivot = pd.pivot_table(df, values='Number of Conflict', index='Aircraft1', columns='Aircraft2')
    if len(df) <= 10:
        conf_heat_map = sns.heatmap(conf_pivot, annot=True, fmt='.2g', cmap='Greens', ax=axes[0], square=True, cbar_kws={"orientation": "horizontal", "pad": 0.2})
    else:
        conf_heat_map = sns.heatmap(conf_pivot, cmap='Greens', ax=axes[0], square=True, cbar_kws={"orientation": "horizontal", "pad": 0.2})
    conf_heat_map.set_title('Number of Conflict')

    # Plot LoS heat map
    los_pivot = pd.pivot_table(
        df, values='Number of LoS', index='Aircraft1', columns='Aircraft2')
    if len(df) <= 10:
        los_heat_map = sns.heatmap(los_pivot, annot=True, fmt='.2g', cmap='Blues',
                                   ax=axes[1], square=True, cbar_kws={"orientation": "horizontal", "pad": 0.2})
    else:
        los_heat_map = sns.heatmap(los_pivot, cmap='Blues', ax=axes[1], square = True, cbar_kws = {"orientation": "horizontal", "pad": 0.2})
    los_heat_map.set_title('Number of LoS')

    # Plot Separation Score heat map
    score_pivot = pd.pivot_table(
        df, values='Separation Score', index='Aircraft1', columns='Aircraft2')
    if len(df) <= 10:
        score_heat_map = sns.heatmap(score_pivot, annot=True, fmt='.2g', cmap='Oranges', ax=axes[2], annot_kws={"size": 8}, square=True, cbar_kws={"orientation": "horizontal", "pad": 0.2})
    else:
        score_heat_map = sns.heatmap(score_pivot, cmap='Oranges', ax=axes[2], annot_kws={"size": 8}, square=True, cbar_kws={"orientation": "horizontal", "pad": 0.2})
    score_heat_map.set_title('Separation Score (Normalized)')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

    if "Spatial" in file_name:
        print("Simulation Analysis, Heatmap, Spatial Features, is completed.")
    else:
        print("Simulation Analysis, Heatmap, All Features, is completed.")

    print("Simulation Analysis, Output file: {}.png".format(file_name))

# -----------------------------------------------------------------------------

# [For ML TACTICAL]

def tactical_pairwise_spatial_conf_plot(log_fname, count_fname, mode, count_timeseries, period_fname=None, use_entire_content=False, output_extend=False):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]
    # Load log file
    d = data_import(log_fname, 1)
    # load conf_pair
    count_d = data_import(count_fname, 3)
    count_dic = None
    count_category_name = None
    if mode == 'conf':
        count_dic = output_pair_dic(count_d, ' confNum')
        count_category_name = 'Conflict'
    elif mode == 'los':
        count_dic = output_pair_dic(count_d, ' losNum')
        count_category_name = 'LoS'
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    secs = sorted(list(d['# simt'])[1:], key=lambda x: float(x))
    unique_secs, secsIndex = np.unique(secs, return_index=True)
    unique_secs = np.sort(np.array(unique_secs, np.float32))

    # Get aircrafts in sec
    aircrafts_in_sec = aircraft_num_in_sec_list(log_fname) # sec: [ids]
    # Load duration
    duration_d = None
    duration_start_dic, duration_end_dic = None, None
    if period_fname is not None:
        duration_d = data_import(period_fname, 2)
        duration_start_dic = output_pair_dic(duration_d, ' start duration Time[s]')
        duration_end_dic = output_pair_dic(duration_d, ' end duration Time[s]')

    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    # ----------------------------------------------------------------------------
    for idx_i, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output CSV with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for idx_j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[idx_j][k] = data_a[content]
                aircraft_b_data[idx_j][k] = data_b[content]
            # get sep score in time=sec
            euc_score = get_sep_score(aircraft_a_data.reshape(1, -1), aircraft_b_data.reshape(1, -1))
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = dict()
            sec_separation_dic[sec][key] = euc_score
    # -----------------------------------------------------------------------------------------
    # Pairwise -----------------------------------------
    # Append timeseries data
    aircraft_num_timeseries = []
    avg_sep_score_timeseries_dic = dict() # sec: score
    max_sep_score_timeseries_dic = dict() # sec: score
    all_pair_name_list = None

    for idx, sec in enumerate(sec_separation_dic.keys()):
        aircraft_num = len(aircrafts_in_sec[sec])
        aircraft_num_timeseries.append(aircraft_num)
        # get average
        sum_sec_sep = sum(sec_separation_dic[sec].values())
        avg_sec_sep =sum_sec_sep / len(sec_separation_dic[sec].keys())
        max_sec_sep = max(sec_separation_dic[sec].values())
        # get average in seconds
        avg_sep_score_timeseries_dic[sec] = avg_sec_sep
        # get max in seconds
        max_sep_score_timeseries_dic[sec] = max_sec_sep

    # Create plots
    output_avg_plot = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Tactical Pairwise {} Separation with {} for {}'.format(sep_category_name, count_category_name, log_name))
    timeseries_sep_and_aircrafts_plot(output_avg_plot, (avg_sep_score_timeseries_dic, max_sep_score_timeseries_dic),
                                      aircraft_num_timeseries, count_timeseries, count_category_name, sep_category_name, log_fname)
    # Single -------------------------------------------
    # single_separation_score_plot(log_fname, count_fname, count_timeseries, count_category_name, sep_category_name, calc_category_name='Local', use_tactical=True)
    # single_separation_score_plot(log_fname, count_fname, count_timeseries, count_category_name, sep_category_name, calc_category_name='General', use_tactical=True)

def output_tactical_plot(log_fname, count_fname, mode, count_timeseries, period_fname=None, use_entire_content=False, output_extend=False):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]
    # Load log file
    d = data_import(log_fname, 1)
    # load conf_pair
    count_d = data_import(count_fname, 3)
    count_dic = None
    count_category_name = None
    if mode == 'conf':
        count_dic = output_pair_dic(count_d, ' confNum')
        count_category_name = 'Conflict'
    elif mode == 'los':
        count_dic = output_pair_dic(count_d, ' losNum')
        count_category_name = 'LoS'
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    secs = sorted(list(d['# simt'])[1:], key=lambda x: float(x))
    unique_secs, secsIndex = np.unique(secs, return_index=True)
    unique_secs = np.sort(np.array(unique_secs, np.float32))

    # Get aircrafts in sec
    aircrafts_in_sec = aircraft_num_in_sec_list(log_fname) # sec: [ids]
    # Load duration
    duration_d = None
    duration_start_dic, duration_end_dic = None, None
    if period_fname is not None:
        duration_d = data_import(period_fname, 2)
        duration_start_dic = output_pair_dic(duration_d, ' start duration Time[s]')
        duration_end_dic = output_pair_dic(duration_d, ' end duration Time[s]')

    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    sec_separation_dic = {} # {time:{pair_name0 : sep_score0, pair_name1 : sep_score1, ...}}
    # ----------------------------------------------------------------------------
    for idx_i, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output CSV with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for idx_j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            sec = float(data_a['# simt'])
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[idx_j][k] = data_a[content]
                aircraft_b_data[idx_j][k] = data_b[content]
            # get sep score in time=sec
            euc_score = get_sep_score(aircraft_a_data.reshape(1, -1), aircraft_b_data.reshape(1, -1))
            if sec not in sec_separation_dic.keys():
                sec_separation_dic[sec] = dict()
            sec_separation_dic[sec][key] = euc_score
    # -----------------------------------------------------------------------------------------
    # Pairwise -----------------------------------------
    # Append timeseries data
    aircraft_num_timeseries = []
    avg_sep_score_timeseries_dic = dict() # sec: score
    max_sep_score_timeseries_dic = dict() # sec: score
    all_pair_name_list = None

    for idx, sec in enumerate(sec_separation_dic.keys()):
        aircraft_num = len(aircrafts_in_sec[sec])
        aircraft_num_timeseries.append(aircraft_num)
        # get average
        sum_sec_sep = sum(sec_separation_dic[sec].values())
        avg_sec_sep =sum_sec_sep / len(sec_separation_dic[sec].keys())
        max_sec_sep = max(sec_separation_dic[sec].values())
        # get average in seconds
        avg_sep_score_timeseries_dic[sec] = avg_sec_sep
        # get max in seconds
        max_sep_score_timeseries_dic[sec] = max_sec_sep

    # Create plots
    output_avg_plot = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Tactical Pairwise {} Separation with {} for {}'.format(sep_category_name, count_category_name, log_name))
    timeseries_sep_and_aircrafts_plot(output_avg_plot, (avg_sep_score_timeseries_dic, max_sep_score_timeseries_dic),
                                      aircraft_num_timeseries, count_timeseries, count_category_name, sep_category_name, log_fname)
    # Single -------------------------------------------
    single_separation_score_plot(log_fname, count_fname, count_timeseries, count_category_name, sep_category_name, calc_category_name='Local', use_tactical=True)
    single_separation_score_plot(log_fname, count_fname, count_timeseries, count_category_name, sep_category_name, calc_category_name='General', use_tactical=True)

    # [For ML TACTICAL EXTEND]
    if output_extend is True:
        # Extract duration list with count
        extracted_duration_start_dic, extracted_duration_end_dic = dict(), dict()
        for pairname, count_num in count_dic.items():
            if count_num != 0:
                extracted_duration_start_dic[pairname] = []
                extracted_duration_end_dic[pairname] = []
                for s, e in zip(duration_start_dic[pairname], duration_end_dic[pairname]):
                    extracted_duration_start_dic[pairname].append(s)
                    extracted_duration_end_dic[pairname].append(e)

        output_avg_plot = os.path.join(os.path.dirname(log_fname), 'USEPE_ML Tactical Pairwise {} Separation with {} Duration for {}'.format(sep_category_name, count_category_name, log_name))
        timeseries_sep_and_aircrafts_plot_with_duration(output_avg_plot, (avg_sep_score_timeseries_dic, max_sep_score_timeseries_dic), (duration_start_dic, duration_end_dic),
                                                        count_category_name, sep_category_name)

        single_separation_score_csv_and_plot_with_duration(log_fname, count_fname, count_timeseries, (extracted_duration_start_dic, extracted_duration_end_dic),
                                                           count_category_name=count_category_name, sep_category_name=sep_category_name, calc_category_name='Local', use_tactical=True)
        single_separation_score_csv_and_plot_with_duration(log_fname, count_fname, count_timeseries, (extracted_duration_start_dic, extracted_duration_end_dic),
                                                           count_category_name=count_category_name, sep_category_name=sep_category_name, calc_category_name='General', use_tactical=True)

def output_plots_from_count_dic(log_fname, count_fname, mode='conf', use_entire_content=True):
    
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]
    # Load log file
    d = data_import(log_fname, 1)
    # load conf_pair
    count_d = data_import(count_fname, 3)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    # Conf count ------------------------------------------------------------
    if mode == 'conf':
        count_dic = output_pair_dic(count_d, ' confNum')
        count_category_name = 'Conflict'
        Y0_name = 'Number of Conflicts (Normalized)'
    elif mode == 'los':
        count_dic = output_pair_dic(count_d, ' losNum')
        count_category_name = 'LoS'
        Y0_name = 'Number of LoS (Normalized)'
    # Sort by conflict num
    count_dic = dict(sorted(count_dic.items(), key=lambda x: x[1]))
    # --------------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for key in count_dic:
        aircraft_a, aircraft_b = key.split('&')
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_graph_dir = os.path.dirname(log_fname)

    # Output num & sep plots
    # num_and_sep_plot(file_name, count_dic, sep_dic, sep_category_name, Y0_name, Y1_name):
    conf_count_plot_name = os.path.join(output_graph_dir, 'USEPE_ML Pairwise {} {} for {}'.format(sep_category_name, count_category_name, log_name))
    num_and_sep_plot(conf_count_plot_name, unique_ids, count_dic, separation_dic, sep_category_name, Y0_name=Y0_name, Y1_name='Separation Score (Normalized)', mode=mode)

def output_plots_from_duration(log_fname, period_fname, mode='conf', use_entire_content=True):

    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]
    # Load log file
    d = data_import(log_fname, 1)
    # load duration
    duration_d = data_import(period_fname, 2)
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    # Get start time & endtime------------------------------------------------------------
    duration_start_dic = output_pair_dic(duration_d, ' start duration Time[s]')
    duration_end_dic = output_pair_dic(duration_d, ' end duration Time[s]')
    # --------------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    if mode == 'conf':
        count_category_name = 'Conflict'
    elif mode == 'los':
        count_category_name = 'LoS'
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output Plot (duration period&sep) with sep={} mode={}'.format(sep_category_name, mode))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)): # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
        # About duration --------------------------------------------------------------------------
        # Add conflict duration time as 0
        if (key not in duration_start_dic.keys()) or (key not in duration_end_dic.keys()):
            duration_start_dic[key] = [0]
            duration_end_dic[key] = [0]
        # Offset conflict duration time use -1
        else:
            for idx, (s, e) in enumerate(zip(duration_start_dic[key], duration_end_dic[key])):
                duration_start_dic[key][idx] = s
                duration_end_dic[key][idx] = e
        # -------------------------------------------------------------------------------------------

    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_graph_dir = os.path.dirname(log_fname)

    # Output num & sep plots
    # num_and_sep_plot(file_name, count_dic, sep_dic, sep_category_name, Y0_name, Y1_name):
    duration_plot_fname = os.path.join(output_graph_dir, 'USEPE_ML Pairwise {} {} Duration for {}'.format(sep_category_name, count_category_name, log_name))
    duration_and_sep_plot(duration_plot_fname, unique_ids, duration_start_dic, duration_end_dic, separation_dic, sep_category_name, mode)

def output_heatmap_from_conf_and_los_dic(log_fname, conf_fname, los_fname, use_entire_content=True):
    # About file path -------------------------------------------------------
    log_name = os.path.basename(log_fname)[:-4]
    # Load log file
    d = data_import(log_fname, 1)
    # load conf_pair
    count_conf_d = data_import(conf_fname, 3)
    count_conf_dic = output_pair_dic(count_conf_d, ' confNum')
    count_los_d = data_import(los_fname, 3)
    count_los_dic = output_pair_dic(count_los_d, ' losNum')
    # extract ids
    ids = list(d[' id'])[1:]
    unique_ids, idsIndex = np.unique(ids, return_index=True)
    # secs = list(d['# simt'])[1:]
    # sec, secIndex = np.unique(secs, return_index=True)
    # -----------------------------------------------------------------------
    # Define separation distance -------------------------------------------------
    # extract contents
    if use_entire_content is True:
        separation_contents = [' lat', ' lon', ' alt', ' distflown', ' hdg', ' trk', ' tas', ' gs', ' gsnorth', ' gseast', ' cas', ' M', ' selspd', ' aptas', ' selalt']
        sep_category_name = 'All Features'
    else:
        separation_contents = [' lat', ' lon', ' alt']
        sep_category_name = 'Spatial Features'
    separation_dic = {}
    # ----------------------------------------------------------------------------
    for idx, pair in enumerate(itertools.combinations(unique_ids, 2)):
        aircraft_a, aircraft_b = pair
        # print('Extract pair {}&{} in {}'.format(aircraft_a, aircraft_b, log_name))
        # print('\t for output Heatmap Plot with sep={}'.format(sep_category_name))
        key = '{}&{}'.format(aircraft_a, aircraft_b)
        # About Euclidean distance ----------------------------------------------------------------
        a_mask = np.where(d[' id'] == aircraft_a)[0]
        b_mask = np.where(d[' id'] == aircraft_b)[0]
        output_sep_len = len(list(zip(a_mask, b_mask)))
        aircraft_a_data = np.zeros((output_sep_len, len(separation_contents)))
        aircraft_b_data = np.zeros((output_sep_len, len(separation_contents)))
        # get separation content
        for j, (index_a, index_b) in enumerate(zip(a_mask, b_mask)):  # adjust mask len (for broadcast)
            data_a = d.iloc[index_a]
            data_b = d.iloc[index_b]
            # get value of each separation content
            for k, content in enumerate(separation_contents):
                aircraft_a_data[j][k] = data_a[content]
                aircraft_b_data[j][k] = data_b[content]
        # get separation
        euc_score = get_sep_score(aircraft_a_data, aircraft_b_data)
        separation_dic[key] = euc_score
        # -----------------------------------------------------------------------------------------
    # Sort separation score
    separation_dic = dict(sorted(separation_dic.items(), key=lambda x: x[1], reverse=True))
    # Set output plot dir
    output_plot_dir = os.path.dirname(log_fname)

    # Output
    heatmap_name = os.path.join(output_plot_dir, 'USEPE_ML Heatmap {} for {}'.format(sep_category_name, log_name))
    heatmap_plot(heatmap_name, unique_ids, count_conf_dic, count_los_dic, separation_dic)