""" This plugin manages all the data loggers for the USEPE project. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, traf, stack  # , settings, navdb, sim, scr, tools
from bluesky.tools import datalog

# List of the names of all the data loggers
loggers = ['USEPECONFLOG', 'USEPELOSLOG', 'USEPECPALOG', 'USEPEFLIGHTLOG']

# The data loggers
conflog = None
loslog = None
cpalog = None
flightlog = None
updateInterval = 1.0

# Parameters used when logging
confheader = \
    'CONFLICTS LOG\n' + \
    'Start and end of all conflicts\n\n' + \
    'Simulation Time [s], UAS1, UAS2, Start/End'

losheader = \
    'LOSS OF SEPARATION LOG\n' + \
    'Start and end of all loss of separation\n\n' + \
    'Simulation Time [s], UAS1, UAS2, Start/End'

cpaheader = \
    'CLOSEST POINT OF APPROACH LOG\n' + \
    'Shortest distance between 2 UAS during their conflict\n\n' + \
    'Simulation Time [s], UAS1, UAS2, Distance [m]'

flightheader = \
    'FLIGHT LOG\n' + \
    'Takeoff, landing, and distance flown of each UAS.\n\n' + \
    'Simulation Time [s], UAS, Takeoff/Landing, Distance [m]'

### Initialisation function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate the UsepeLogger entity
    usepelogger = UsepeLogger()

    # Create the loggers
    global conflog
    global loslog
    global cpalog
    global flightlog
    conflog = datalog.crelog('USEPECONFLOG', None, confheader)
    loslog = datalog.crelog('USEPELOSLOG', None, losheader)
    cpalog = datalog.crelog('USEPECPALOG', None, cpaheader)
    flightlog = datalog.crelog('USEPEFLIGHTLOG', None, flightheader)

    # Configuration parameters
    config = {
        'plugin_name': 'USEPELOGGER',
        'plugin_type': 'sim',
        'update_interval': updateInterval,
        'update': usepelogger.update
        }

    stackfunctions = {
        'USEPELOGGER': [
            'USEPELOGGER LIST/ON',
            'txt',
            usepelogger.usepelogger,
            'List/enable all the available data loggers'
        ]
    }

    # init_plugin() should always return a configuration dict.
    return config, stackfunctions

class UsepeLogger(core.Entity):
    ''' Provides the needed funcionality for each log. '''

    def __init__(self):
        super().__init__()

        # These lists stores the events from the previous step,
        # ensuring each event is logged only once and that we know when they have ended.
        self.prevconf = list()
        self.prevlos = list()

    def update(self):
        ''' Periodic function calling each logger function. '''
        self.conf_logger()
        self.los_logger()
        self.cpa_logger()

    def conf_logger(self):
        ''' Sorts current conflicts and logs new and ended events. '''
        currentconf = list()

        # Go through all conflict pairs and sort the IDs for easier matching
        currentconf = [sorted(pair) for pair in traf.cd.confpairs_unique]

        # Create lists of all new and ended conflicts
        startconf = [currpair for currpair in currentconf if currpair not in self.prevconf]
        endconf = [prevpair for prevpair in self.prevconf if prevpair not in currentconf]

        # Log start and end of conflicts
        conflog.log(startconf, 'start')
        conflog.log(endconf, 'end')

        # Store the new conflict environment
        self.prevconf = currentconf

    def los_logger(self):
        ''' Sorts current LoS and logs new and ended events. '''
        currentlos = list()

        # Go through all loss of separation pairs and sort the IDs for easier matching
        currentlos = [sorted(pair) for pair in traf.cd.lospairs_unique]

        # Create lists of all new and ended LoS
        startlos = [currpair for currpair in currentlos if currpair not in self.prevlos]
        endlos = [prevpair for prevpair in self.prevlos if prevpair not in currentlos]

        # Log start and end of LoS
        loslog.log(startlos, 'start')
        loslog.log(endlos, 'end')

        # Store the new loss of separation environment
        self.prevlos = currentlos

    def cpa_logger(self):
        ''' Logs CPA for each conflict. '''
        loggedpair = list()

        # For the best estimate, CPA is logged just after it happens (when tcpa turns negative)
        for x in range(len(traf.cd.confpairs)):
            pair = sorted(traf.cd.confpairs[x])
            if (-1 * updateInterval < traf.cd.tcpa[x] <= 0) and (pair not in loggedpair):
                cpalog.log(pair[0], pair[1], traf.cd.dcpa[x])
                loggedpair.append(pair)

    def create(self, n=1):
        super().create(n)

        takeoff = traf.id[-n:]

        flightlog.log(takeoff, 'takeoff', 0)

    def delete(self, idx):
        super().delete(idx)

        flightlog.log(traf.id[idx[0]], 'landing', traf.distflown[idx[0]])

    def usepelogger( self, cmd ):
        ''' USEPELOGGER command for the plugin.
            Options:
            LIST: List all the available data loggers for the project
            ON: Enable all the data loggers '''
        if cmd == 'LIST':
            return True, f'Available data loggers: {str.join(", ", loggers)}'
        elif cmd == 'ON':
            for x in range( len( loggers ) ):
                stack.stack( f'{loggers[x]} ON' )
            return True, f'All data loggers for USEPE enabled.'
        else:
            return False, f'Available commands are: LIST, ON'
