
import argparse

from experimentManager import experimenter
from run import experimentRunner



parser = argparse.ArgumentParser()

parser.add_argument("-eG", "--experimentGroup", type=str, default=None)
parser.add_argument("-r", "--reRun", type=str, default=False)

argv  = parser.parse_args()



if(not isinstance(argv.experimentGroup, type(None))):
        experimenter(argv.experimentGroup, argv.reRun)
else:

        from option import getOptions
        options = getOptions()

        options.expGroup = "Default"
        options.expName = "mainActivations"
         
        experimentRunner(options, True)

