import sys
sys.path.append("/auto/k2/abedel/Documents/Msc/terms/Msc_Term_1/DataProcessor/")

from GraphExtractor.fcCalculator.calFunctionalConnectivity import calculateFc

methods = {"fcMethod" : "distance", "numberOfConnectionsPerRoi": 400, "nfMethod" : None, "atlasName" : "Schaefer"}
fcMatrix_augments, labels_augments, centralCoordinates_augments, headMotion = calculateFc("HCP1200", "Rest1", 100206, 2, methods, None,  None)