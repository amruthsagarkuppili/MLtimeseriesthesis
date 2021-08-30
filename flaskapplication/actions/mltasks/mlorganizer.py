
######################################
#                                    #
# @author Amruth Sagar Kuppili       #
# @university Dalhousie University   #
# @ID B00844506                      #
#                                    #
######################################

from flaskapplication.actions.mltasks.cleandatahelper import cleandata
from flaskapplication.actions.mltasks.trainingmodelhelper import training
from flaskapplication.actions.mltasks.predictionhelper import prediction
from flaskapplication.actions.mltasks.variables import directoryloc
from flaskapplication.actions.schedulerjobs.schedulejobshelper import scheduler, base




import pandas as pd

from os import path
import os as os

cleandatainst = cleandata()
traininginst = training()
predictioninst = prediction()
class mlorganizer:

    def dataPreProcess(self, isSchedulerhit):
        ECCCdf, residualECCC = cleandatainst.cleanECCC()
        SMAdf, residualSMA = cleandatainst.cleanSMA()
        if all(temp is None for temp in [ECCCdf, SMAdf]):
            return None, None
        if not ((ECCCdf is None) or (SMAdf is None)):
            cleandatainst.combineData(ECCCdf, residualECCC, SMAdf, residualSMA, isSchedulerhit)
        return residualECCC, residualSMA


    def train(self, residualECCC, residualSMA):
        if path.exists(directoryloc+"accumulated_data_to_train.csv"):
            accdata = pd.read_csv(directoryloc+"accumulated_data_to_train.csv")
            traininginst.learnAlgorithm(accdata, residualECCC, residualSMA)
            os.remove(directoryloc+"accumulated_data_to_train.csv")
        else:
            combineddf = pd.read_csv(directoryloc+"SMA_ECCC_1H.csv")
            traininginst.learnAlgorithm(combineddf, residualECCC, residualSMA)
        return "Trained"


    def predict(self, ismaincall):
        infodict = predictioninst.predict(scheduler, base, ismaincall)
        return infodict


