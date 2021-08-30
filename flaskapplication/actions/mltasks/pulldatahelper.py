
######################################
#                                    #
# @author Amruth Sagar Kuppili       #
# @university Dalhousie University   #
# @ID B00844506                      #
#                                    #
######################################

import urllib.request
import zipfile
import io
import pandas as pd
import logging

from os import path



from flaskapplication.actions.mltasks.variables import directoryloc,predloc,ECCClink,SMAlink


class pullinformation:


    def comparePullData(self, pulledLatestDF, fileName, moduleName):
        residualexists = False
        if moduleName == "ECCC":
            extractexisteddf = pd.read_csv(fileName, index_col='DATE')
        else:
            extractexisteddf = pd.read_csv(fileName, index_col='time')
        residualdf = pulledLatestDF[~pulledLatestDF.index.isin(extractexisteddf.index)]
        residualdf.reset_index(inplace=True)
        extractexisteddf.reset_index(inplace=True)
        if moduleName == "SMA" and (not (residualdf.empty)):
            # extractexisteddf.drop("Unnamed: 0", axis=1, inplace=True)
            headerexisted = extractexisteddf.iloc[0:1, :]  # getting first row i.e header to be merged
            residualdf = pd.concat([headerexisted, residualdf])
        if not (residualdf.empty):
            residualexists = True
        return residualdf, residualexists



    def pullData(self, link, modulePull):
        df = pd.DataFrame()
        with urllib.request.urlopen(link) as f:
            onlinefile = f.read()
        filebytes = io.BytesIO(onlinefile)
        # link for ECCC is Zip
        if modulePull == 'ECCC':
            myzipfile = zipfile.ZipFile(filebytes, mode='r')
            df = pd.read_csv(myzipfile.open(myzipfile.namelist()[0]), index_col='DATE')
            if path.exists(directoryloc + "ECCC_latest_pull.csv"):
                residualECCCdf, residualECCCexists = self.comparePullData(df, directoryloc + "ECCC_latest_pull.csv",
                                                                          "ECCC")
                df.to_csv(directoryloc + "ECCC_latest_pull.csv")
                if residualECCCexists:
                    # residualECCCdf.to_csv("ECCC_residual.csv")
                    logging.debug("residual for ECCC")
                    return residualECCCdf, residualECCCexists
                else:
                    logging.debug("no residual for ECCC")
                    return None, False
            else:
                df.to_csv(directoryloc + "ECCC_latest_pull.csv")

        # Link for SMA is csv
        elif modulePull == 'SMA':
            df = pd.read_csv(filebytes, index_col='time')
            if path.exists(directoryloc + "SMA_latest_pull.csv"):
                residualSMAdf, residualSMAexists = self.comparePullData(df, directoryloc + "SMA_latest_pull.csv", "SMA")
                df.to_csv(directoryloc + "SMA_latest_pull.csv")
                if residualSMAexists:
                    # residualSMAdf.to_csv("SMA_residual.csv")
                    logging.debug("residual for SMA")
                    return residualSMAdf, residualSMAexists
                else:
                    logging.debug("no residual for SMA")
                    return None, False
            else:
                df.to_csv(directoryloc + "SMA_latest_pull.csv")
        df.reset_index(inplace=True)
        return df, False