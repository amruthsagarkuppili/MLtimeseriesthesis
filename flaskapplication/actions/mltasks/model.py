
######################################
#                                    #
# @author Amruth Sagar Kuppili       #
# @university Dalhousie University   #
# @ID B00844506                      #
#                                    #
######################################

# import urllib.request
# import zipfile
# import io
# import pandas as pd
# import logging
# import seaborn as sns
# import matplotlib.pyplot as plt
# from apscheduler.schedulers import SchedulerAlreadyRunningError, SchedulerNotRunningError
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import pickle
# from os import path
# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.triggers.interval import IntervalTrigger
# import os as os
# import apscheduler.schedulers.base as base
# from sklearn.linear_model import LassoCV
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from sklearn.ensemble import ExtraTreesRegressor
#
# # link for ECCC data
# ECCClink = 'http://www.meds-sdmm.dfo-mpo.gc.ca/alphapro/wave/waveshare/csvData/c44258_csv.zip'
# SMAlink = 'https://www.smartatlantic.ca/erddap/tabledap/SMA_halifax.csv?station_name%2Ctime%2Clongitude%2Clatitude%2Cwind_spd_avg%2Cwind_spd_max%2Cwind_dir_avg%2Cair_temp_avg%2Cair_pressure_avg%2Csurface_temp_avg%2Cwave_ht_max%2Cwave_ht_sig%2Cwave_dir_avg%2Cwave_spread_avg%2Cwave_period_max%2Ccurr_spd_avg%2Ccurr_dir_avg%2Ccurr_spd2_avg%2Ccurr_dir2_avg%2Ccurr_spd3_avg%2Ccurr_dir3_avg%2Ccurr_spd4_avg%2Ccurr_dir4_avg%2Ccurr_spd5_avg%2Ccurr_dir5_avg%2Ccurr_spd6_avg%2Ccurr_dir6_avg%2Ccurr_spd7_avg%2Ccurr_dir7_avg%2Ccurr_spd8_avg%2Ccurr_dir8_avg%2Ccurr_spd9_avg%2Ccurr_dir9_avg%2Ccurr_spd10_avg%2Ccurr_dir10_avg%2Ccurr_spd11_avg%2Ccurr_dir11_avg%2Ccurr_spd12_avg%2Ccurr_dir12_avg%2Ccurr_spd13_avg%2Ccurr_dir13_avg%2Ccurr_spd14_avg%2Ccurr_dir14_avg%2Ccurr_spd15_avg%2Ccurr_dir15_avg%2Ccurr_spd16_avg%2Ccurr_dir16_avg%2Ccurr_spd17_avg%2Ccurr_dir17_avg%2Ccurr_spd18_avg%2Ccurr_dir18_avg%2Ccurr_spd19_avg%2Ccurr_dir19_avg%2Ccurr_spd20_avg%2Ccurr_dir20_avg&time%3E=2013-11-07T16%3A23%3A01Z&time%3C=2020-09-07T18%3A53%3A01Z'
#
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
# scheduler = BackgroundScheduler()
#
#
# class MLservices:
#
#     # method to split time column and create date column
#     def parseDateECCC(self, data):
#         time = data["DATE & TIME"].split(" ")[0]
#         return time
#
#     # to merge rows 0 with index i.e to merge the units in the first row with the column headers.
#     def mergeRow(self, tdf, residual):
#         temp_c = tdf.columns.tolist()  # convert column headers to list and assign to a variable
#         temp_r = tdf[:][
#                  :1].values.tolist()  # Convert the values in the first row to a list object and assign to a variable
#         temp_row = [i for x in temp_r for i in x]  # assign values in the temp_r 2-dimension list to the temp_row list
#         # check if value in temp_row is nan (float) if not merge it with the column values and assign it to the temp_c list
#         temp_c = [temp_c[n] + " " + "unit(" + temp_row[n] + ")" if type(temp_row[n]) != float else temp_c[n] + "" for n
#                   in range(len(temp_c))]
#         tdf.columns = temp_c  # set the values of the column to the temp_c list
#         if residual:
#             tdf = tdf.iloc[1:, :]
#         else:
#             tdf.drop([0, 1], inplace=True)  # drop the first 2 row
#         tdf.reset_index(drop=True, inplace=True)  # reset the index to start from 0 again
#         return tdf  # return the transformed dataframe
#
#     # method to split time column and create date column for SMA
#     def parseDate(self, data):
#         time = data["time unit(UTC)"].split("T")[
#             0]  # split the time unit column and take on the date part of the column values
#         return time  # return the date value
#
#     # for SMA
#     def processDate(self, data):
#         date = data["time unit(UTC)"].replace("T", " ")  # replace T in date with " "
#         date = date.replace("Z", "")  # replace Z in date with nothing
#         return date  # return the new date value
#
#     def comparePullData(self, pulledLatestDF, fileName, moduleName):
#         residualexists = False
#         if moduleName == "ECCC":
#             extractexisteddf = pd.read_csv(fileName, index_col='DATE')
#         else:
#             extractexisteddf = pd.read_csv(fileName, index_col='time')
#         residualdf = pulledLatestDF[~pulledLatestDF.index.isin(extractexisteddf.index)]
#         residualdf.reset_index(inplace=True)
#         extractexisteddf.reset_index(inplace=True)
#         if moduleName == "SMA" and (not (residualdf.empty)):
#             # extractexisteddf.drop("Unnamed: 0", axis=1, inplace=True)
#             headerexisted = extractexisteddf.iloc[0:1, :]  # getting first row i.e header to be merged
#             residualdf = pd.concat([headerexisted, residualdf])
#         if not (residualdf.empty):
#             residualexists = True
#         return residualdf, residualexists
#
#     # Pull data with the given link
#     def pullData(self, link, modulePull):
#         df = pd.DataFrame()
#         with urllib.request.urlopen(link) as f:
#             onlinefile = f.read()
#         filebytes = io.BytesIO(onlinefile)
#         # link for ECCC is Zip
#         if modulePull == 'ECCC':
#             myzipfile = zipfile.ZipFile(filebytes, mode='r')
#             df = pd.read_csv(myzipfile.open(myzipfile.namelist()[0]), index_col='DATE')
#             if path.exists("ECCC_latest_pull.csv"):
#                 residualECCCdf, residualECCCexists = self.comparePullData(df, "ECCC_latest_pull.csv", "ECCC")
#                 df.to_csv("ECCC_latest_pull.csv")
#                 if residualECCCexists:
#                     # residualECCCdf.to_csv("ECCC_residual.csv")
#                     logging.debug("residual for ECCC")
#                     return residualECCCdf, residualECCCexists
#                 else:
#                     logging.debug("no residual for ECCC")
#                     return None, False
#             else:
#                 df.to_csv("ECCC_latest_pull.csv")
#
#         # Link for SMA is csv
#         elif modulePull == 'SMA':
#             df = pd.read_csv(filebytes, index_col='time')
#             if path.exists("SMA_latest_pull.csv"):
#                 residualSMAdf, residualSMAexists = self.comparePullData(df, "SMA_latest_pull.csv", "SMA")
#                 df.to_csv("SMA_latest_pull.csv")
#                 if residualSMAexists:
#                     # residualSMAdf.to_csv("SMA_residual.csv")
#                     logging.debug("residual for SMA")
#                     return residualSMAdf, residualSMAexists
#                 else:
#                     logging.debug("no residual for SMA")
#                     return None, False
#             else:
#                 df.to_csv("SMA_latest_pull.csv")
#         df.reset_index(inplace=True)
#         return df, False
#
#     # ECCC Data pulling and cleaning
#     def cleanECCC(self):
#         ECCCdf, residualECCC = self.pullData(ECCClink, "ECCC")
#         if not (ECCCdf is None):
#             temp_df = ECCCdf
#             ECCCdf.drop("Unnamed: 23", axis=1,
#                         inplace=True)  # drop the unnamed: 23 column that gets added during the read in
#             ECCCdf = ECCCdf.rename(columns={"DATE": "DATE & TIME"})
#             ECCCdf["DATE"] = ECCCdf.apply(lambda data: self.parseDateECCC(data), axis=1)
#             cols = ECCCdf.columns.tolist()  # rearrange columns
#             cols = cols[0:2] + cols[-1:] + cols[2:-1]
#             ECCCdf = ECCCdf[cols]
#             dateparse = lambda date: pd.datetime.strptime(date,
#                                                           '%m/%d/%Y %H:%M')  # anoymous function to turn time unit colum to datetime values
#             ECCCdf["DATE & TIME"] = ECCCdf["DATE & TIME"].apply(dateparse)  # applying the anonymous function
#             ECCCdf["LONGITUDE"] = ECCCdf[
#                                       "LONGITUDE"] * -1  # process longtitude column by  adding the minus to get actual coordinates
#             if residualECCC:
#                 ECCCdf.to_csv("ECCC_residual.csv", index=None, header=True)
#             else:
#                 ECCCdf.to_csv("c44258_cleaned.csv", index=None, header=True)  # save to file
#             return ECCCdf, residualECCC
#         return None, False
#
#     # SMA Data pulling and cleaning
#     def cleanSMA(self):
#         SMAdf, residualSMA = self.pullData(SMAlink, "SMA")
#         if not (SMAdf is None):
#             temp_df = SMAdf  # make a copy of file
#             SMAdf = self.mergeRow(SMAdf, residualSMA)  # Assign df to the returned dataframe from the mergeRow function
#             SMAdf["Date"] = SMAdf.apply(lambda data: self.parseDate(data),
#                                         axis=1)  # applying the functions defined in the sections above
#             SMAdf["time unit(UTC)"] = SMAdf.apply(lambda data: self.processDate(data),
#                                                   axis=1)  # applying the functions defined in the sections above
#             # rearrange columns
#             cols = SMAdf.columns.tolist()
#             cols = cols[0:2] + cols[-1:] + cols[2:-1]
#             SMAdf = SMAdf[cols]
#             dateparse = lambda date: pd.datetime.strptime(date,
#                                                           '%Y-%m-%d %H:%M:%S')  # anoymous function to turn time unit colum to datetime values
#             SMAdf["time unit(UTC)"] = SMAdf["time unit(UTC)"].apply(dateparse)  # applying the anonymous function
#             if residualSMA:
#                 SMAdf.to_csv("SMA_residual.csv", index=None, header=True)
#             else:
#                 SMAdf.to_csv("SMA_buoy_cleaned.csv", index=None, header=True)  # saving dataset to file
#             return SMAdf, residualSMA
#         return None, False
#
#     # Combine both ECCC and SMA data
#     def combineData(self, eccc, residualECCC, SMA, residualSMA):
#         eccc['DATE & TIME'] = eccc['DATE & TIME'].astype(str)
#         SMA['time unit(UTC)'] = SMA['time unit(UTC)'].astype(str)
#         # converting the date and time columns for both the ECCC and SMA buoys to datetime datatype
#         eccc_dateparse = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
#         SMA_dateparse = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
#         eccc["DATE & TIME"] = eccc["DATE & TIME"].apply(eccc_dateparse)
#         SMA['time unit(UTC)'] = SMA['time unit(UTC)'].apply(SMA_dateparse)
#         tempeccccols = eccc.columns.drop('DATE & TIME')
#         tempsmacols = SMA.columns.drop('time unit(UTC)')
#         eccc[tempeccccols] = eccc[tempeccccols].apply(pd.to_numeric, errors='ignore')
#         SMA[tempsmacols] = SMA[tempsmacols].apply(pd.to_numeric, errors='ignore')
#         # binning the SMA buoy dataset to one hour averages
#         SMA = SMA.groupby(pd.Grouper(key='time unit(UTC)', freq='1H')).mean()
#         # binning the ECCC buoy dataset to one hour averages
#         ecc = eccc.groupby(pd.Grouper(key='DATE & TIME', freq='1H')).mean()
#         # re-indexing the ECCC hourly buoy data to enable removing rows that don't align with the SMA buoy's and empty rows in
#         # the ECCC hourly buoy data
#         ecc.reset_index(inplace=True)
#         # ecc.index[ecc["DATE & TIME"] == "2013-11-07 16:00:00"] for checking the index number. this gives 120433
#         # dropping all the rows that aren't contained in the SMA buoy
#         # ecc.drop(ecc.index[0:120433], inplace=True, axis=0)
#         # reseting the index of the eccc hourly buoy back to the date time column
#         # ecc.set_index('DATE & TIME', inplace=True)
#         df = pd.DataFrame()  # creating an empty dataframe to hold the combined ECCC & SMA data values
#
#         # below get first row of the ECCC and SMA hourly data value in a dataframe variable
#         if residualECCC and residualSMA:
#             ecc.set_index('DATE & TIME', inplace=True)
#             ec = pd.DataFrame(ecc[["VCAR", "VWH$", "VCMX", "VTP$", "WDIR", "WSPD", "GSPD"]].iloc[0])
#             sm = pd.DataFrame(SMA[["wave_ht_sig unit(m)"]].iloc[0])
#         else:
#             ecc.drop(ecc.index[0:120433], inplace=True, axis=0)
#             ecc.set_index('DATE & TIME', inplace=True)
#             ec = pd.DataFrame(ecc.loc["2013-11-07 16:00:00", ["VCAR", "VWH$", "VCMX", "VTP$", "WDIR", "WSPD", "GSPD"]])
#             sm = pd.DataFrame(SMA.loc["2013-11-07 16:00:00", ["wave_ht_sig unit(m)"]])
#         ec, sm = ec.T, sm.T  # Transpose the returned dataframe
#         df = pd.concat([ec, sm], axis=1)  # combine the first rows of both buoys and save them in a new dataframe
#         # go through all rows in the both the ECCC and SMA hourly datasets and add them to the dataframe
#         for i in ecc.index[1:-1]:
#             if i in SMA.index:
#                 NOA = pd.DataFrame(ecc.loc[i, ["VCAR", "VWH$", "VCMX", "VTP$", "WDIR", "WSPD", "GSPD"]])
#                 Smart = pd.DataFrame(SMA.loc[i, ["wave_ht_sig unit(m)"]])
#                 NOA, Smart = NOA.T, Smart.T
#                 df1 = pd.concat([NOA, Smart], axis=1)
#                 df = df.append(df1)
#         df.reset_index(inplace=True)  # reseting index to enable dropping empty rows in the dataset
#         # ECCC has no data for below dates
#         # retrieving the index for the first empty row
#         # df.index[df["index"] == "2015-07-31 07:00:00"] gives 15135
#         # retrieving the index for the last empty row
#         # df.index[df["index"] == "2018-10-29 21:00:00"] gives 43613
#         if not (residualECCC or residualSMA):
#             df.drop(df.index[15135:43613], inplace=True, axis=0)
#         df.set_index('index', inplace=True)
#         eccc.set_index('DATE & TIME', inplace=True)
#         # logging.debug(eccc.dtypes)
#         # create a list to hold the indexes in the original ECCC buoy dataset
#         position = []
#         if not (residualECCC and residualSMA):
#
#             for i in df.index:
#                 position.append(eccc.index[eccc.index.get_loc(i, method='ffill')])
#             obj = eccc.loc[position[24].to_datetime64(), ["VCAR"]]
#         else:
#             for i in df.index:
#                 position.append(eccc.index[eccc.index.get_loc(i, method='nearest')])
#         # creating columns for the last VCAR and VWH$ before the hourly interval i.e it'll store the last value recorded before
#         # the hour
#         df["last_VCAR"] = ""
#         df["last_VWH$"] = ""
#
#         # method to get the last values before VCAR and VWH$ values before the hourly interval and store them in the
#         # "last_VCAR" and "last_VWH$" columns
#         def create_cols(data):
#             method = "ffill" if not (residualECCC and residualSMA) else "nearest"
#             for i in data.index:
#                 data.loc[[i], ["last_VCAR"]] = eccc.loc[eccc.index[eccc.index.get_loc(i, method=method)], ["VCAR"]][0]
#                 data.loc[[i], ["last_VWH$"]] = eccc.loc[eccc.index[eccc.index.get_loc(i, method=method)], ["VWH$"]][0]
#             return data
#
#         df = create_cols(df)
#         # rearranging the columns
#         cols = df.columns.tolist()
#         cols = cols[0:7] + cols[-2:] + cols[7:-2]
#         df = df[cols]
#
#         df.to_csv("SMA_ECCC_1H_only_wave_ht_residual.csv", header=True) if (
#                 residualECCC and residualSMA) else df.to_csv("SMA_ECCC_1H_only_wave_ht.csv",
#                                                              header=True)  # saving dataset to file
#         return df
#
#     # Training the algorithm
#     def learnAlgorithm(self, df, residualECCC, residualSMA):
#         df.set_index("index", inplace=True)
#         # save the list of the dataframe's column in the cols variable for easy manipulation
#         cols = df.columns.tolist()
#         # creating feature variables from cols and saving it in cols2
#         cols2 = cols[:6]
#         # dropping empty and non-number rows and then checking the dimensions again
#         df.dropna(inplace=True)
#         # splitting the data into train test bins and settign the target value
#         target = df['wave_ht_sig unit(m)']
#         if not (residualECCC and residualSMA):
#             X_train, X_test, y_train, y_test = train_test_split(df[cols2], target, test_size=0.2, random_state=0)
#             # scaling the features to ensure they all have the same scales
#             sc = StandardScaler()
#             sc.fit(X_train)
#             # scaling the features
#             X_train_std = sc.transform(X_train)
#             X_test_std = sc.transform(X_test)
#             # training and testing the model
#             rfr_1h_wave_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=109,
#                                                       max_features='auto', max_leaf_nodes=None,
#                                                       min_impurity_decrease=0.0, min_impurity_split=None,
#                                                       min_samples_leaf=2, min_samples_split=9,
#                                                       min_weight_fraction_leaf=0.0, n_estimators=2000,
#                                                       oob_score=False, random_state=None,
#                                                       verbose=0, warm_start=False)
#             rfr_1h_wave_model.fit(X_train_std, y_train)
#         else:
#             sc = StandardScaler()
#             sc.fit(df[cols2])
#             features = sc.transform(df[cols2])
#             loaded_model = pickle.load(open("rfr_1h_wave_model.p", 'rb'))
#             loaded_model.fit(features, target)
#             rfr_1h_wave_model = loaded_model
#         # pred = rfr_1h_wave_model.predict(X_test_std)
#         # mae = str(mean_absolute_error(y_test, pred))
#         # mse = str(mean_squared_error(y_test, pred))
#         # logging.debug("Mean absolute error(MAE):{}".format(mae))
#         # logging.debug("Mean squared error(MSE):{}".format(mse))
#         # using pickle to save the model
#         # if residualECCC and residualSMA:
#         #     os.remove("ECCC_residual.csv")
#         #     os.remove("SMA_residual.csv")
#         pickle.dump(rfr_1h_wave_model, open("rfr_1h_wave_model.p", "wb"))
#
#     # ALL scheduler functions start from here
#
#     # @scheduler.scheduled_job(MLservices.deleteUpdateFile(),'cron', id='delete_file_sched', year='*', month='*', day='*', week='*', day_of_week='*', hour='*', minute=1, second=0)
#     def deleteUpdateFile(self):
#         logging.debug("deleted")
#         os.remove("/Users/amruthkuppili/Desktop/proj/SMAFlask/ECCC_residual.csv")
#         os.remove("/Users/amruthkuppili/Desktop/proj/SMAFlask/SMA_residual.csv")
#
#     def train(self):
#         ECCCdf, residualECCC = self.cleanECCC()
#         SMAdf, residualSMA = self.cleanSMA()
#         if all(temp is None for temp in [ECCCdf, SMAdf]):
#             return "Already trained on Latest Pull"
#         if not ((ECCCdf is None) or (SMAdf is None)):
#             self.combineData(ECCCdf, residualECCC, SMAdf, residualSMA)
#             combineddf = pd.read_csv("SMA_ECCC_1H_only_wave_ht.csv")
#             self.learnAlgorithm(combineddf, residualECCC, residualSMA)
#         return "Trained"
#
#     def predict(self):
#         if path.exists("SMA_ECCC_1H_only_wave_ht_residual.csv"):
#             residualdf = pd.read_csv("SMA_ECCC_1H_only_wave_ht_residual.csv")
#             residualdf.set_index("index", inplace=True)
#             cols = residualdf.columns.tolist()
#             cols2 = cols[:6]
#             residualdf.dropna(inplace=True)
#             # residualdf.drop("wave_ht_sig unit(m)", axis=1, inplace=True)
#             sc = StandardScaler()
#             sc.fit(residualdf[cols2])
#             features = sc.transform(residualdf[cols2])
#             loadedmodel = pickle.load(open("rfr_1h_wave_model.p", 'rb'))
#             prediction = loadedmodel.predict(features)
#             originalvalue = residualdf["wave_ht_sig unit(m)"][0]
#             return {"predictedvalue":prediction[0], "originalvalue":originalvalue, "message":None}
#
#         return {"message":"system has started to train, hang on for a moment..... "} if scheduler.state == base.STATE_RUNNING else {"message":"Click on Train Now for immediate prediction"}
#
#
# class ScheduleJobs:
#
#     def deleteUpdateFile(self):
#         logging.debug("deleted")
#         os.remove("/Users/amruthkuppili/Desktop/proj/SMAFlask/ECCC_residual.csv")
#
#     def startPrediction(self):
#         mls = MLservices()
#         mls.train()
#         mls.predict()
#
#
# class OperateSchedulers:
#
#     def jobsForEveryHour(self):
#         if scheduler.state != base.STATE_RUNNING:
#             scheduler.start()
#             schedulejobs = ScheduleJobs()
#             scheduler.add_job(schedulejobs.deleteUpdateFile, 'cron', id='delete_file_sched', year='*', month='*',
#                               day='*',
#                               week='*', day_of_week='*', hour='*', minute=55, second=0)
#             scheduler.add_job(schedulejobs.startPrediction, 'cron', id='start_prediction_sched', year='*', month='*',
#                               day='*',
#                               week='*', day_of_week='*', hour='*', minute=56, second=0)
#             return "Scheduler Jobs Started"
#         else:
#             return "Jobs Already Running"
#
#     def stopJobs(self):
#         if scheduler.state != base.STATE_STOPPED:
#             scheduler.shutdown(wait=True)
#             return "jobs are removed and operation is shutting down....."
#         else:
#             return "No Scheduler is running"
#
# # scheduler.remove_job('delete_file_sched')
# # scheduler.shutdown()








# # # # # # # # # # # # # # # # # # # # #
#
# Version 2 i.e including all variables
#
# # # # # # # # # # # # # # # # # # # # #


import urllib.request
import zipfile
import io
import pandas as pd
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from apscheduler.schedulers import SchedulerAlreadyRunningError, SchedulerNotRunningError
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from os import path
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import os as os
import apscheduler.schedulers.base as base
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import ExtraTreesRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, RNN
from tensorflow import keras



# link for ECCC data
ECCClink = 'http://www.meds-sdmm.dfo-mpo.gc.ca/alphapro/wave/waveshare/csvData/c44258_csv.zip'
SMAlink = 'https://www.smartatlantic.ca/erddap/tabledap/SMA_halifax.csv?station_name%2Ctime%2Clongitude%2Clatitude%2Cwind_spd_avg%2Cwind_spd_max%2Cwind_dir_avg%2Cair_temp_avg%2Cair_pressure_avg%2Csurface_temp_avg%2Cwave_ht_max%2Cwave_ht_sig%2Cwave_dir_avg%2Cwave_spread_avg%2Cwave_period_max%2Ccurr_spd_avg%2Ccurr_dir_avg%2Ccurr_spd2_avg%2Ccurr_dir2_avg%2Ccurr_spd3_avg%2Ccurr_dir3_avg%2Ccurr_spd4_avg%2Ccurr_dir4_avg%2Ccurr_spd5_avg%2Ccurr_dir5_avg%2Ccurr_spd6_avg%2Ccurr_dir6_avg%2Ccurr_spd7_avg%2Ccurr_dir7_avg%2Ccurr_spd8_avg%2Ccurr_dir8_avg%2Ccurr_spd9_avg%2Ccurr_dir9_avg%2Ccurr_spd10_avg%2Ccurr_dir10_avg%2Ccurr_spd11_avg%2Ccurr_dir11_avg%2Ccurr_spd12_avg%2Ccurr_dir12_avg%2Ccurr_spd13_avg%2Ccurr_dir13_avg%2Ccurr_spd14_avg%2Ccurr_dir14_avg%2Ccurr_spd15_avg%2Ccurr_dir15_avg%2Ccurr_spd16_avg%2Ccurr_dir16_avg%2Ccurr_spd17_avg%2Ccurr_dir17_avg%2Ccurr_spd18_avg%2Ccurr_dir18_avg%2Ccurr_spd19_avg%2Ccurr_dir19_avg%2Ccurr_spd20_avg%2Ccurr_dir20_avg&time%3E=2013-11-07T16%3A23%3A01Z&time%3C=2020-09-07T18%3A53%3A01Z'

directoryloc = 'flaskapplication/actions/generatedcsv/'
predloc = '/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/generatedcsv/predictions.csv'

logging.basicConfig(filename='example.log', level=logging.DEBUG)
scheduler = BackgroundScheduler()


class MLservices:

    # method to split time column and create date column
    def parseDateECCC(self, data):
        time = data["DATE & TIME"].split(" ")[0]
        return time

    # to merge rows 0 with index i.e to merge the units in the first row with the column headers.
    def mergeRow(self, tdf, residual):
        temp_c = tdf.columns.tolist()  # convert column headers to list and assign to a variable
        temp_r = tdf[:][
                 :1].values.tolist()  # Convert the values in the first row to a list object and assign to a variable
        temp_row = [i for x in temp_r for i in x]  # assign values in the temp_r 2-dimension list to the temp_row list
        # check if value in temp_row is nan (float) if not merge it with the column values and assign it to the temp_c list
        temp_c = [temp_c[n] + " " + "unit(" + temp_row[n] + ")" if type(temp_row[n]) != float else temp_c[n] + "" for n
                  in range(len(temp_c))]
        tdf.columns = temp_c  # set the values of the column to the temp_c list
        if residual:
            tdf = tdf.iloc[1:, :]
        else:
            tdf.drop([0, 1], inplace=True)  # drop the first 2 row
        tdf.reset_index(drop=True, inplace=True)  # reset the index to start from 0 again
        return tdf  # return the transformed dataframe

    # method to split time column and create date column for SMA
    def parseDate(self, data):
        time = data["time unit(UTC)"].split("T")[
            0]  # split the time unit column and take on the date part of the column values
        return time  # return the date value

    # for SMA
    def processDate(self, data):
        date = data["time unit(UTC)"].replace("T", " ")  # replace T in date with " "
        date = date.replace("Z", "")  # replace Z in date with nothing
        return date  # return the new date value

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

    # Pull data with the given link
    def pullData(self, link, modulePull):
        df = pd.DataFrame()
        with urllib.request.urlopen(link) as f:
            onlinefile = f.read()
        filebytes = io.BytesIO(onlinefile)
        # link for ECCC is Zip
        if modulePull == 'ECCC':
            myzipfile = zipfile.ZipFile(filebytes, mode='r')
            df = pd.read_csv(myzipfile.open(myzipfile.namelist()[0]), index_col='DATE')
            if path.exists(directoryloc+"ECCC_latest_pull.csv"):
                residualECCCdf, residualECCCexists = self.comparePullData(df, directoryloc+"ECCC_latest_pull.csv", "ECCC")
                df.to_csv(directoryloc+"ECCC_latest_pull.csv")
                if residualECCCexists:
                    # residualECCCdf.to_csv("ECCC_residual.csv")
                    logging.debug("residual for ECCC")
                    return residualECCCdf, residualECCCexists
                else:
                    logging.debug("no residual for ECCC")
                    return None, False
            else:
                df.to_csv(directoryloc+"ECCC_latest_pull.csv")

        # Link for SMA is csv
        elif modulePull == 'SMA':
            df = pd.read_csv(filebytes, index_col='time')
            if path.exists(directoryloc+"SMA_latest_pull.csv"):
                residualSMAdf, residualSMAexists = self.comparePullData(df, directoryloc+"SMA_latest_pull.csv", "SMA")
                df.to_csv(directoryloc+"SMA_latest_pull.csv")
                if residualSMAexists:
                    # residualSMAdf.to_csv("SMA_residual.csv")
                    logging.debug("residual for SMA")
                    return residualSMAdf, residualSMAexists
                else:
                    logging.debug("no residual for SMA")
                    return None, False
            else:
                df.to_csv(directoryloc+"SMA_latest_pull.csv")
        df.reset_index(inplace=True)
        return df, False

    # ECCC Data pulling and cleaning
    def cleanECCC(self):
        ECCCdf, residualECCC = self.pullData(ECCClink, "ECCC")
        if not (ECCCdf is None):
            temp_df = ECCCdf
            ECCCdf.drop("Unnamed: 23", axis=1,
                        inplace=True)  # drop the unnamed: 23 column that gets added during the read in
            ECCCdf = ECCCdf.rename(columns={"DATE": "DATE & TIME"})
            ECCCdf["DATE"] = ECCCdf.apply(lambda data: self.parseDateECCC(data), axis=1)
            cols = ECCCdf.columns.tolist()  # rearrange columns
            cols = cols[0:2] + cols[-1:] + cols[2:-1]
            ECCCdf = ECCCdf[cols]
            dateparse = lambda date: pd.datetime.strptime(date,
                                                          '%m/%d/%Y %H:%M')  # anoymous function to turn time unit colum to datetime values
            ECCCdf["DATE & TIME"] = ECCCdf["DATE & TIME"].apply(dateparse)  # applying the anonymous function
            ECCCdf["LONGITUDE"] = ECCCdf[
                                      "LONGITUDE"] * -1  # process longtitude column by  adding the minus to get actual coordinates
            if residualECCC:
                ECCCdf.to_csv(directoryloc+"ECCC_residual.csv", index=None, header=True)
            else:
                ECCCdf.to_csv(directoryloc+"c44258_cleaned.csv", index=None, header=True)  # save to file
            return ECCCdf, residualECCC
        return None, False

    # SMA Data pulling and cleaning
    def cleanSMA(self):
        SMAdf, residualSMA = self.pullData(SMAlink, "SMA")
        if not (SMAdf is None):
            temp_df = SMAdf  # make a copy of file
            SMAdf = self.mergeRow(SMAdf, residualSMA)  # Assign df to the returned dataframe from the mergeRow function
            SMAdf["Date"] = SMAdf.apply(lambda data: self.parseDate(data),
                                        axis=1)  # applying the functions defined in the sections above
            SMAdf["time unit(UTC)"] = SMAdf.apply(lambda data: self.processDate(data),
                                                  axis=1)  # applying the functions defined in the sections above
            # rearrange columns
            cols = SMAdf.columns.tolist()
            cols = cols[0:2] + cols[-1:] + cols[2:-1]
            SMAdf = SMAdf[cols]
            dateparse = lambda date: pd.datetime.strptime(date,
                                                          '%Y-%m-%d %H:%M:%S')  # anoymous function to turn time unit colum to datetime values
            SMAdf["time unit(UTC)"] = SMAdf["time unit(UTC)"].apply(dateparse)  # applying the anonymous function
            if residualSMA:
                SMAdf.to_csv(directoryloc+"SMA_residual.csv", index=None, header=True)
            else:
                SMAdf.to_csv(directoryloc+"SMA_buoy_cleaned.csv", index=None, header=True)  # saving dataset to file
            return SMAdf, residualSMA
        return None, False

    # Combine both ECCC and SMA data
    def combineData(self, eccc, residualECCC, SMA, residualSMA, isSchedulerhit):
        eccc['DATE & TIME'] = eccc['DATE & TIME'].astype(str)
        SMA['time unit(UTC)'] = SMA['time unit(UTC)'].astype(str)
        # converting the date and time columns for both the ECCC and SMA buoys to datetime datatype
        eccc_dateparse = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        SMA_dateparse = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        eccc["DATE & TIME"] = eccc["DATE & TIME"].apply(eccc_dateparse)
        SMA['time unit(UTC)'] = SMA['time unit(UTC)'].apply(SMA_dateparse)
        tempeccccols = eccc.columns.drop('DATE & TIME')
        tempsmacols = SMA.columns.drop('time unit(UTC)')
        eccc[tempeccccols] = eccc[tempeccccols].apply(pd.to_numeric, errors='ignore')
        SMA[tempsmacols] = SMA[tempsmacols].apply(pd.to_numeric, errors='ignore')
        # binning the SMA buoy dataset to one hour averages
        SMA = SMA.groupby(pd.Grouper(key='time unit(UTC)', freq='1H')).mean()
        # binning the ECCC buoy dataset to one hour averages
        ecc = eccc.groupby(pd.Grouper(key='DATE & TIME', freq='1H')).mean()
        # re-indexing the ECCC hourly buoy data to enable removing rows that don't align with the SMA buoy's and empty rows in
        # the ECCC hourly buoy data
        ecc.reset_index(inplace=True)
        # ecc.index[ecc["DATE & TIME"] == "2013-11-07 16:00:00"] for checking the index number. this gives 120433
        # dropping all the rows that aren't contained in the SMA buoy
        if not (residualECCC or residualSMA):
            ecc.drop(ecc.index[0:120433], inplace=True, axis=0)
        # reseting the index of the eccc hourly buoy back to the date time column
        # ecc.set_index('DATE & TIME', inplace=True)
        df = pd.DataFrame()  # creating an empty dataframe to hold the combined ECCC & SMA data values
        ecc.set_index('DATE & TIME', inplace=True)
        # below get first row of the ECCC and SMA hourly data value in a dataframe variable
        # if residualECCC and residualSMA:
        #     ecc.set_index('DATE & TIME', inplace=True)
        #     ec = pd.DataFrame(ecc[["VCAR", "VWH$", "VCMX", "VTP$", "WDIR", "WSPD", "GSPD"]].iloc[0])
        #     sm = pd.DataFrame(SMA[["wave_ht_sig unit(m)","wind_spd_avg unit(m s-1)","wave_ht_max unit(m)","wave_dir_avg unit(degree)","wave_period_max unit(s)"]].iloc[0])
        # else:
        #     ecc.drop(ecc.index[0:120433], inplace=True, axis=0)
        #     ecc.set_index('DATE & TIME', inplace=True)
        #     ec = pd.DataFrame(ecc.loc["2013-11-07 16:00:00", ["VCAR", "VWH$", "VCMX", "VTP$", "WDIR", "WSPD", "GSPD"]])
        #     sm = pd.DataFrame(SMA.loc["2013-11-07 16:00:00", ["wave_ht_sig unit(m)","wind_spd_avg unit(m s-1)","wave_ht_max unit(m)","wave_dir_avg unit(degree)","wave_period_max unit(s)"]])
        # ec, sm = ec.T, sm.T  # Transpose the returned dataframe
        # df = pd.concat([ec, sm], axis=1)  # combine the first rows of both buoys and save them in a new dataframe
        # go through all rows in the both the ECCC and SMA hourly datasets and add them to the dataframe
        for i in ecc.index:
            if i in SMA.index:
                NOA = pd.DataFrame(ecc.loc[i, ["VCAR", "VWH$", "VCMX", "VTP$", "WDIR", "WSPD", "GSPD"]])
                Smart = pd.DataFrame(SMA.loc[i, ["wave_ht_sig unit(m)","wind_spd_avg unit(m s-1)","wave_ht_max unit(m)","wave_dir_avg unit(degree)","wave_period_max unit(s)"]])
                NOA, Smart = NOA.T, Smart.T
                df1 = pd.concat([NOA, Smart], axis=1)
                df = df.append(df1)
        df.reset_index(inplace=True)  # reseting index to enable dropping empty rows in the dataset
        # ECCC has no data for below dates
        # retrieving the index for the first empty row
        # df.index[df["index"] == "2015-07-31 07:00:00"] gives 15135
        # retrieving the index for the last empty row
        # df.index[df["index"] == "2018-10-29 21:00:00"] gives 43613
        if not (residualECCC or residualSMA):
            df.drop(df.index[15135:43613], inplace=True, axis=0)
        df.set_index('index', inplace=True)
        eccc.set_index('DATE & TIME', inplace=True)
        # logging.debug(eccc.dtypes)
        # create a list to hold the indexes in the original ECCC buoy dataset
        position = []
        if not (residualECCC and residualSMA):

            for i in df.index:
                position.append(eccc.index[eccc.index.get_loc(i, method='ffill')])
            obj = eccc.loc[position[24].to_datetime64(), ["VCAR"]]
        else:
            for i in df.index:
                position.append(eccc.index[eccc.index.get_loc(i, method='nearest')])
        # creating columns for the last VCAR and VWH$ before the hourly interval i.e it'll store the last value recorded before
        # the hour
        df["last_VCAR"] = ""
        df["last_VWH$"] = ""

        # method to get the last values before VCAR and VWH$ values before the hourly interval and store them in the
        # "last_VCAR" and "last_VWH$" columns
        def create_cols(data):
            method = "ffill" if not (residualECCC and residualSMA) else "nearest"
            for i in data.index:
                data.loc[[i], ["last_VCAR"]] = eccc.loc[eccc.index[eccc.index.get_loc(i, method=method)], ["VCAR"]][0]
                data.loc[[i], ["last_VWH$"]] = eccc.loc[eccc.index[eccc.index.get_loc(i, method=method)], ["VWH$"]][0]
            return data

        df = create_cols(df)
        # rearranging the columns
        cols = df.columns.tolist()
        cols = cols[0:7] + cols[-2:] + cols[7:-2]
        df = df[cols]
        df.to_csv(directoryloc+"SMA_ECCC_1H.csv", header=True)

        if (residualECCC and residualSMA):
            df.to_csv(directoryloc+"SMA_ECCC_1H_residual.csv", header=True)
        else:
            dfend = df.tail(1)
            dfend.to_csv(directoryloc+"SMA_ECCC_1H_residual.csv", header=True)

        if isSchedulerhit:
            if path.exists(directoryloc+"accumulated_data_to_train.csv"):
                accdata = pd.read_csv(directoryloc+"accumulated_data_to_train.csv")
                accdata = pd.concat([accdata, df], axis=1)
                accdata.to_csv(directoryloc+"accumulated_data_to_train.csv", header=True)
            else:
                df.to_csv(directoryloc+"accumulated_data_to_train.csv", header=True)


        # df.to_csv("SMA_ECCC_1H_residual.csv", header=True) if (
        #         residualECCC and residualSMA) else (df.to_csv("SMA_ECCC_1H.csv",header=True)
        #                                             and dfend.to_csv("SMA_ECCC_1H_residual.csv",header=True))  # saving dataset to file
        return df

    # Training the algorithm
    def learnAlgorithm(self, df, residualECCC, residualSMA):

        df.set_index("index", inplace=True)
        # save the list of the dataframe's column in the cols variable for easy manipulation
        cols = df.columns.tolist()
        # creating feature variables from cols and saving it in cols2
        cols2 = cols[:7]
        # dropping empty and non-number rows and then checking the dimensions again
        df.dropna(inplace=True)
        # splitting the data into train test bins and settign the target value
        target = df[['wave_ht_sig unit(m)','wind_spd_avg unit(m s-1)','wave_ht_max unit(m)','wave_period_max unit(s)']]
        if not (residualECCC and residualSMA):
            X_train, X_test, y_train, y_test = train_test_split(df[cols2], target, test_size=0.2, random_state=0)
            # scaling the features to ensure they all have the same scales
            sc = StandardScaler()
            sc.fit(X_train)
            # scaling the features
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            # training and testing the model
            rfr_1h_wave_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=109,
                                                      max_features='auto', max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0, min_impurity_split=None,
                                                      min_samples_leaf=2, min_samples_split=9,
                                                      min_weight_fraction_leaf=0.0, n_estimators=2000,
                                                      oob_score=False, random_state=None,
                                                      verbose=0, warm_start=False)
            MLmodel = MultiOutputRegressor(rfr_1h_wave_model)
            MLmodel.fit(X_train_std, y_train)
        else:
            sc = StandardScaler()
            sc.fit(df[cols2])
            features = sc.transform(df[cols2])
            loaded_model = pickle.load(open("rfr_1h_wave_model.p", 'rb'))
            loaded_model.fit(features, target)
            MLmodel = loaded_model
        # pred = rfr_1h_wave_model.predict(X_test_std)
        # mae = str(mean_absolute_error(y_test, pred))
        # mse = str(mean_squared_error(y_test, pred))
        # logging.debug("Mean absolute error(MAE):{}".format(mae))
        # logging.debug("Mean squared error(MSE):{}".format(mse))
        # using pickle to save the model
        # if residualECCC and residualSMA:
        #     os.remove("ECCC_residual.csv")
        #     os.remove("SMA_residual.csv")
        pickle.dump(MLmodel, open(directoryloc+"rfr_1h_wave_model.p", "wb"))



    def dataPreProcess(self, isSchedulerhit):
        ECCCdf, residualECCC = self.cleanECCC()
        SMAdf, residualSMA = self.cleanSMA()
        if all(temp is None for temp in [ECCCdf, SMAdf]):
            return None, None
        if not ((ECCCdf is None) or (SMAdf is None)):
            self.combineData(ECCCdf, residualECCC, SMAdf, residualSMA, isSchedulerhit)
        return residualECCC, residualSMA


    def train(self, residualECCC, residualSMA):
        if path.exists(directoryloc+"accumulated_data_to_train.csv"):
            accdata = pd.read_csv(directoryloc+"accumulated_data_to_train.csv")
            self.learnAlgorithm(accdata, residualECCC, residualSMA)
            os.remove(directoryloc+"accumulated_data_to_train.csv")
        else:
            combineddf = pd.read_csv(directoryloc+"SMA_ECCC_1H.csv")
            self.learnAlgorithm(combineddf, residualECCC, residualSMA)
        return "Trained"


    def predict(self):
        if path.exists(directoryloc+"SMA_ECCC_1H_residual.csv"):
            residualdf = pd.read_csv(directoryloc+"SMA_ECCC_1H_residual.csv")
            residualdf.set_index("index", inplace=True)
            cols = residualdf.columns.tolist()
            cols2 = cols[:7]
            residualdf.dropna(inplace=True)
            featuredf = residualdf[cols2]
            # residualdf.drop("wave_ht_sig unit(m)", axis=1, inplace=True)
            sc = StandardScaler()
            sc.fit(featuredf)
            features = sc.transform(featuredf)
            loadedmodel = pickle.load(open(directoryloc+"rfr_1h_wave_model.p", 'rb'))
            prediction = loadedmodel.predict(features)
            originalvalue = residualdf[["wave_ht_sig unit(m)",'wind_spd_avg unit(m s-1)','wave_ht_max unit(m)','wave_period_max unit(s)']]
            originalvalue = originalvalue.values
            preddict = {"time":featuredf.index, "sig_wv_ht":prediction[0][0], "mx_wv_ht":prediction[0][2],
                        "max_wv_prd":prediction[0][3], "wnd_spd":prediction[0][1]}
            preddf = pd.DataFrame(preddict)
            preddf.set_index('time',inplace=True)
            if path.exists(predloc):
                predictiondf = pd.read_csv(predloc)
                preddf = pd.concat([predictiondf, preddf])
                preddf.to_csv(predloc)
            return {"predictedvalue":prediction[0], "originalvalue":originalvalue, "message":None}

        return {"message":"system has started to train, hang on for a moment..... "} if scheduler.state == base.STATE_RUNNING else {"message":"Click on Train Now for immediate prediction"}



# scheduler.remove_job('delete_file_sched')
# scheduler.shutdown()





