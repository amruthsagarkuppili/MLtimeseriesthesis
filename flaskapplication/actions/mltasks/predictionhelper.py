
######################################
#                                    #
# @author Amruth Sagar Kuppili       #
# @university Dalhousie University   #
# @ID B00844506                      #
#                                    #
######################################

from os import path
import numpy as np
import pandas as pd
from pickle import dump, load
from tensorflow import keras



from flaskapplication.actions.mltasks.variables import directoryloc, predloc, historypredloc, featurecolumns, scalermodel, timestepsmodel, confidence, anomalytimestepsmodel


class prediction:

    timesteps = timestepsmodel
    anomalytimesteps = anomalytimestepsmodel
    featurecols = featurecolumns
    confidence =  1 - (confidence/100)

    def create_sequences(self, X, time_steps=timesteps):
        Xs = []
        if len(X) > time_steps:
            for i in range(len(X) - time_steps + 1):
                Xs.append(X.iloc[i:(i + time_steps), :self.featurecols].values)
            return np.array(Xs)
        Xs.append(X)
        xarr = np.array(Xs)
        return xarr

    def create_sequences_combined(self, X, time_steps=timesteps):
        Xs =  []
        for i in range(len(X) - time_steps + 1):
            Xs.append(X.iloc[i:(i + time_steps), :2].values)

        return np.array(Xs)

    def create_sequences_anomaly(self, X, y, time_steps=anomalytimesteps):
        Xs = []
        for i in range(len(X) - time_steps + 1):
            Xs.append(X.iloc[i:(i + time_steps), :6].values)
            # ys.append(y.iloc[i + time_steps - 1, :6].values)

        return np.array(Xs)



    featurecolums = featurecolumns

    def predict(self, scheduler, base, ismaincall):
        if ismaincall and path.exists(historypredloc):
            historydf = pd.read_csv(historypredloc, index_col='time')
            historydftail = historydf.tail(1)
            preddict = None
            # historydftail.reset_index(inplace=True)
            for index, row in historydftail.iterrows():
                preddict = {"time": [index], "sig_wv_ht": [row['sig_wv_ht']],
                             "mx_wv_ht": [row['mx_wv_ht']],
                            "max_wv_prd": [row['max_wv_prd']], "wnd_spd": [row['wnd_spd']],
                            "stderrsight":[row['stderrsight']],"stderrmxwvht":[row['stderrmxwvht']],
                            "stderrwvprd":[row['stderrwvprd']],"stderrwndspd":[row['stderrwndspd']],
                            "isanomaly":[row['isanomaly']]}
            preddf = pd.DataFrame(preddict)
            preddf.set_index('time', inplace=True)
            return {"predictedvalue": preddf, "originalvalue": "", "message": None}
        else:
            if path.exists(directoryloc + "SMA_ECCC_1H_residual.csv"):
                residualdf = pd.read_csv(directoryloc + "SMA_ECCC_1H_residual.csv")
                tempcols = residualdf.columns.tolist()
                tempcols = [c for c in tempcols if "Unnamed" not in c]
                residualdf = residualdf[tempcols]
                # residualdf.drop("Unnamed: 0", axis=1,
                #         inplace=True)
                residualdf.set_index("index", inplace=True)
                residualdf.drop("WDIR", axis=1, inplace=True)
                cols = residualdf.columns.tolist()
                cols2 = cols[:self.featurecolums]
                residualdf.dropna(inplace=True)
                featuredforg = residualdf[cols2]
                featuredf = featuredforg.copy()
                featuredf.index = pd.to_datetime(featuredf.index)
                exogvariablesight = featuredf[['VCAR', 'VWH$', 'VCMX']]
                exogvariablemxht = featuredf[['VCAR', 'VWH$', 'VCMX']]
                exogvariablewvprd = featuredf[['VCAR', 'VWH$', 'VCMX', 'VTP$']]
                exogvariablewndspd = featuredf[['VCAR', 'VWH$', 'VCMX', 'WSPD', 'GSPD']]
                # residualdf.drop("wave_ht_sig unit(m)", axis=1, inplace=True)



                # LSTM prediction
                featurescaler = load(open(scalermodel + 'featurescaler.pkl', 'rb'))
                targetscaler = load(open(scalermodel + 'targetscaler.pkl', 'rb'))
                featurelstm = featuredf.copy()
                featurelstm[cols2] = featurescaler.transform(featuredf)
                changedfeatures = self.create_sequences(featurelstm)
                # changedfeatures = changedfeatures.reshape((changedfeatures.shape[0], self.timesteps, featurecolumns))
                loadedmodel = keras.models.load_model(scalermodel + "lstm_ml_model")
                prediction = loadedmodel.predict(changedfeatures)
                lstminversedprediction = targetscaler.inverse_transform(prediction)
                lstminversedprediction = np.round(lstminversedprediction[:, :], decimals=2)

                # Arima prediction
                arimaalgodict = load(open(scalermodel + 'arimamodels.pkl', 'rb'))
                sightarimaalgo = arimaalgodict['wave_ht_sig unit(m)']
                mxwvhtarimaalgo = arimaalgodict['wave_ht_max unit(m)']
                wvprdarimaalgo = arimaalgodict['wave_period_max unit(s)']
                wndspdarimaalgo = arimaalgodict['wind_spd_avg unit(m s-1)']
                startindex = str(featuredf.index.max())

                arimasight = sightarimaalgo.forecast(len(featuredf), exog=exogvariablesight)
                arimamxwvht = mxwvhtarimaalgo.forecast(len(featuredf), exog=exogvariablemxht)
                arimawvprd = wvprdarimaalgo.forecast(len(featuredf), exog=exogvariablewvprd)
                arimawndspd = wndspdarimaalgo.forecast(len(featuredf), exog=exogvariablewndspd)

                indexing = (self.timesteps - 1)
                arimasight = arimasight[indexing:]
                arimamxwvht = arimamxwvht[indexing:]
                arimawvprd = arimawvprd[indexing:]
                arimawndspd = arimawndspd[indexing:]

                arimasightpredwrap = sightarimaalgo.get_forecast(len(featuredf), exog=exogvariablesight)
                arimamxwvhtpredwrap = mxwvhtarimaalgo.get_forecast(len(featuredf), exog=exogvariablemxht)
                arimawvprdpredwrap = wvprdarimaalgo.get_forecast(len(featuredf), exog=exogvariablewvprd)
                arimawndspdpredwrap = wndspdarimaalgo.get_forecast(len(featuredf), exog=exogvariablewndspd)

                confdfsight = arimasightpredwrap.conf_int(alpha=self.confidence)
                confdfmxwvht = arimamxwvhtpredwrap.conf_int(alpha=self.confidence)
                confdfwvprd = arimawvprdpredwrap.conf_int(alpha=self.confidence)
                confdfwndspd = arimawndspdpredwrap.conf_int(alpha=self.confidence)

                confdfsight[confdfsight < 0] = 0
                confdfmxwvht[confdfmxwvht < 0] = 0
                confdfwvprd[confdfwvprd < 0] = 0
                confdfwndspd[confdfwndspd < 0] = 0

                intervalarraysight = np.abs(
                    confdfsight['upper wave_ht_sig unit(m)'].values - confdfsight['lower wave_ht_sig unit(m)'].values)
                intervalarraymxwvht = np.abs(
                    confdfmxwvht['upper wave_ht_max unit(m)'].values - confdfmxwvht['lower wave_ht_max unit(m)'].values)
                intervalarraywvprd = np.abs(
                    confdfwvprd['upper wave_period_max unit(s)'].values - confdfwvprd['lower wave_period_max unit(s)'].values)
                intervalarraywndspd = np.abs(
                    confdfwndspd['upper wind_spd_avg unit(m s-1)'].values - confdfwndspd['lower wind_spd_avg unit(m s-1)'].values)

                stderrsight = intervalarraysight[-1] / 2
                stderrmxwvht = intervalarraymxwvht[-1] / 2
                stderrwvprd = intervalarraywvprd[-1] / 2
                stderrwndspd = intervalarraywndspd[-1] / 2

                # arimasightindex = arimasight.index.max()
                # arimamxwvhtindex = arimamxwvht.index.max()
                # arimawvprdindex = arimawvprd.index.max()
                # arimawndspdindex = arimawndspd.index.max()

                allmodel_sight = pd.DataFrame({'lstm': lstminversedprediction[:, 0], 'arima': arimasight,
                                              'index':arimasight.index})
                allmodel_mxwvht = pd.DataFrame({'lstm': lstminversedprediction[:, 2], 'arima': arimamxwvht,
                                              'index':arimamxwvht.index})
                allmodel_wndspd = pd.DataFrame({'lstm': lstminversedprediction[:, 1], 'arima': arimawndspd,
                                              'index':arimawndspd.index})
                allmodel_wvprd = pd.DataFrame({'lstm': lstminversedprediction[:, 3], 'arima': arimawvprd,
                                               'index': arimawvprd.index})

                allmodel_sight.set_index('index',inplace=True)
                allmodel_mxwvht.set_index('index', inplace=True)
                allmodel_wvprd.set_index('index', inplace=True)
                allmodel_wndspd.set_index('index', inplace=True)

                X_train_combined_sight= self.create_sequences_combined(allmodel_sight)
                X_train_combined_mxwvht = self.create_sequences_combined(allmodel_mxwvht)
                X_train_combined_wvprd = self.create_sequences_combined(allmodel_wvprd)
                X_train_combined_wndspd = self.create_sequences_combined(allmodel_wndspd)

                sightlstm = keras.models.load_model(scalermodel + "combined_lstm_sight")
                mxwvhtlstm = keras.models.load_model(scalermodel + "combined_lstm_mxwvht")
                wvprdlstm = keras.models.load_model(scalermodel + "combined_lstm_wvprd")
                wndspdlstm = keras.models.load_model(scalermodel + "combined_lstm_wndspd")

                sightprediction = sightlstm.predict(X_train_combined_sight)
                mxwvhtprediction = mxwvhtlstm.predict(X_train_combined_mxwvht)
                wvprdprediction = wvprdlstm.predict(X_train_combined_wvprd)
                wndspdprediction = wndspdlstm.predict(X_train_combined_wndspd)

                # anomaly detection code
                anomalyscaler = load(open(scalermodel + 'anomalyscaler.pkl', 'rb'))
                reconerrors = load(open(scalermodel + 'reconstructionerrors.pkl', 'rb'))
                anomaly_model = keras.models.load_model(scalermodel + "lstm_ml_model_anomaly")
                featurecolsanomaly = cols[:self.featurecols]
                lastres = residualdf.tail(1)
                lastres[featurecolsanomaly] = anomalyscaler.transform(lastres[featurecolsanomaly])
                anomalydata = self.create_sequences_anomaly(lastres,lastres)
                anomalyprediction = anomaly_model.predict(anomalydata)
                loss = np.abs(anomalydata - anomalyprediction)
                vcarloss = loss[0][:, 0][0]
                vwhloss = loss[0][:, 1][0]
                vcmxloss = loss[0][:, 2][0]
                vtploss = loss[0][:, 3][0]
                wspdloss = loss[0][:, 4][0]
                gspdloss = loss[0][:, 5][0]
                anomalycount = 0
                isanomaly = False
                for x in range(len(anomalyprediction)):
                    if vcarloss>=reconerrors['vcarthreshold']:
                        anomalycount += 1
                    if vwhloss>=reconerrors['vwh$threshold']:
                        anomalycount += 1
                    if vcmxloss>=reconerrors['vcmxthreshold']:
                        anomalycount += 1
                    if vtploss>=reconerrors['vtp$threshold']:
                        anomalycount += 1
                    if wspdloss>=reconerrors['wspdthreshold']:
                        anomalycount += 1
                    if gspdloss>=reconerrors['gspdthreshold']:
                        anomalycount += 1
                if anomalycount > 1:
                    isanomaly = True



                originalvalue = residualdf[
                    ["wave_ht_sig unit(m)", 'wind_spd_avg unit(m s-1)', 'wave_ht_max unit(m)', 'wave_period_max unit(s)']]
                originalvalue = originalvalue.values
                preddict = {"time": str(featuredf.index.max()), "sig_wv_ht": (sightprediction[0])[0], "mx_wv_ht": (mxwvhtprediction[0])[0],
                            "max_wv_prd": (wvprdprediction[0])[0], "wnd_spd": (wndspdprediction[0])[0], "stderrsight": stderrsight,
                            "stderrmxwvht": stderrmxwvht, "stderrwvprd": stderrwvprd, "stderrwndspd": stderrwndspd, "isanomaly": isanomaly
                            }
                preddf = pd.DataFrame(preddict, index=[0])
                preddf = np.round(preddf, decimals=2)
                preddf.set_index('time', inplace=True)
                if path.exists(predloc):
                    predictiondf = pd.read_csv(predloc)
                    finaldf = pd.concat([predictiondf, preddf])
                    finaldf.to_csv(predloc)
                else:
                    preddf.to_csv(predloc, header=True)
                return {"predictedvalue": preddf, "originalvalue": originalvalue, "message": None}

        return {
            "message": "system has started to train, hang on for a moment..... "} if scheduler.state == base.STATE_RUNNING else {
            "message": "Click on Train Now for immediate prediction"}

