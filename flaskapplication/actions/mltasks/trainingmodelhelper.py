import pickle
import numpy as np
import math
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from flaskapplication.actions.mltasks.variables import directoryloc, scalermodel, timestepsmodel, featurecolumns, thresholds, anomalytimestepsmodel

import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, RNN
from tensorflow import keras
from pickle import dump, load
from statsmodels.tsa.arima.model import ARIMA






class training:

    timesteps = timestepsmodel
    anomalytimesteps = anomalytimestepsmodel
    featurecols = featurecolumns

    def create_sequences(self, X, y, time_steps=timesteps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps + 1):
            Xs.append(X.iloc[i:(i + time_steps), :self.featurecols].values)
            ys.append(y.iloc[(i + time_steps-1), self.featurecols:].values)

        return np.array(Xs), np.array(ys)

    def create_sequences_combined(self, X, y, time_steps=timesteps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps + 1):
            Xs.append(X.iloc[i:(i + time_steps), :2].values)
            ys.append(y.iloc[i + time_steps - 1, 2:].values)

        return np.array(Xs), np.array(ys)

    def create_sequences_anomaly(self, X, y, time_steps=anomalytimesteps):
        Xs = []
        for i in range(len(X) - time_steps + 1):
            Xs.append(X.iloc[i:(i + time_steps), :6].values)
            # ys.append(y.iloc[i + time_steps - 1, :6].values)

        return np.array(Xs)


    def createensemble(self, data, features, targets, isresidualeccc, isresidualsma, time_steps=timesteps):
        exogvariablesight = data[['VCAR', 'VWH$', 'VCMX']]
        exogvariablemxht = data[['VCAR', 'VWH$', 'VCMX']]
        exogvariablewvprd = data[['VCAR', 'VWH$', 'VCMX', 'VTP$']]
        exogvariablewndspd = data[['VCAR', 'VWH$', 'VCMX', 'WSPD', 'GSPD']]


        featurescaler = load(open(scalermodel + 'featurescaler.pkl', 'rb'))
        targetscaler = load(open(scalermodel + 'targetscaler.pkl', 'rb'))
        lstmx = data.copy()
        lstmx[features] = featurescaler.transform(data[features])
        lstmfeatures, lstmtargets = self.create_sequences(lstmx, lstmx)
        featurescount = lstmfeatures.shape[2]
        lstmfeatures = lstmfeatures.reshape((lstmfeatures.shape[0], self.timesteps, featurescount))
        lstmalgo = keras.models.load_model(scalermodel + "lstm_ml_model")
        lstmpred = lstmalgo.predict(lstmfeatures)
        lstminversedprediction = targetscaler.inverse_transform(lstmpred)
        lstminversedprediction = np.round(lstminversedprediction[:, :], decimals=2)

        arimaalgodict = load(open(scalermodel + 'arimamodels.pkl', 'rb'))
        sightarimaalgo = arimaalgodict['wave_ht_sig unit(m)']
        mxwvhtarimaalgo = arimaalgodict['wave_ht_max unit(m)']
        wvprdarimaalgo = arimaalgodict['wave_period_max unit(s)']
        wndspdarimaalgo = arimaalgodict['wind_spd_avg unit(m s-1)']

        # arimasight, xc, xsd = sightarimaalgo.forecast(1, exog=exogvariablesight[0, :])
        arimasight= sightarimaalgo.forecast(len(data), exog=exogvariablesight)
        arimamxwvht = mxwvhtarimaalgo.forecast(len(data), exog=exogvariablemxht)
        arimawvprd = wvprdarimaalgo.forecast(len(data), exog=exogvariablewvprd)
        arimawndspd = wndspdarimaalgo.forecast(len(data), exog=exogvariablewndspd)


        arimasight = arimasight.reset_index(drop=True)
        arimamxwvht = arimamxwvht.reset_index(drop=True)
        arimawvprd = arimawvprd.reset_index(drop=True)
        arimawndspd = arimawndspd.reset_index(drop=True)

        indexing = (time_steps-1)
        arimasight = arimasight[indexing:]
        arimamxwvht = arimamxwvht[indexing:]
        arimawvprd = arimawvprd[indexing:]
        arimawndspd = arimawndspd[indexing:]



        allmodel_sight = pd.DataFrame({'lstm': lstminversedprediction[:, 0], 'arima': arimasight.values, 'actual': data['wave_ht_sig unit(m)']},
                                             index=data[indexing:].index)
        allmodel_mxwvht = pd.DataFrame({'lstm': lstminversedprediction[:, 2], 'arima': arimamxwvht.values, 'actual': data['wave_ht_max unit(m)']},
                                      index=data[indexing:].index)
        allmodel_wvprd = pd.DataFrame({'lstm': lstminversedprediction[:, 3], 'arima': arimawvprd.values, 'actual': data['wave_period_max unit(s)']},
                                      index=data[indexing:].index)
        allmodel_wndspd = pd.DataFrame({'lstm': lstminversedprediction[:, 1], 'arima': arimawndspd.values, 'actual': data['wind_spd_avg unit(m s-1)']},
                                      index=data[indexing:].index)

        X_train_combined_sight, y_train_combined_sight = self.create_sequences_combined(allmodel_sight, allmodel_sight)
        X_train_combined_mxwvht, y_train_combined_mxwvht = self.create_sequences_combined(allmodel_mxwvht, allmodel_mxwvht)
        X_train_combined_wvprd, y_train_combined_wvprd = self.create_sequences_combined(allmodel_wvprd, allmodel_wvprd)
        X_train_combined_wndspd, y_train_combined_wndspd = self.create_sequences_combined(allmodel_wndspd, allmodel_wndspd)

        if not (isresidualeccc and isresidualsma):


            activation_func = 'tanh'
            weight_initializer = 'lecun_normal'
            recurrent_init = 'lecun_normal'
            recurrent_activation_func = 'hard_sigmoid'


            sightlstm = Sequential()
            sightlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func,
                               return_sequences=True, kernel_initializer=weight_initializer,
                               recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            sightlstm.add(LSTM(17, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            sightlstm.add(RepeatVector(self.timesteps))
            sightlstm.add(LSTM(17, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            sightlstm.add(LSTM(25, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            sightlstm.add(Dense(1))
            sightlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])

            # sightlstm = Sequential()
            # sightlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func, return_sequences=False,
            #                    kernel_initializer=weight_initializer, recurrent_activation=recurrent_act_func,
            #                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)))
            # # sightlstm.add(Dropout(rate=0.4))
            # sightlstm.add(Dense(1))
            # sightlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])

            mxwvhtlstm = Sequential()
            mxwvhtlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func,
                               return_sequences=True, kernel_initializer=weight_initializer,
                               recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            mxwvhtlstm.add(
                LSTM(17, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            mxwvhtlstm.add(RepeatVector(self.timesteps))
            mxwvhtlstm.add(
                LSTM(17, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            mxwvhtlstm.add(
                LSTM(25, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            mxwvhtlstm.add(Dense(1))
            mxwvhtlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])


            # mxwvhtlstm = Sequential()
            # mxwvhtlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func, return_sequences=False,
            #                    kernel_initializer=weight_initializer, recurrent_activation=recurrent_act_func,
            #                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)))
            # mxwvhtlstm.add(Dropout(rate=0.4))
            # mxwvhtlstm.add(Dense(1))
            # mxwvhtlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])

            wvprdlstm = Sequential()
            wvprdlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func,
                               return_sequences=True, kernel_initializer=weight_initializer,
                               recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wvprdlstm.add(
                LSTM(17, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wvprdlstm.add(RepeatVector(self.timesteps))
            wvprdlstm.add(
                LSTM(17, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wvprdlstm.add(
                LSTM(25, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wvprdlstm.add(Dense(1))
            wvprdlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])


            # wvprdlstm = Sequential()
            # wvprdlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func, return_sequences=False,
            #                    kernel_initializer=weight_initializer, recurrent_activation=recurrent_act_func,
            #                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)))
            # wvprdlstm.add(Dropout(rate=0.4))
            # wvprdlstm.add(Dense(1))
            # wvprdlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])

            wndspdlstm = Sequential()
            wndspdlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func,
                               return_sequences=True, kernel_initializer=weight_initializer,
                               recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wndspdlstm.add(
                LSTM(17, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wndspdlstm.add(RepeatVector(self.timesteps))
            wndspdlstm.add(
                LSTM(17, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wndspdlstm.add(
                LSTM(25, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init, recurrent_activation=recurrent_activation_func))
            wndspdlstm.add(Dense(1))
            wndspdlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])



            # wndspdlstm = Sequential()
            # wndspdlstm.add(LSTM(25, input_shape=(self.timesteps, 2), activation=activation_func, return_sequences=False,
            #                    kernel_initializer=weight_initializer, recurrent_activation=recurrent_act_func,
            #                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)))
            # wndspdlstm.add(Dropout(rate=0.4))
            # wndspdlstm.add(Dense(1))
            # wndspdlstm.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])

        else :
            sightlstm = keras.models.load_model(scalermodel + "combined_lstm_sight")
            mxwvhtlstm = keras.models.load_model(scalermodel + "combined_lstm_mxwvht")
            wvprdlstm = keras.models.load_model(scalermodel + "combined_lstm_wvprd")
            wndspdlstm = keras.models.load_model(scalermodel + "combined_lstm_wndspd")

        historycombsight = sightlstm.fit(X_train_combined_sight, y_train_combined_sight, epochs=50,
                                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                    shuffle=False)
        historycombmxwvht = mxwvhtlstm.fit(X_train_combined_mxwvht, y_train_combined_mxwvht, epochs=50,
                                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                    shuffle=False)
        historycombwvprd = wvprdlstm.fit(X_train_combined_wvprd, y_train_combined_wvprd, epochs=50,
                                      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                      shuffle=False)
        historycombwndspd = wndspdlstm.fit(X_train_combined_wndspd, y_train_combined_wndspd, epochs=50,
                                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                        shuffle=False)

        sightlstm.save(scalermodel + "combined_lstm_sight")
        mxwvhtlstm.save(scalermodel + "combined_lstm_mxwvht")
        wvprdlstm.save(scalermodel + "combined_lstm_wvprd")
        wndspdlstm.save(scalermodel + "combined_lstm_wndspd")





    # Training the algorithm
    def learnAlgorithm(self, df, residualECCC, residualSMA):
        np.random.seed(1)
        tf.random.set_seed(1)

        df.drop("Unnamed: 0", axis=1,
                   inplace=True)

        df.set_index("index", inplace=True)
        subdata = df[['VCAR', 'VWH$', 'VCMX', 'VTP$', 'WSPD', 'GSPD', 'last_VCAR', 'last_VWH$', 'wave_ht_sig unit(m)',
                      'wind_spd_avg unit(m s-1)', 'wave_ht_max unit(m)', 'wave_period_max unit(s)',
                      'wave_dir_avg unit(degree)']]
        usedf = subdata.copy()
        usedf.drop(['last_VCAR', 'last_VWH$', 'wave_dir_avg unit(degree)'], inplace=True, axis=1)
        totalcols = usedf.columns.tolist()
        featurecols = totalcols[:self.featurecols]
        targetcols = totalcols[self.featurecols:]
        testorg = usedf.copy()
        trainorg = usedf.copy()
        usedfforanomaly = usedf.copy()


        if not (residualECCC and residualSMA):
            # asdf = self.anomalydetectiontrain(usedfforanomaly.copy(), residualECCC, residualSMA)
            # if asdf:
            #      return None
            trainendindex = math.ceil(usedf.shape[0] * 0.95)
            teststartindex = trainendindex + 1
            trainorg, testorg = usedf.iloc[:trainendindex, :], usedf.iloc[teststartindex:, :]
            train = trainorg.copy()
            test = testorg.copy()
            featurescaler = MinMaxScaler(feature_range=(-1,1))
            targetscaler = MinMaxScaler(feature_range=(-1,1))
            featurescaler = featurescaler.fit(train[featurecols])
            targetscaler = targetscaler.fit(train[targetcols])
            train[featurecols] = featurescaler.transform(train[featurecols])
            test[featurecols] = featurescaler.transform(test[featurecols])
            train[targetcols] = targetscaler.transform(train[targetcols])
            test[targetcols] = targetscaler.transform(test[targetcols])

            X_train, y_train = self.create_sequences(train[totalcols], train[totalcols])
            X_test, y_test = self.create_sequences(test[totalcols], test[totalcols])

            featurescount = X_train.shape[2]
            X_train = X_train.reshape((X_train.shape[0], self.timesteps, featurescount))
            X_test = X_test.reshape((X_test.shape[0], self.timesteps, featurescount))
            predictvarcount = y_train.shape[1]


            # LSTM model development

            activation_func = 'tanh'
            weight_initializer = 'lecun_normal'
            recurrent_init = 'lecun_normal'
            recurrent_activation_func = 'hard_sigmoid'


            lstmmodel = Sequential()
            lstmmodel.add(LSTM(128, input_shape=(self.timesteps, featurescount), activation=activation_func,
                           return_sequences=True, kernel_initializer=weight_initializer,
                           recurrent_initializer=recurrent_init,recurrent_activation=recurrent_activation_func))
            lstmmodel.add(LSTM(85, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init,recurrent_activation=recurrent_activation_func))
            lstmmodel.add(RepeatVector(self.timesteps))
            lstmmodel.add(LSTM(85, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
                           recurrent_initializer=recurrent_init,recurrent_activation=recurrent_activation_func))
            lstmmodel.add(LSTM(128, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                     recurrent_initializer=recurrent_init,recurrent_activation=recurrent_activation_func))
            lstmmodel.add(Dense(predictvarcount))
            lstmmodel.compile(optimizer='adamax', loss='mse')



            # activation_func = 'tanh'
            # weight_initializer = 'lecun_normal'
            # recurrent_init = 'lecun_normal'
            # recurrent_act_func = 'tanh'
            #
            # lstmmodel = Sequential()
            # lstmmodel.add(LSTM(175, input_shape=(self.timesteps, featurescount), activation=activation_func, return_sequences=True,
            #                kernel_initializer=weight_initializer, recurrent_activation=recurrent_act_func, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)))
            # lstmmodel.add(Dropout(rate=0.2))
            # lstmmodel.add(LSTM(175, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
            #                recurrent_activation=recurrent_act_func, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)))
            # lstmmodel.add(Dropout(rate=0.2))
            # lstmmodel.add(LSTM(175, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
            #                recurrent_activation=recurrent_act_func, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)))
            # lstmmodel.add(Dropout(rate=0.2))
            # lstmmodel.add(Dense(predictvarcount,activation=activation_func))
            # lstmmodel.compile(optimizer='adamax', loss='mse')

            history = lstmmodel.fit(X_train, y_train, epochs=30, validation_split=0.1,
                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                shuffle=False)

            MLmodel = lstmmodel
            # save the scaler object
            dump(featurescaler, open(scalermodel + 'featurescaler.pkl', 'wb'))
            dump(targetscaler, open(scalermodel + 'targetscaler.pkl', 'wb'))
            MLmodel.save(scalermodel + "lstm_ml_model")
            self.trainarima(trainorg.copy(), residualECCC, residualSMA)
            self.createensemble(testorg.copy(), featurecols, targetcols, residualECCC, residualSMA)
            self.anomalydetectiontrain(usedfforanomaly.copy(), residualECCC, residualSMA)




        else:
            incomingdata = usedf.copy()
            self.createensemble(testorg.copy(), featurecols, targetcols, residualECCC, residualSMA)

            # load the scaler
            featurescaler = load(open(scalermodel+'featurescaler.pkl', 'rb'))
            targetscaler = load(open(scalermodel + 'targetscaler.pkl', 'rb'))

            incomingdata[featurecols] = featurescaler.transform(incomingdata[featurecols])
            incomingdata[targetcols] = targetscaler.transform(incomingdata[targetcols])
            loaded_model = keras.models.load_model(scalermodel + "lstm_ml_model")
            X_income, y_income = self.create_sequences(incomingdata[totalcols], incomingdata[totalcols])
            featurescount = X_income.shape[2]
            X_income = incomingdata.reshape((X_income.shape[0], self.timesteps, featurescount))
            loaded_model.fit(X_income, y_income)

            os.remove(directoryloc+"ECCC_residual.csv")
            os.remove(directoryloc+"SMA_residual.csv")

            MLmodel = loaded_model
            self.trainarima(trainorg.copy(), residualECCC, residualSMA)

            MLmodel.save(scalermodel + "lstm_ml_model")




    def trainarima(self, usedf, isresidualeccc, isresidualsma):
        sightbeforenormal = usedf['wave_ht_sig unit(m)']
        exogvariablesight = usedf[['VCAR', 'VWH$', 'VCMX']]
        mxhtbeforenormal = usedf['wave_ht_max unit(m)']
        exogvariablemxht = usedf[['VCAR', 'VWH$', 'VCMX']]
        wvprdbeforenormal = usedf['wave_period_max unit(s)']
        exogvariablewvprd = usedf[['VCAR', 'VWH$', 'VCMX', 'VTP$']]
        wndspdbeforenormal = usedf['wind_spd_avg unit(m s-1)']
        exogvariablewndspd = usedf[['VCAR', 'VWH$', 'VCMX', 'WSPD', 'GSPD']]
        exogdict = {'sight' : exogvariablesight, 'mxwvht' : exogvariablemxht,
                    'wvprd' : exogvariablewvprd, 'wndspd' : exogvariablewndspd
                    }
        if not (isresidualeccc and isresidualsma):
            fittingdatalen = math.ceil(sightbeforenormal.shape[0] * 0.9)
            testingdatalen = fittingdatalen + 1
            armodelsight = ARIMA(sightbeforenormal, order=(3, 1, 4), exog=exogvariablesight)
            modelfitsight = armodelsight.fit()
            armodelmxwvht = ARIMA(mxhtbeforenormal, order=(2, 1, 1), exog=exogvariablemxht)
            modelfitmxwvht = armodelmxwvht.fit()
            armodelwvprd = ARIMA(wvprdbeforenormal, order=(4, 0, 1), exog=exogvariablewvprd)
            modelfitwvprd = armodelwvprd.fit()
            armodelwndspd = ARIMA(wndspdbeforenormal, order=(1, 0, 1), exog=exogvariablewndspd)
            modelfitwndspd = armodelwndspd.fit()


        else:
            arimamodels = load(open(scalermodel + 'arimamodels.pkl', 'rb'))
            sightarima = arimamodels['wave_ht_sig unit(m)']
            mxwvhtarima = arimamodels['wave_ht_max unit(m)']
            wvprdarima = arimamodels['wave_period_max unit(s)']
            wndspdarima = arimamodels['wind_spd_avg unit(m s-1)']

            modelfitsight = sightarima.append(endog=sightbeforenormal, exog=exogvariablesight, refit=True)
            modelfitmxwvht = mxwvhtarima.append(endog=mxhtbeforenormal, exog=exogvariablemxht, refit=True)
            modelfitwvprd = wvprdarima.append(endog=wvprdbeforenormal, exog=exogvariablewvprd, refit=True)
            modelfitwndspd = wndspdarima.append(endog=wndspdbeforenormal, exog=exogvariablewndspd, refit=True)


        arimamodeldict = {
            "wave_ht_sig unit(m)": modelfitsight, "wave_ht_max unit(m)": modelfitmxwvht,
            "wave_period_max unit(s)": modelfitwvprd, "wind_spd_avg unit(m s-1)": modelfitwndspd
        }

        dump(arimamodeldict, open(scalermodel + 'arimamodels.pkl', 'wb'))


    def anomalydetectiontrain(self, anomalydf, isresidualecc, isresidualsma):
        wvhtthreshold = thresholds['sig_wv_ht_threshold_moderate']
        wvmxhtthreshold = thresholds['mx_wv_ht_threshold_moderate']
        wvprdmaxthreshold = thresholds['wv_prd_threshold_moderate']
        wndspdthreshold = thresholds['wnd_spd_ht_threshold_moderate']

        if not (isresidualecc and isresidualsma):
            cols = anomalydf.columns.tolist()
            featurecolsanomaly = cols[:self.featurecols]
            anomalydf['wave_ht_sig unit(m)'] = np.where(anomalydf['wave_ht_sig unit(m)'] >= wvhtthreshold, 1, 0)
            anomalydf['wind_spd_avg unit(m s-1)'] = np.where(anomalydf['wind_spd_avg unit(m s-1)'] >= wndspdthreshold,
                                                             1, 0)
            anomalydf['wave_ht_max unit(m)'] = np.where(anomalydf['wave_ht_max unit(m)'] >= wvmxhtthreshold, 1, 0)
            anomalydf['wave_period_max unit(s)'] = np.where(anomalydf['wave_period_max unit(s)'] >= wvprdmaxthreshold,
                                                            1, 0)
            anomalydf['isanomaly'] = [1 if x > 1 else 0 for x in np.sum(
                anomalydf.drop(['VCAR', 'VWH$', 'VCMX', 'VTP$', 'WSPD', 'GSPD'], 1).values == 1, 1)]
            trainendindex = math.ceil(anomalydf.shape[0] * 0.95)
            teststartindex = trainendindex + 1
            trainorg, testorg = anomalydf.iloc[:trainendindex, :], anomalydf.iloc[teststartindex:, :]
            train = trainorg.copy()
            test = testorg.copy()
            train = train[(train["isanomaly"] == 0)]
            anomalyscaler = MinMaxScaler(feature_range=(-1, 1))
            anomalyscaler = anomalyscaler.fit(train[featurecolsanomaly])
            train[featurecolsanomaly] = anomalyscaler.transform(train[featurecolsanomaly])
            X_train = self.create_sequences_anomaly(train, train)
            featurescount = X_train.shape[2]
            X_train = X_train.reshape((X_train.shape[0], self.anomalytimesteps, featurescount))

            activation_func = 'tanh'
            weight_initializer = 'lecun_normal'
            recurrent_init = 'lecun_normal'
            recurrent_activation_func = 'hard_sigmoid'

            anomalyts = (self.anomalytimesteps, featurescount)

            anomalymodel = Sequential()
            anomalymodel.add(LSTM(84, input_shape=anomalyts, activation=activation_func,
                           return_sequences=True, kernel_initializer=weight_initializer,
                           recurrent_initializer=recurrent_init,
                           recurrent_activation=recurrent_activation_func))
            anomalymodel.add(LSTM(56, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer,
                           recurrent_initializer=recurrent_init,
                           recurrent_activation=recurrent_activation_func))
            # model.add(LSTM(114, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer, recurrent_initializer=recurrent_init,
            #                recurrent_activation=recurrent_activation_func))
            # model.add(LSTM(76, activation=activation_func, return_sequences=False, kernel_initializer=weight_initializer, recurrent_initializer=recurrent_init,
            #                recurrent_activation=recurrent_activation_func))
            anomalymodel.add(RepeatVector(self.anomalytimesteps))
            # model.add(LSTM(76, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer, recurrent_initializer=recurrent_init,
            #                recurrent_activation=recurrent_activation_func))
            # model.add(LSTM(114, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer, recurrent_initializer=recurrent_init,
            #                recurrent_activation=recurrent_activation_func))
            anomalymodel.add(LSTM(56, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
                           recurrent_initializer=recurrent_init,
                           recurrent_activation=recurrent_activation_func))
            anomalymodel.add(LSTM(84, activation=activation_func, return_sequences=True, kernel_initializer=weight_initializer,
                           recurrent_initializer=recurrent_init,
                           recurrent_activation=recurrent_activation_func))
            anomalymodel.add(TimeDistributed(Dense(featurescount)))
            anomalymodel.compile(optimizer='adamax', loss='mse')

            anomalyhistory = anomalymodel.fit(X_train, X_train, epochs=20, validation_split=0.2,
                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                                shuffle=False)

            x_train_pred = anomalymodel.predict(X_train, verbose=0)
            train_mae_loss = np.abs(x_train_pred - X_train)

            # Maximum of every column
            reconstructionthreshold = (np.max(train_mae_loss, axis=0))
            print(f'Reconstruction error threshold: {reconstructionthreshold}')
            reconstructdict = {
                "vcarthreshold": reconstructionthreshold[0][0],
                "vwh$threshold": reconstructionthreshold[0][1],
                "vcmxthreshold": reconstructionthreshold[0][2],
                "vtp$threshold": reconstructionthreshold[0][3],
                "wspdthreshold": reconstructionthreshold[0][4],
                "gspdthreshold": reconstructionthreshold[0][5]
            }
            dump(anomalyscaler, open(scalermodel + 'anomalyscaler.pkl', 'wb'))
            dump(reconstructdict, open(scalermodel + 'reconstructionerrors.pkl', 'wb'))
            anomalymodel.save(scalermodel + "lstm_ml_model_anomaly")
            # return False
        # else:
        #     anomalyscaler = load(open(scalermodel + 'anomalyscaler.pkl', 'rb'))
        #     loaded_model = keras.models.load_model(scalermodel + "lstm_ml_model_anomaly")
        #     anomalydf = anomalyscaler.transform(anomalydf)
        #     train = self.create_sequences_anomaly(anomalydf,anomalydf)



