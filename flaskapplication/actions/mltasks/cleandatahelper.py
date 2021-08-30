
######################################
#                                    #
# @author Amruth Sagar Kuppili       #
# @university Dalhousie University   #
# @ID B00844506                      #
#                                    #
######################################


from flaskapplication.actions.mltasks.variables import directoryloc,predloc,ECCClink,SMAlink, timestepsmodel
from flaskapplication.actions.mltasks.pulldatahelper import pullinformation


import pandas as pd
from os import path
import numpy as np
import datetime as dt

pullinfo = pullinformation()

class cleandata:


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

    # Handle intermediate missing years by copying the previous years data
    def handlemissingyears(self, df):

        # 2015 remaining missing  data
        halfyeardata = df.loc['2014-07-31 7:00':'2014-12-31 23:00']
        halfyeardata.reset_index(inplace=True)
        halfyeardata['index'] = halfyeardata['index'].mask(halfyeardata['index'].dt.year == 2014,
                                                           halfyeardata['index'] + pd.offsets.DateOffset(year=2015))
        halfyeardata.set_index('index', inplace=True)


        # 2016 and 2017 data generation
        presentyeardata = df[df.index.year == 2014]
        missingyeardataone = presentyeardata.copy()
        missingyeardatatwo = presentyeardata.copy()
        missingyeardataone.reset_index(inplace=True)
        missingyeardataone['index'] = missingyeardataone['index'].mask(missingyeardataone['index'].dt.year == 2014,
                                                                       missingyeardataone[
                                                                           'index'] + pd.offsets.DateOffset(year=2016))
        missingyeardataone.set_index('index', inplace=True)

        missingyeardatatwo.reset_index(inplace=True)
        missingyeardatatwo['index'] = missingyeardatatwo['index'].mask(missingyeardatatwo['index'].dt.year == 2014,
                                                                       missingyeardatatwo[
                                                                           'index'] + pd.offsets.DateOffset(year=2017))
        missingyeardatatwo.set_index('index', inplace=True)


        # 2018 january to october generation
        twohalfyeardata = df.loc['2014-01-01 0:00':'2014-10-29 18:00']
        twohalfyeardata.reset_index(inplace=True)
        twohalfyeardata['index'] = twohalfyeardata['index'].mask(twohalfyeardata['index'].dt.year == 2014,
                                                                 twohalfyeardata['index'] + pd.offsets.DateOffset(
                                                                     year=2018))
        twohalfyeardata.set_index('index', inplace=True)

        # Final concatenation
        firsthalf = df[df.index.year < 2017]
        secondhalf = df[df.index.year > 2017]
        df = pd.concat(
            [firsthalf, halfyeardata, missingyeardataone, missingyeardatatwo, twohalfyeardata, secondhalf], axis=0)

        return df



    def checknoise(self, df):
        vcarmean = 1.33
        vwmean = 1.34
        vcmxmean = 2.23
        vtpmean = 8.3
        wspdmean = 4.04
        gspdmean = 4.97
        df.loc[((df['VCAR']<=0)|(df['VCAR']>=15)), 'VCAR'] = vcarmean
        df.loc[((df['VWH$'] <= 0) | (df['VWH$'] >= 15)), 'VWH$'] = vwmean
        df.loc[((df['VCMX'] <= 0) | (df['VCMX'] >= 40)), 'VCMX'] = vcmxmean
        df.loc[((df['VTP$'] <= 0) | (df['VTP$'] >= 20)), 'VTP$'] = vtpmean
        df.loc[((df['WSPD'] <= 0) | (df['WSPD'] >= 35)), 'WSPD'] = wspdmean
        df.loc[((df['GSPD'] <= 0) | (df['GSPD'] >= 45)), 'GSPD'] = gspdmean

        return df

    # ECCC Data pulling and cleaning
    def cleanECCC(self):
        ECCCdf, residualECCC = pullinfo.pullData(ECCClink, "ECCC")
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
                ECCCdf.to_csv(directoryloc + "ECCC_residual.csv", index=None, header=True)
            else:
                ECCCdf.to_csv(directoryloc + "c44258_cleaned.csv", index=None, header=True)  # save to file
            return ECCCdf, residualECCC
        return None, False


    # SMA Data pulling and cleaning
    def cleanSMA(self):
        SMAdf, residualSMA = pullinfo.pullData(SMAlink, "SMA")
        if not (SMAdf is None):
            temp_df = SMAdf  # make a copy of file
            SMAdf = self.mergeRow(SMAdf,
                                  residualSMA)  # Assign df to the returned dataframe from the mergeRow function
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
                SMAdf.to_csv(directoryloc + "SMA_residual.csv", index=None, header=True)
            else:
                SMAdf.to_csv(directoryloc + "SMA_buoy_cleaned.csv", index=None,
                             header=True)  # saving dataset to file
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
        if not (residualECCC or residualSMA):
            ecc.drop(ecc.index[0:120433], inplace=True, axis=0)
        df = pd.DataFrame()  # creating an empty dataframe to hold the combined ECCC & SMA data values
        ecc.set_index('DATE & TIME', inplace=True)

        for i in ecc.index:
            if i in SMA.index:
                NOA = pd.DataFrame(ecc.loc[i, ["VCAR", "VWH$", "VCMX", "VTP$", "WDIR", "WSPD", "GSPD"]])
                Smart = pd.DataFrame(SMA.loc[i, ["wave_ht_sig unit(m)","wind_spd_avg unit(m s-1)","wave_ht_max unit(m)","wave_dir_avg unit(degree)","wave_period_max unit(s)"]])
                NOA, Smart = NOA.T, Smart.T
                df1 = pd.concat([NOA, Smart], axis=1)
                df = df.append(df1)
        df.reset_index(inplace=True)  # reseting index to enable dropping empty rows in the dataset

        if not (residualECCC or residualSMA):
            df.drop(df.index[15135:43613], inplace=True, axis=0)
        df.set_index('index', inplace=True)
        eccc.set_index('DATE & TIME', inplace=True)
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
        # df.set_index("index", inplace=True)
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        df['VCAR'] = pd.to_numeric(df['VCAR'])
        df['VCMX'] = pd.to_numeric(df['VCMX'])
        df['VWH$'] = pd.to_numeric(df['VWH$'])
        df['WSPD'] = pd.to_numeric(df['WSPD'])
        df['GSPD'] = pd.to_numeric(df['GSPD'])
        df['WDIR'] = pd.to_numeric(df['WDIR'])
        df['last_VWH$'] = pd.to_numeric(df['last_VWH$'])
        df['wave_ht_sig unit(m)'] = pd.to_numeric(df['wave_ht_sig unit(m)'])
        df['wind_spd_avg unit(m s-1)'] = pd.to_numeric(df['wind_spd_avg unit(m s-1)'])

        df = self.checknoise(df)
        df.replace(to_replace=np.nan, value=0, inplace=True)
        if not (residualECCC and residualSMA):
            df.replace(to_replace=0, value=np.nan, inplace=True)
        df = df.interpolate(method='from_derivatives', inplace=False, axis=0, limit_direction='forward')
        df.dropna(inplace=True)
        if not (residualECCC and residualSMA):
            df = self.handlemissingyears(df)
        df.reset_index(inplace=True)
        if path.exists(directoryloc + "SMA_ECCC_1H.csv"):
            dfretr = pd.read_csv(directoryloc + "SMA_ECCC_1H.csv")
            dfretr.drop("Unnamed: 0", axis=1,
                              inplace=True)
            if len(dfretr) >= timestepsmodel:
                dfretr = dfretr.tail(2)
                dfretr = pd.concat([dfretr, df], axis=0)
                dfretr.to_csv(directoryloc + "SMA_ECCC_1H.csv", header=True)
        else:
            df.to_csv(directoryloc+"SMA_ECCC_1H.csv", header=True)
        if (residualECCC and residualSMA):
            if path.exists(directoryloc+"SMA_ECCC_1H_residual.csv"):
                residualretr = pd.read_csv(directoryloc+"SMA_ECCC_1H_residual.csv")
                residualretr.drop("Unnamed: 0", axis=1,
                        inplace=True)
                # File should have timesteps number of rows for lstm
                if len(residualretr) >= timestepsmodel:
                    smallindex = residualretr['index'].min()
                    residualretr = residualretr[residualretr['index']!= smallindex]
                    residualretr = pd.concat([residualretr, df], axis=0)
                    residualretr.to_csv(directoryloc + "SMA_ECCC_1H_residual.csv", header=True)
                else:
                    residualretr = pd.concat([residualretr, df], axis=0)
                    residualretr.to_csv(directoryloc + "SMA_ECCC_1H_residual.csv", header=True)
            else:
                df.to_csv(directoryloc+"SMA_ECCC_1H_residual.csv", header=True)
        else:
            dfend = df.tail(timestepsmodel * 2 - 1) # required number of records for combined lstm
            dfend.to_csv(directoryloc+"SMA_ECCC_1H_residual.csv", header=True)

        if isSchedulerhit:
            if path.exists(directoryloc+"accumulated_data_to_train.csv"):
                accdata = pd.read_csv(directoryloc+"accumulated_data_to_train.csv")
                accdata = pd.concat([accdata, df], axis=1)
                accdata.to_csv(directoryloc+"accumulated_data_to_train.csv", header=True)
            else:
                df.to_csv(directoryloc+"accumulated_data_to_train.csv", header=True)


        return df

