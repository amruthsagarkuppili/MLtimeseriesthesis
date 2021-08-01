
directoryloc = 'flaskapplication/actions/generatedcsv/'
historypredloc= '/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/generatedcsv/history_predictions.csv'
predloc = '/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/generatedcsv/predictions.csv'
scalermodel = '/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/scalermodelrepo/'
ECCClink = 'http://www.meds-sdmm.dfo-mpo.gc.ca/alphapro/wave/waveshare/csvData/c44258_csv.zip'
SMAlink = 'https://www.smartatlantic.ca/erddap/tabledap/SMA_halifax.csv?station_name%2Ctime%2Clongitude%2Clatitude%2Cwind_spd_avg%2Cwind_spd_max%2Cwind_dir_avg%2Cair_temp_avg%2Cair_pressure_avg%2Csurface_temp_avg%2Cwave_ht_max%2Cwave_ht_sig%2Cwave_dir_avg%2Cwave_spread_avg%2Cwave_period_max%2Ccurr_spd_avg%2Ccurr_dir_avg%2Ccurr_spd2_avg%2Ccurr_dir2_avg%2Ccurr_spd3_avg%2Ccurr_dir3_avg%2Ccurr_spd4_avg%2Ccurr_dir4_avg%2Ccurr_spd5_avg%2Ccurr_dir5_avg%2Ccurr_spd6_avg%2Ccurr_dir6_avg%2Ccurr_spd7_avg%2Ccurr_dir7_avg%2Ccurr_spd8_avg%2Ccurr_dir8_avg%2Ccurr_spd9_avg%2Ccurr_dir9_avg%2Ccurr_spd10_avg%2Ccurr_dir10_avg%2Ccurr_spd11_avg%2Ccurr_dir11_avg%2Ccurr_spd12_avg%2Ccurr_dir12_avg%2Ccurr_spd13_avg%2Ccurr_dir13_avg%2Ccurr_spd14_avg%2Ccurr_dir14_avg%2Ccurr_spd15_avg%2Ccurr_dir15_avg%2Ccurr_spd16_avg%2Ccurr_dir16_avg%2Ccurr_spd17_avg%2Ccurr_dir17_avg%2Ccurr_spd18_avg%2Ccurr_dir18_avg%2Ccurr_spd19_avg%2Ccurr_dir19_avg%2Ccurr_spd20_avg%2Ccurr_dir20_avg&time%3E=2013-11-07T16%3A23%3A01Z&time%3C=2020-09-07T18%3A53%3A01Z'

thresholds = {
    "sig_wv_ht_threshold_safe" : 1.25,
    "sig_wv_ht_threshold_moderate" : 2,
    "mx_wv_ht_threshold_safe" : 5,
    "mx_wv_ht_threshold_moderate" : 6,
    "wv_prd_threshold_safe" : 6,
    "wv_prd_threshold_moderate" : 7,
    "wnd_spd_threshold_safe" : 10,
    "wnd_spd_ht_threshold_moderate" : 15
              }

timestepsmodel = 3
anomalytimestepsmodel = 1
featurecolumns = 6
targetcolumns = 4
confidence = 95