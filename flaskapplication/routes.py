from flask import render_template, Blueprint

from flaskapplication.actions.mltasks.mlorganizer import mlorganizer
from flaskapplication.actions.schedulerjobs.schedulejobshelper import OperateSchedulers
import math
# Blueprint Configuration
home_bp = Blueprint(
    'home_bp', __name__
)

@home_bp.route("/")
def home():
    return render_template('index.html')


@home_bp.route("/train", methods=['POST'])
def train():
    mlorg = mlorganizer()
    # residualECCC, residualSMA = mlorg.dataPreProcess(False)
    # if all(temp is None for temp in [residualECCC, residualSMA]):
    #     df = "already trained on latest pull"
    # else:
    #     df = mlorg.train(residualECCC, residualSMA)
    df = mlorg.train(False, False)
    return render_template('index.html', trainframe=df)


@home_bp.route("/predict", methods=['POST'])
def predict():
    mlorg = mlorganizer()
    informationDict = mlorg.predict(True)
    predictions = informationDict["predictedvalue"]
    predictedValue = None
    lowersight = round(predictions['sig_wv_ht'].values[0]-predictions['stderrsight'].values[0], 2)
    uppersight = round(predictions['sig_wv_ht'].values[0] + predictions['stderrsight'].values[0], 2)
    lowermxwvht = round(predictions['mx_wv_ht'].values[0] - predictions['stderrmxwvht'].values[0], 2)
    uppermxwvht = round(predictions['mx_wv_ht'].values[0] + predictions['stderrmxwvht'].values[0], 2)
    lowerwvprd = round(predictions['max_wv_prd'].values[0] - predictions['stderrwvprd'].values[0], 2)
    upperwvprd = round(predictions['max_wv_prd'].values[0] + predictions['stderrwvprd'].values[0], 2)
    lowerwndspd = round(predictions['wnd_spd'].values[0] - predictions['stderrwndspd'].values[0], 2)
    upperwndspd = round(predictions['wnd_spd'].values[0] + predictions['stderrwndspd'].values[0], 2)
    if lowersight <= 0:
        lowersight=0
    if lowermxwvht <= 0:
        lowermxwvht = 0
    if lowerwvprd <= 0:
        lowerwvprd = 0
    if lowerwndspd <= 0:
        lowerwndspd = 0
    if informationDict["message"] is None:

        return render_template('index.html',
                               predictedValue = True,
                               time = predictions.index,
                               sight = predictions['sig_wv_ht'].values,
                               lowersight = lowersight,
                               uppersight = uppersight,
                               mxwvht = predictions['mx_wv_ht'].values,
                               lowermxwvht=lowermxwvht,
                               uppermxwvht=uppermxwvht,
                               wvprd = predictions['max_wv_prd'].values,
                               lowerwvprd=lowerwvprd,
                               upperwvprd=upperwvprd,
                               wndspd = predictions['wnd_spd'].values,
                               lowerwndspd=lowerwndspd,
                               upperwndspd=upperwndspd,
                               isanomaly=predictions['isanomaly'].values,
                               actualValue=informationDict["originalvalue"])
    else:
        return render_template('index.html', message=str(informationDict["message"]))



