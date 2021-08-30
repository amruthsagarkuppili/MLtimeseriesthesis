
######################################
#                                    #
# @author Amruth Sagar Kuppili       #
# @university Dalhousie University   #
# @ID B00844506                      #
#                                    #
######################################

from dash import Dash
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from collections import deque
import plotly
import plotly.graph_objs as go
import pandas as pd
import os as os
from os import path
import numpy as np

from flaskapplication.actions.mltasks.variables import thresholds


from flaskapplication.actions.schedulerjobs.schedulejobshelper import OperateSchedulers

anomaly = deque(maxlen=1)
sightX = deque(maxlen=12)
mxwvhtX = deque(maxlen=12)
wvprdX = deque(maxlen=12)
wndspdX = deque(maxlen=12)
sightY = deque(maxlen=12)
mxhtY = deque(maxlen=12)
wvprdY = deque(maxlen=12)
wndspdY = deque(maxlen=12)
sighterr = deque(maxlen=12)
mxwvhterr = deque(maxlen=12)
wvprderr = deque(maxlen=12)
wndspderr = deque(maxlen=12)
suffix = ""
titletxt=""
fig = go.Figure()
onesecond = 1000
onemin = 60 * onesecond
onehour = 60 * onemin
oneday = 24 * onehour
predictionloc = '/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/generatedcsv/predictions.csv'
historypredloc= '/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/generatedcsv/history_predictions.csv'

def init_dashboard(server):
    """Create a Plotly Dash dashboard."""

    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets = ['/static/css/styles.css',
                                'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css',
                                dbc.themes.BOOTSTRAP]

    )
    tab_style = {
        'borderBottom': '1px solid #d6d6d6',
        'padding': '6px',
        'fontWeight': 'bold'
    }

    tab_selected_style = {
        'borderTop': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'backgroundColor': '#34495E',
        'color': 'white',
        'padding': '6px'
    }

    dash_app.layout = html.Div(
        [
            html.Div([
                dcc.Tabs(id='tabs-data', value='sig_wv_ht',children=[
                    dcc.Tab(label='Significant Wave Height', value='sig_wv_ht', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Maximum Wave Height',value='wv_ht_max', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Wave Period Maximum',value='wv_prd_max', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Wind Speed',value='wnd_spd', style=tab_style, selected_style=tab_selected_style)
                ]),
                html.Br(),
                dbc.Alert(
                    html.P("ALERT! SYSTEM HAS DETECTED CLIMATIC ABBERATION IN RECENT PREDICTION. PLEASE WATCH OUT.", className="alert-content"),
                    id="alert-anomaly",
                    dismissable=True,
                    is_open=False,
                    color='danger'
                ),
                dcc.Graph(id='tab-data', animate=False)
            ], className='sim-graph'
            ),

            dcc.Interval(
                id='graph-update',
                interval= 0.1*onemin
            ),
            html.Div(html.Div([
                html.Button('start prediction', id='startpred',
                            className='waves-effect waves-light btn-large main-button btn-1 land-buttons'),
                html.Button('stop prediction', id='stoppred',
                            className='waves-effect waves-light btn-large main-button btn-2 land-buttons'),
                html.A(html.Button('Home Page', id='homepg',
                            className='waves-effect waves-light btn-large main-button btn-2 land-buttons'), href= f'http://localhost:5000'),
                html.Script()

            ], className='form-container-dash'), className='container'),
            html.Div(id='jobsoutput', className='status-text')
        ]
    )

    init_callbacks(dash_app)

    return dash_app.server


def init_callbacks(dash_app):
    @dash_app.callback(
        [Output('tab-data', 'figure'),Output('alert-anomaly', 'is_open')],
        [Input('graph-update', 'n_intervals'),
         Input('tabs-data','value')],
        [State('alert-anomaly', 'is_open')],
    )
    def update_graph(interval_number, tabid, is_open):



        X = deque()
        Y = deque()
        stderror = None
        X.clear()
        Y.clear()
        fig = go.Figure()
        suffix = ""
        safethresh = 0
        moderthresh = 0
        # default layout
        layout = dict()
        isconcatenated = False
        tabsig, tabmxwv, tabwvprd, tabwnd = False, False, False, False
        titletxt = ""
        listtitle =[]
        if tabid=='sig_wv_ht':
            tabsig = True
        elif tabid=='wv_ht_max':
            tabmxwv = True
        elif tabid=='wv_prd_max':
            tabwvprd = True
        elif tabid=='wnd_spd':
            tabwnd = True


        if path.exists(predictionloc):
            predictiondf = pd.read_csv(predictionloc, index_col='time')
            historydf = pd.read_csv(historypredloc, index_col='time')
            if interval_number == None:
                fig = go.Figure()
                clearalldq()
                historydftail = historydf.tail(5)
                predictiondf = pd.concat([historydftail,predictiondf], axis=0)
                isconcatenated = True

            for index, row in predictiondf.iterrows():
                appendalldq(index, row)
                if tabsig:
                    X = sightX
                    Y = sightY
                    stderror = sighterr
                    titletxt = 'Significant Wave height Pattern'
                    suffix = 'meters'
                    safethresh = thresholds['sig_wv_ht_threshold_safe']
                    moderthresh = thresholds['sig_wv_ht_threshold_moderate']
                elif tabmxwv:
                    X = mxwvhtX
                    Y = mxhtY
                    stderror = mxwvhterr
                    titletxt = 'Maximum Wave height Pattern'
                    suffix = 'meters'
                    safethresh = thresholds['mx_wv_ht_threshold_safe']
                    moderthresh = thresholds['mx_wv_ht_threshold_moderate']
                elif tabwvprd:
                    X = wvprdX
                    Y = wvprdY
                    stderror = wvprderr
                    titletxt = 'Wave Period Pattern'
                    suffix = 'seconds'
                    safethresh = thresholds['wv_prd_threshold_safe']
                    moderthresh = thresholds['wv_prd_threshold_moderate']
                elif tabwnd:
                    X = wndspdX
                    Y = wndspdY
                    stderror = wndspderr
                    titletxt = 'Wind Speed Pattern'
                    suffix = 'meters/second'
                    safethresh = thresholds['wnd_spd_threshold_safe']
                    moderthresh = thresholds['wnd_spd_ht_threshold_moderate']

            ytitle = titletxt.rsplit(' ', 1)[0] + ' (' + suffix + ')'  # trimming pattern  word
            layout = dict(xaxis=dict(title='Timeline', autorange=True, ticklabelposition='outside bottom', ticklen=2),
                          yaxis=dict(title=ytitle, autorange=True),
                          title=dict(text=titletxt, font=dict(size=22, color='black'), x=0.1))

            # To ensure already pulled rows are not being concatenated in the history_predictions
            if isconcatenated:
                historypred = pd.concat([historydf, predictiondf.tail(1)], axis=0)
            else:
                historypred = pd.concat([historydf, predictiondf], axis=0)
            historypred.to_csv(historypredloc)
            hstrytail = historypred.tail(1)
            anomaly.clear()
            anomaly.append(hstrytail['isanomaly'].values[0])

            os.remove(predictionloc)

        else:
            if interval_number == None:
                fig = go.Figure()
                clearalldq()
                historydf = pd.read_csv(historypredloc, index_col='time')
                historydftail = historydf.tail(5)
                for index, row in historydftail.iterrows():
                    appendalldq(index, row)
                    if tabsig:
                        X = sightX
                        Y = sightY
                        stderror = sighterr
                        titletxt = 'Significant Wave height Pattern'
                        suffix = 'meters'
                        safethresh = thresholds['sig_wv_ht_threshold_safe']
                        moderthresh = thresholds['sig_wv_ht_threshold_moderate']
                    elif tabmxwv:
                        X = mxwvhtX
                        Y = mxhtY
                        stderror = mxwvhterr
                        titletxt = 'Maximum Wave height Pattern'
                        suffix = 'meters'
                        safethresh = thresholds['mx_wv_ht_threshold_safe']
                        moderthresh = thresholds['mx_wv_ht_threshold_moderate']
                    elif tabwvprd:
                        X = wvprdX
                        Y = wvprdY
                        stderror = wvprderr
                        titletxt = 'Wave Period Pattern'
                        suffix = 'seconds'
                        safethresh = thresholds['wv_prd_threshold_safe']
                        moderthresh = thresholds['wv_prd_threshold_moderate']
                    elif tabwnd:
                        X = wndspdX
                        Y = wndspdY
                        stderror = wndspderr
                        titletxt = 'Wind Speed Pattern'
                        suffix = 'meters/second'
                        safethresh = thresholds['wnd_spd_threshold_safe']
                        moderthresh = thresholds['wnd_spd_ht_threshold_moderate']
                hstrylast = historydf.tail(1)
                anomaly.clear()
                anomaly.append(hstrylast['isanomaly'].values[0])

            else:

                if tabsig:
                    X = sightX
                    Y = sightY
                    stderror = sighterr
                    titletxt = 'Significant Wave height Pattern'
                    suffix = 'meters'
                    safethresh = thresholds['sig_wv_ht_threshold_safe']
                    moderthresh = thresholds['sig_wv_ht_threshold_moderate']
                elif tabmxwv:
                    X = mxwvhtX
                    Y = mxhtY
                    stderror = mxwvhterr
                    titletxt = 'Maximum Wave height Pattern'
                    suffix = 'meters'
                    safethresh = thresholds['mx_wv_ht_threshold_safe']
                    moderthresh = thresholds['mx_wv_ht_threshold_moderate']
                elif tabwvprd:
                    X = wvprdX
                    Y = wvprdY
                    stderror = wvprderr
                    titletxt = 'Wave Period Pattern'
                    suffix = 'seconds'
                    safethresh = thresholds['wv_prd_threshold_safe']
                    moderthresh = thresholds['wv_prd_threshold_moderate']
                elif tabwnd:
                    X = wndspdX
                    Y = wndspdY
                    stderror = wndspderr
                    titletxt = 'Wind Speed Pattern'
                    suffix = 'meters/second'
                    safethresh = thresholds['wnd_spd_threshold_safe']
                    moderthresh = thresholds['wnd_spd_ht_threshold_moderate']

            ytitle = titletxt.rsplit(' ',1)[0] + ' (' + suffix + ')' # trimming pattern  word
            layout = dict(xaxis=dict(title='Timeline',autorange=True, ticks='outside', ticklen=10, tickcolor='#FDFEFE'),
                               yaxis=dict(title=ytitle,autorange=True,ticks='outside', ticklen=10, tickcolor='#FDFEFE'),
                               title = dict(text=titletxt, font=dict(size=22,color='black'), x=0.1),
                               plot_bgcolor='#EAF2F8',
                          )

        for i in range(len(X)):
            listtitle.append(titletxt)

        # lbl = list()
        # for i in Y:
        #   if i < safethresh:
        #       lbl.append('safe')
        #   elif i >= safethresh and i <= moderthresh:
        #       lbl.append('moderate')
        #   else:
        #       lbl.append('danger')
        #
        # layoutdf = pd.DataFrame({'label':lbl,'x':X,'y':Y})
        # for lbl in layoutdf['label'].unique():
        #     dfp = layoutdf[layoutdf['label'] == lbl]
        #     # print(dfp)
        #     fig.add_trace(go.Scatter(x=dfp['x'], y=dfp['y'],
        #                               name=lbl,
        #                               mode='lines+markers',
        #                               marker=dict(color=colorleg[lbl], size=12)
        #                               ))
        data = go.Scatter(
            x=list(X),
            y=list(Y),
            name=' ',
            error_y= dict(array=list(stderror)),
            mode='lines+markers',

            marker = dict(size=12, color=list(map(SetColor, Y, listtitle))),
            line = dict(width=2.5, color='#2471A3'),
            showlegend = False
        )
        ldatablue = go.Scatter(x=[None], y=[None], mode='markers',
                       marker=dict(size=12, color='green'),
                       legendgroup='Safe', showlegend=True, name='Safe')

        ldataorange = (go.Scatter(x=[None], y=[None], mode='markers',
                       marker=dict(size=12, color='#F39C12'),
                       legendgroup='moderate', showlegend=True, name='Moderate'))

        ldatared = (go.Scatter(x=[None], y=[None], mode='markers',
                                  marker=dict(size=12, color='red'),
                                  legendgroup='danger', showlegend=True, name='Danger'))


        fig.add_trace(data)
        fig.add_trace(ldatablue)
        fig.add_trace(ldataorange)
        fig.add_trace(ldatared)
        fig.update_layout(layout)
        fig.add_hline(y=0, line_dash='dash', line_color='green', annotation_text="safe zone",line_width=3,annotation_font_color="#1D8348",
                      annotation_position="top right", annotation=dict(font_size=20, font_family="Times New Roman"))
        fig.add_hline(y=safethresh, line_dash='dash', line_color = '#F39C12',annotation_text="moderate zone",line_width=3,annotation_font_color="#F39C12",
              annotation_position="top right", annotation=dict(font_size=20, font_family="Times New Roman"))
        fig.add_hline(y=moderthresh, line_dash='dash', line_color='red',annotation_text="danger zone",line_width=3,annotation_font_color="#C0392B",
              annotation_position="top right",annotation=dict(font_size=20, font_family="Times New Roman"))
        fig.add_hrect(y0=0, y1=safethresh, line_width=0, fillcolor="green", opacity=0.1)
        fig.add_hrect(y0=safethresh, y1=moderthresh, line_width=0, fillcolor="#F39C12", opacity=0.15)
        # fig.add_hrect(y0=moderthresh, y1=100,  line_width=0, fillcolor="red", opacity=0.1)


        # return {'data': [data], 'layout': layout}, titletxt
        return fig, anomaly[0]


    @dash_app.callback(
        Output('jobsoutput', 'children'),
        Input('startpred', 'n_clicks'),
        Input('stoppred', 'n_clicks'),
    )
    def operateJobs(click1, click2):
        operate = OperateSchedulers()
        status = operate.getstatus()
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'startpred' in changed_id:
            status = operate.jobsForEveryHour()
        elif 'stoppred' in changed_id:
            status = operate.stopJobs()
        return status


    # Clearing dq's whenever the page is newly opened
    def clearalldq():
        sightX.clear()
        sightY.clear()
        sighterr.clear()
        mxwvhtX.clear()
        mxhtY.clear()
        mxwvhterr.clear()
        wvprdX.clear()
        wvprdY.clear()
        wvprderr.clear()
        wndspdX.clear()
        wndspdY.clear()
        wndspderr.clear()

    # appending data into the dq
    def appendalldq(index, row):
        sightX.append(index)
        sightY.append(row['sig_wv_ht'])
        sighterr.append(row['stderrsight'])
        mxwvhtX.append(index)
        mxhtY.append(row['mx_wv_ht'])
        mxwvhterr.append(row['stderrmxwvht'])
        wvprdX.append(index)
        wvprdY.append(row['max_wv_prd'])
        wvprderr.append(row['stderrwvprd'])
        wndspdX.append(index)
        wndspdY.append(row['wnd_spd'])
        wndspderr.append(row['stderrwndspd'])

    def SetColor(y, titlet):
        low = 0
        upper = 0
        if 'Significant' in titlet:
            low = thresholds['sig_wv_ht_threshold_safe']
            upper = thresholds['sig_wv_ht_threshold_moderate']
        elif 'Maximum' in titlet:
            low = thresholds['mx_wv_ht_threshold_safe']
            upper = thresholds['mx_wv_ht_threshold_moderate']
        elif 'Period' in titlet:
            low = thresholds['wv_prd_threshold_safe']
            upper = thresholds['wv_prd_threshold_moderate']
        elif 'Speed' in titlet:
            low = thresholds['wnd_spd_threshold_safe']
            upper = thresholds['wnd_spd_ht_threshold_moderate']
        if (y < low):
            return "green"
        elif (low <= y <= upper):
            return "#F39C12"
        elif (y > upper):
            return "red"





