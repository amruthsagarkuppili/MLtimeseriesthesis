import logging
from apscheduler.schedulers.background import BackgroundScheduler
import os as os
import apscheduler.schedulers.base as base


scheduler = BackgroundScheduler()
logging.basicConfig(filename='scheduler.log', level=logging.DEBUG)

class ScheduleJobs:

    def deleteUpdateFile(self):
        logging.debug("deleted")
        os.remove("/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/generatedcsv/ECCC_residual.csv")
        os.remove("/Users/amruthkuppili/Desktop/proj/SMAFlaskOrganized/flaskapplication/actions/generatedcsv/SMA_residual.csv")

    def startPrediction(self):
        from flaskapplication.actions.mltasks.mlorganizer import mlorganizer
        mlorg = mlorganizer()
        mlorg.dataPreProcess(True)
        mlorg.predict(False)




class OperateSchedulers:

    def jobsForEveryHour(self):
        if scheduler.state != base.STATE_RUNNING:
            scheduler.start()
            schedulejobs = ScheduleJobs()
            scheduler.add_job(schedulejobs.deleteUpdateFile, 'cron', id='delete_file_sched', year='*', month='*',
                              day='*',
                              week='*', day_of_week='*', hour='*', minute=55, second=0)
            scheduler.add_job(schedulejobs.startPrediction, 'cron', id='start_prediction_sched', year='*', month='*',
                              day='*',
                              week='*', day_of_week='*', hour='*', minute=56, second=0)
            return "Scheduler Jobs Started"
        else:
            return "Jobs Already Running"

    def stopJobs(self):
        if scheduler.state != base.STATE_STOPPED:
            scheduler.shutdown(wait=True)
            return "jobs are removed and operation is shutting down....."
        else:
            return "No Scheduler is running"

    def getstatus(self):
        if scheduler.state == base.STATE_RUNNING:
            return "Jobs are running"
        elif scheduler.state == base.STATE_STOPPED:
            return "No Jobs are scheduled, click on Start Prediction"
