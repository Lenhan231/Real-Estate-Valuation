from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler

def display():
    print("This function has been executed")

scheduler = BlockingScheduler()
scheduler.add_job(display, 'interval', seconds = 5)

scheduler.start()