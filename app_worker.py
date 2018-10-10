from celery import Celery
import csv

app = Celery('tasks', broker='redis://localhost:6379/')


@app.task
def write_data_hist(x, y):
    with open(r'reward_history.csv', 'a', newline='') as csvfile:
        fieldnames = ['x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'x': x, 'y': y})
        x += 1
