import numpy as np
import csv
def calculate_completion_metrics(completed_c_tasks,DDL):
    total_tasks = len(completed_c_tasks)
    if total_tasks == 0:
        return 0, 0  # 如果没有完成的任务，返回 0

    on_time_count = sum(1 for task in completed_c_tasks if task[2] >= 0)
    on_time_ratio = on_time_count / total_tasks

    total_execution_time = sum(DDL - task[2] for task in completed_c_tasks)

    return on_time_ratio, total_execution_time


def calculate_interruption_ratio(task_array):
    total_steps = len(task_array)
    if total_steps == 0:
        return 0  # 如果没有 step，返回 0

    interruption_count = sum(1 for step in task_array if step == False)
    interruption_ratio = interruption_count / total_steps

    return interruption_ratio

def calculate_security_level(c_level):
    total_security = sum(c_level)
    return total_security

