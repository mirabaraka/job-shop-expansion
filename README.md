# job-shop-expansion

The job shop problem is an optimization problem that involves scheduling multiple jobs on multiple machines. Each job has a sequence of tasks that must be executed in a specific order, each task has a specific duration and can be processed on a specific machine. The problem is to find the optimal schedule for all jobs that minimizes the length of the schedule (makespan).
This problem can be solved using the open source Google OR-Tools library.
This repo provides an expansion on the original problem, handling more complex constraints.

### Input CSV Format
The program takes in three different csv files, you can replace these by your own custome input files:
1. **orders_input.csv:** This includes all orders/jobs that need to be scheduled.
 
  It should have the following columns:
  
  | Column Name | Description                                                   | Example Value |
  
  |-------------|---------------------------------------------------------------|-------------- |
  
  | job_number  | a unique number assigned to each job                          | 1             |
  
  | job_code    | a code that maps to which tasks would be included in this job | 11            |
  
  | start_date  | the date on which this job should start                       | 19/01/2025    |

2. **job_task_mapping.csv:** This includes each job code and the tasks mapped to that code along with each task operation time in hours and move time in hours(how much time is needed to go from one task to the next).

  It should have the following columns:
  
  | Column Name | Description                                                   | Example Value |
  
  |-------------|---------------------------------------------------------------|-------------- |
  
  | job_code  | a code that maps to which tasks would be included in this job   | 11 |
  
  | task_name    | a unique name for each task  | task_a |
  | task_time_hours    | length of task  | 5.5 |
  | move_time_hours    | length of move time  | 1.2 |

3. **employees_shift.csv:** This includes the number of days/shofts available for the schedule and the number of employees available on each day/shift. Each day/shift is represented by a new column.

  It should have the following columns:
  
  | Column Name | Description                                                   | Example Value |
  
  |-------------|---------------------------------------------------------------|-------------- |
  
  | employee  | employee name or number   | employee 1 |
  
  | day_shift_1    | a boolean that is set to 1 if the employee is available on that day/shift or 0 if employee is not available   | 1 |
  | day_shift_2    | a boolean that is set to 1 if the employee is available on that day/shift or 0 if employee is not available   | 1 |
  ...
  
