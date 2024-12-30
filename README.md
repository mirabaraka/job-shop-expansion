# job-shop-expansion

The job shop problem is an optimization problem that involves scheduling multiple jobs on multiple machines. Each job has a sequence of tasks that must be executed in a specific order, each task has a specific duration and can be processed on a specific machine. The problem is to find the optimal schedule for all jobs that minimizes the length of the schedule (makespan).
This problem can be solved using the open source Google OR-Tools library.
This repo provides an expansion on the original problem, handling more complex constraints.

### Input CSV Format
The program takes in three different csv files, you can replace these by your own custome input files:
1. **orders_input.csv:** This include all orders/jobs that need to be scheduled.
  It should have the following columns:
  | Column Name | Description                                                   | Example Value |
  |-------------|---------------------------------------------------------------|-------------- |
  | job_number  | a unique number assigned to each job                          | 1             |
  | job_code    | a code that maps to which tasks would be included in this job | 11            |
  | start_date  | the date on which this job should start                       | 19/01/2025    |
