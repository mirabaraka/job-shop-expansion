import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import collections
from ortools.sat.python import cp_model
import calendar
import time

class Solver_model:
    def __init__(self,start_hour,start_minute):
        # static tables/dictionaries
        self.job_task_mapping = pd.read_csv('./data/job_task_mapping.csv')
        

        # uploaded by the user given a specific template
        self.employee_shifts = pd.read_csv('./data/employees_shifts.csv')
        self.orders_df = pd.read_csv('./data/orders_input.csv')
        self.orders_df = self.orders_df.dropna()
        self.date = self.orders_df.start_date.min() # it's the start date of the schedule, could be any week day, not a weekend (Saturday/Sunday)
        self.start_hour = start_hour
        self.start_minute = start_minute

        # This dictionary maps the task names with the names of the machines available
        self.task_machine_mapping_dict =  {
                                                'task_a':['task_a_machine_1', 'task_a_machine_2'],
                                                 'task_c':['task_c_machine_1', 'task_c_machine_2']                                               
                                             } 

        self.main()
        
    # Data Pre-processing
    # employee data processing function
    def get_employee_availability(self):
        '''Takes as input the employee_shifts dataframe and the start date, it then creates a shift for each day and calculates the start and end time
            in minutes of each shift. It also automatically adds a gap between the end of one shift and the start of the next.
            The function also handles the weekends and assumes there is no shifts on Saturday and Sundays and thus adds a weekend gap
            Parameters:
            employees_df(dataframe): employees and shifts data
            date(string): start date of schedule

            Returns:
            shift_times: dictionary of each day shift and the start and end times in minutes
            shift_minutes: length of shift in minutes
            all_shifts:list of all shift names
        '''
        
        date_formatted = [int(self.date[6:10]), int(self.date[3:5]), int(self.date[0:2])]
        number_of_days = len(self.employee_shifts.columns) -1
        # assuming each shift is 9 hours
        shift_minutes = 9*60
        # assuming shift is between 8 am and 5 pm
        shutdown_time_minutes = (24 - 9) * 60
        # Given that no work on Saturdays & Sundays
        weekend_time_minutes = shutdown_time_minutes + (24*2*60)
        all_shifts = []
        shift_times = {}
        i = calendar.weekday(date_formatted[0], date_formatted[1],date_formatted[2])
        idx = 0
        for day in range(number_of_days):
            all_shifts.append("day_shift_" + str(day+1))
            if day == 0:
                shift_times["day_shift_" + str(day+1)] = [0,shift_minutes]
            else:
                if i % 5 == 0:
                    start_time_minutes = list(shift_times.values())[idx-1][1] + weekend_time_minutes
                else:
                    start_time_minutes = list(shift_times.values())[idx-1][1] + shutdown_time_minutes
                
                shift_times["day_shift_" + str(day+1)] = [ start_time_minutes ,start_time_minutes + shift_minutes]
            
            i+=1
            idx +=1
        self.shift_times= shift_times
        print('shift times',self.shift_times)
        self.all_shifts = all_shifts
        self.shift_minutes = shift_minutes
        

    def create_job_task_mapping(self):
        ''' create a table with each job code, tasks, and task time and move time in minutes
            Parameters:
            job_task_mapping(dataframe): dataframe containing mapping between job code, tasks, the task operation time and move time.
          
            Returns:
            final_job_task_mapping: dataframe that contains each job code mapped to its tasks and the time
            per task and the task time + move time between each task and the next.
            '''
        self.final_job_task_mapping = self.job_task_mapping.copy()
        self.final_job_task_mapping['task_time_minutes'] = round(self.final_job_task_mapping['task_time_hours'] * 60 ,0).astype('int')                 
        self.final_job_task_mapping['move_time_minutes'] = round(self.final_job_task_mapping['move_time_hours'] * 60 ,0).astype('int')
        print('final_job_task_mapping',self.final_job_task_mapping)
        
        
    # Create jobs from work orders and relevant assets
    def create_jobs(self):
        ''' Create a list of all jobs and tasks.
            Parameters:
            orders_df(dataframe): dataframe containing the jobs and their corresponding job codes 
            final_job_task_mapping(dataframe): output table from create_job_task_mapping function

            Returns:
            jobs_data(list): a list of each job containing a list of tuples (task name, duration of task)
        '''
        self.orders_df['job_number'] = self.orders_df['job_number'].astype('str')
        self.jobs_data = []
        
        for index,job in self.orders_df.iterrows():
            job_list = []
            
            for index,row in self.final_job_task_mapping[self.final_job_task_mapping['job_code'] == job['job_code']].iterrows():
                # get the duration(standard_time_minutes) from the final_job_task_mapping
                job_list.append((row['task_name'],row['task_time_minutes']))
        
            self.jobs_data.append(job_list)
        print('jobs_data',self.jobs_data)
        
    # function to get start dates
    def get_start_times(self):
        '''return the start time in minutes for each workorder'''
        self.start_date = pd.to_datetime(self.date, dayfirst = True)
        self.orders_df['start_time'] = (
                                            (pd.to_datetime(self.orders_df['start_date'], dayfirst = True
                                                           ) - self.start_date
                                            )/ pd.Timedelta(minutes=1)
                                       ).astype('int')
        # print(self.orders_df)
    
    def get_machines(self):
        '''
        Get the list of machine names.
        Note: for simplicity we set the machine name as the task name
        Parameters:
        jobs_data(list): has the name of tasks/machines for each job
        Returns:
        machines_list(list): list of machines/task names        
        '''
        self.machines_list = []
        for job in self.jobs_data:
            for task in job: 
                machine = task[0]
                if machine not in self.machines_list:
                        self.machines_list.append(machine)
        print('machines_list', self.machines_list)
         
        
    
    def get_asset_names(self):
        '''
        Get the list of task names that are required for tasks that can be executed on more than one machine.

        Parameters:
        task_machine_mapping_dict(dictionary): includes the task names that can be executed on more than one machine and the list of
        machine names.
        machines_list(list): list of all task/machine names produced by get_machines function.

        Returns:
        extended_machines_list(list): list of all machine names with tasks that have more than one machine 
        substituted with the new machine names. ex: 'task_a' will be substituted by 'task_a_machine_1'and 'task_a_machine_2'.
        '''
        
        machine_names = [x for x in self.task_machine_mapping_dict.values()]
        self.machine_names_list = []
        for y in machine_names:
            for x in y:
                self.machine_names_list.append(x)
        self.task_names = [x for x in self.task_machine_mapping_dict.keys()]
        print('task_names', self.task_names)
        self.extended_machines_list = self.machines_list.copy()
        # for each task name, if it is in the extended machines list then remove it and add the new machine-task name
        for x in self.task_names:
            try:
                self.extended_machines_list.remove(x)
            except:
                print("Task doesn't exist")
        
        self.extended_machines_list.extend(self.machine_names_list)
        self.extended_machines_list = list(set(self.extended_machines_list))
        # print(self.extended_machines_list)
    
    # calculate the horizon, which includes all tasks' durations and all tasks moving time
    def get_horizon(self):
        '''returns the overall duration of the whole schedule provided in employee_shifts table in minutes'''
        # using shift_times, take the end time of the last shift
        self.horizon = self.shift_times[list(self.shift_times)[-1]][1]
        
    
    def create_model(self):
        '''
        Creates the model, the model variables and the model constraints
        '''
        # Initiate the model.
        self.model = cp_model.CpModel()
        
        # Create storage objects to store all created variables
        
        # Named tuple to store information about created variables.
        self.task_type = collections.namedtuple("task_type", "start end interval")
            
        
        # dictionary to save all job/task interval variables
        self.all_tasks = {}
        
        # list to save all tasks interval variables assigned to each machine
        self.machine_to_intervals = collections.defaultdict(list)
        
        # list to save all interval variables for each employee
        self.employee_to_intervals = collections.defaultdict(list)
        
        # dictionary to store data for each boolean variable for each job, task, employee and shift
        self.employee_shift = {}
        
        #dictionary to save the boolean variables for assets that have more than one machine
        self.machine_to_task = {}
        
        #list of machines' interval variables
        self.machine_to_task_list = collections.defaultdict(list)
        
        # for each job_id/work order and each task_id/operation within the work order, create an interval variable
        for job_id, job in enumerate(self.jobs_data): 
            
            self.employee_shift[job_id] = {}
            self.machine_to_task[job_id] = {}
            
            for task_id, task in enumerate(job):
                # task is another word for operation name

                # for each task, get the operation name/machine name and the duration of the task
                machine, duration = task
                task_name = task[0]
                
                suffix = f"_{job_id}_{task_id}"

                # create interval variable
                start_var = self.model.NewIntVar(0, self.horizon, "start" + suffix)
                end_var = self.model.NewIntVar(0, self.horizon, "end" + suffix)
                interval_var = self.model.NewIntervalVar(
                    start_var, duration, end_var, "interval" + suffix
                )
                self.all_tasks[job_id, task_id] = self.task_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                # save the interval var in the machine_to_intervals list corresponding to the machine it uses
                self.machine_to_intervals[machine].append(interval_var)
                self.employee_shift[job_id][task_id] = {}
                self.machine_to_task[job_id][task_id] = {}

                # logic for assigning tasks to machines when there is more than one machine that can do the same task
                # only apply the following block of code for tasks that can be executed on more than one machine
                if task_name in self.task_names: 
                    
                    for machine_name in self.task_machine_mapping_dict[task_name]:
                        # for each machine in the machines that are mapped to the asset, create a machine interval                        
                        machine_suffix = f"_{job_id}_{task_id}_{machine_name}"
                        machine_start_var = self.model.NewIntVar(0, self.horizon, "start" + machine_suffix)
                        machine_end_var = self.model.NewIntVar(0, self.horizon, "end" + machine_suffix)
                        machine_duration_var = self.model.NewIntVar(0, duration, "duration" + machine_suffix) 
                        machine_interval_var = self.model.NewIntervalVar(
                            machine_start_var, machine_duration_var, machine_end_var, "interval" + machine_suffix
                                )
                        # create a boolean var for each combination of task and machine
                        self.machine_to_task[job_id][task_id][str(machine_name)] = self.model.NewBoolVar(f"boolean_{machine_suffix}")
                        # set the machine interval equal to the task interval only if the boolean of the task-machine equals 1
                        self.model.Add(start_var == machine_start_var).OnlyEnforceIf(self.machine_to_task[job_id][task_id][str(machine_name)])
                        self.model.Add(start_var != machine_start_var).OnlyEnforceIf(self.machine_to_task[job_id][task_id][str(machine_name)].Not())
                        self.model.Add(end_var == machine_end_var).OnlyEnforceIf(self.machine_to_task[job_id][task_id][str(machine_name)])
                        self.model.Add(end_var != machine_end_var).OnlyEnforceIf(self.machine_to_task[job_id][task_id][str(machine_name)].Not())                
                        self.model.Add(machine_duration_var == 0).OnlyEnforceIf(self.machine_to_task[job_id][task_id][str(machine_name)].Not())
                        self.model.Add(machine_duration_var == duration).OnlyEnforceIf(self.machine_to_task[job_id][task_id][str(machine_name)])
                        # save all intervals for each machine
                        self.machine_to_task_list[machine_name].append(machine_interval_var)
                
                # logic for assigning employees to tasks
                
                for shift in self.all_shifts: #for each day, a day has only one shift
                    self.employee_shift[job_id][task_id][shift]={}            
                    #looping through number of employees available on that day/shift
                    for x in range(len(self.employee_shifts[self.employee_shifts[shift] == 1][shift])): 
                        suffix = f"_{job_id}_{task_id}_{shift}_{'employee_' + str(x+1)}"
                        # for each employee, shift, and task create a boolean variable
                        self.employee_shift[job_id][task_id][shift]['employee_' + str(x+1)] = self.model.NewBoolVar(f"boolean_{suffix}")
                        # create an interval variable for each employee
                        employee_start_var = self.model.NewIntVar(self.shift_times[shift][0], self.shift_times[shift][1], "start" + suffix)
                        employee_end_var = self.model.NewIntVar(self.shift_times[shift][0], self.shift_times[shift][1], "end" + suffix)
                        employee_duration_var = self.model.NewIntVar(0, self.shift_minutes, "duration" + suffix) 
                        employee_interval_var = self.model.NewIntervalVar(
                            employee_start_var, employee_duration_var, employee_end_var, "interval" + suffix
                                )
                        # save interval variables for each employee
                        self.employee_to_intervals['employee_' + str(x+1) + str(shift)].append(employee_interval_var)
                        
                        # set the employee variable to the start variable only if the boolean of the combined employee,task,and shift equals 1
                        self.model.Add(start_var == employee_start_var).OnlyEnforceIf(self.employee_shift[job_id][task_id][shift]['employee_' + str(x+1)])
                        self.model.Add(start_var != employee_start_var).OnlyEnforceIf(self.employee_shift[job_id][task_id][shift]['employee_' + str(x+1)].Not())
                        self.model.Add(end_var == employee_end_var).OnlyEnforceIf(self.employee_shift[job_id][task_id][shift]['employee_' + str(x+1)])
                        self.model.Add(end_var != employee_end_var).OnlyEnforceIf(self.employee_shift[job_id][task_id][shift]['employee_' + str(x+1)].Not())                
                        self.model.Add(employee_duration_var == 0).OnlyEnforceIf(self.employee_shift[job_id][task_id][shift]['employee_' + str(x+1)].Not())
                        self.model.Add(employee_duration_var == duration).OnlyEnforceIf(self.employee_shift[job_id][task_id][shift]['employee_' + str(x+1)])
        
                                 
        # create constraint to assign each task to only one employee
        for job_id, job in enumerate(self.jobs_data):     
            for task_id, task in enumerate(job):
                self.employees_shift_booleans = []
                for shift in self.employee_shift[job_id][task_id]:
                    for employee in self.employee_shift[job_id][task_id][shift]:
                          self.employees_shift_booleans.append(self.employee_shift[job_id][task_id][shift][employee])
                
                self.model.Add(
                    sum(
                        shift_boolean for shift_boolean in self.employees_shift_booleans
                    ) == 1
                )
                
        
        # Create and add disjunctive constraints.
        # add no overlap between tasks on the same machine
        [self.model.AddNoOverlap(self.machine_to_intervals[machine]) for machine in self.extended_machines_list]
        
        # add no overlap between tasks for the same employee
        for employee_shift_list in self.employee_to_intervals:
            self.model.AddNoOverlap(self.employee_to_intervals[employee_shift_list])
            
        # only constraint to have only one machine assigned to any given task 
        for job_id, job in enumerate(self.jobs_data):     
            for task_id, task in enumerate(job):
                task_name = task[0]
                if task_name in self.task_names: 
                    booleans_list = []
                    for machine_name in self.task_machine_mapping_dict[task_name]:
                        booleans_list.append(self.machine_to_task[job_id][task_id][machine_name])
                
                    self.model.Add(
                            sum(machine_boolean for machine_boolean in booleans_list) == 1)
                
        # add no overlap between machine intervals on the same machine
        [self.model.AddNoOverlap(self.machine_to_task_list[machine_name]) for machine_name in self.machine_names_list 
         if self.machine_to_task_list[machine_name]]
        
        # adding moving time between tasks and precedences inside a job
        for job_id, job in enumerate(self.jobs_data):
            filtered_df = self.final_job_task_mapping[self.final_job_task_mapping['job_code'] == self.orders_df['job_code'].iloc[job_id]]
            for task_id in range(len(job) - 1):
                self.model.Add(
                    self.all_tasks[job_id, task_id + 1].start >= self.all_tasks[job_id, task_id].end + filtered_df['move_time_minutes'].iloc[task_id]
                )   
        
        # create constraint to start work orders at the given dates in the order_input sheet
        for job_id, job in enumerate(self.jobs_data):
            filtered_df = self.orders_df.iloc[job_id]     
            self.model.Add(
                self.all_tasks[job_id, 0].start >= filtered_df['start_time']
            ) 
        
    # create the objective function
    # objective is to minimize the end date of all tasks in all jobs
    def set_solve_objective(self):
        self.new_var = [self.all_tasks[job_id, task_id].end 
                        for job_id, job in enumerate(self.jobs_data) for task_id, task in enumerate(job)]
        self.obj_var = sum(self.new_var)
        
        self.model.Minimize(self.obj_var)
        
        # Creates the solver and solve.
        self.solver = cp_model.CpSolver()
        # Sets a time limit of 10 seconds.
        # self.solver.parameters.max_time_in_seconds = 10.0
        #set number of workers
        # self.solver.parameters.num_search_workers = 4
        self.new_status = self.solver.Solve(self.model)
        
    
    def get_solution(self):
    
        if self.new_status == cp_model.OPTIMAL or self.new_status == cp_model.FEASIBLE:
            print("Solution:")
            # Named tuple to manipulate solution information.
            self.assigned_task_type = collections.namedtuple(
                "assigned_task_type", "start job index duration"
            )
            if self.new_status == cp_model.OPTIMAL:
                print('Optimal solution found')
            # Create one list of assigned tasks per machine.
            self.assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(self.jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    self.assigned_jobs[machine].append(
                        self.assigned_task_type(
                            start=self.solver.Value(self.all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[1],
                        )
                    )
        
        
            # Finally print the solution found.
            print(f"Optimal Schedule Length: {self.solver.ObjectiveValue()}")
        
            # Statistics.
            print("\nStatistics")
            print(f"  - conflicts: {self.solver.NumConflicts()}")
            print(f"  - branches : {self.solver.NumBranches()}")
            print(f"  - wall time: {self.solver.WallTime()}s")
        else:
            print("No solution found.")
        
            # Statistics.
            print("\nStatistics")
            print(f"  - conflicts: {self.solver.NumConflicts()}")
            print(f"  - branches : {self.solver.NumBranches()}")
            print(f"  - wall time: {self.solver.WallTime()}s")
    
    # Extract the machine name
    def extract_machine_name(self):
        self.extracted_machine_name = {}
        for job_id,job in enumerate(self.jobs_data):
            
            self.extracted_machine_name[job_id]={}
            for task_id, task in enumerate(job):
                task_name = task[0]
                if task_name in self.task_names: 
                    
                    for machine_name in self.task_machine_mapping_dict[task_name]:
              
                        if self.solver.Value(self.machine_to_task[job_id][task_id][str(machine_name)]) == 1:
                            self.extracted_machine_name[job_id][task_id] = machine_name
                            
    
    # Extract model output
    def extract_solution(self):
        # Extracting solution
        start=[]
        end = []
        self.final_jobs_dict = {}
        
        for job_id,job in enumerate(self.jobs_data):
            self.final_jobs_dict[job_id]={}
            
            for task_id, task in enumerate(job):
                task_name = task[0]
                self.final_jobs_dict[job_id][task_id]={}
                
                start=[]
                end = []
                for machine in self.machines_list:
                    for assigned_task in self.assigned_jobs[machine]:
                        if (assigned_task.job == job_id) and (assigned_task.index == task_id):
                            start.append(assigned_task.start)
                            end.append(assigned_task.start + assigned_task.duration)
        
                            self.final_jobs_dict[job_id][task_id]['start']=start[0]
                            self.final_jobs_dict[job_id][task_id]['end']=end[-1]
                            if task_name in self.task_names: 
                                machine = self.extracted_machine_name[job_id][task_id]
                            self.final_jobs_dict[job_id][task_id]['machine']=machine
      
    
    
        
    def create_schedule_table(self):
        from datetime import datetime, timedelta
        self.new_df = pd.DataFrame(columns=['Task','Start','Finish','Tag'])
        for key in self.final_jobs_dict:
            job_number = self.orders_df.iloc[[key]]['job_number'].values[0]
            for task_key in self.final_jobs_dict[key]:
                data = [[job_number,self.final_jobs_dict[key][task_key]['start'],self.final_jobs_dict[key][task_key]['end'],self.final_jobs_dict[key][task_key]['machine']]]
                new_r = pd.DataFrame(data, columns=['Task','Start','Finish','Tag'])
        
                self.new_df = pd.concat([self.new_df,new_r])
        # create time as a variable
        base_date = datetime(int(self.date[6:10]), int(self.date[3:5]), int(self.date[0:2]), self.start_hour, self.start_minute, 0)
         
        # Adding a new column 'start_date' by adding hours to the base date
        self.new_df['Start'] = base_date + pd.to_timedelta(self.new_df['Start'], unit='m')
        self.new_df['Finish'] = base_date + pd.to_timedelta(self.new_df['Finish'], unit='m')
        
        
        self.final_df = self.new_df.copy()   #change
        self.final_df = self.final_df.reset_index(drop = True)

        for index, row in self.final_df.iterrows():
            print(index)
            print(self.final_df.at[index, 'Tag'])
            
        print(self.final_df)    
        self.final_df['Task'] = 'Job ' + self.final_df['Task'].astype('string')  #change
        self.final_df['Asset'] = self.final_df['Tag'].astype('string') #change
        self.final_df = self.final_df.drop(columns=['Tag'])
        print(self.final_df)
    
        return self.final_df
     
    
    def main(self):
        start = time.time()
        self.get_employee_availability()
        end = time.time()
        print("step 1:", end - start)
        start = time.time()
        self.create_job_task_mapping()
        end = time.time()
        print("step 2:", end - start)
        start = time.time()
        self.create_jobs()
        end = time.time()
        print("step 3:", end - start)
        start = time.time()
        self.get_start_times()
        end = time.time()
        print("step 4:", end - start)
        start = time.time()
        self.get_machines()
        end = time.time()
        print("step 5:", end - start)
        start = time.time()
        self.get_asset_names()
        end = time.time()
        print("step 6:", end - start)
        start = time.time()
        self.get_horizon()
        end = time.time()
        print("step 7:", end - start)
        start = time.time()
        print(self.horizon)
        end = time.time()
        print("step 8:", end - start)
        start = time.time()
        self.create_model()
        end = time.time()
        print("step 9:", end - start)
        start = time.time()
        self.set_solve_objective()
        end = time.time()
        print("step 10:", end - start)
        start = time.time()
        self.get_solution()
        end = time.time()
        print("step 11:", end - start)
        start = time.time()
        self.extract_machine_name()
        end = time.time()
        print("step 12:", end - start)
        start = time.time()
        self.extract_solution()
        end = time.time()
        print("step 13:", end - start)
        
        final_df = self.create_schedule_table()
        
    
        return final_df

def main():
	sm = Solver_model(8,0)
	# provide snap and code
	fig = px.timeline(sm.final_df, x_start="Start", x_end="Finish", y="Task",  #change
				  color = 'Asset',  title = 'Scheduling', color_discrete_sequence=px.colors.qualitative.Set2)
	fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
	fig.update_layout(
		xaxis_title="Date" )
	fig.update_yaxes(categoryorder='category ascending')

	fig.show()
	
if __name__ == "__main__":
    main()