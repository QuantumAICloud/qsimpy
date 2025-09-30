from multiprocessing import Process
from env_creator import qsimpy_env_creator
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from qsimpy import TaskStatus 
import numpy as np 
import random 

class HeuristicSolutions:
    def __init__(self, env, num_episodes=100):
        self.env = env
        self.num_episodes = num_episodes
        self.results = []
        self.rr_index = 0
        self.greedy_index = 0
        
        # --- NEW: Attributes for WRR ---
        self.wrr_list = []
        self.wrr_index = 0
        # --- END NEW ---

    def _initialize_episode_strategies(self):
        """Initializes/resets strategies that require setup at the start of an episode."""
        # Create the weighted list for Weighted Round Robin (WRR) based on CLOPS
        qnodes = self.env.unwrapped.qnodes
        self.wrr_list = []
        # We use a base multiplier to avoid tiny floats and ensure integer repetitions
        clops_values = [node.clops for node in qnodes]
        min_clops = min(c for c in clops_values if c > 0)
        
        for i, node in enumerate(qnodes):
            # The weight is how many times a node's index appears in the list
            weight = int(round(node.clops / min_clops))
            self.wrr_list.extend([i] * weight)
        
        random.shuffle(self.wrr_list) # Shuffle to distribute nodes evenly
        self.wrr_index = 0

    def run(self, control):
        """
        Run the heuristic solutions for the given algorithm (control).
        Args:
            - control (str): The heuristic algorithm to use.
        """
        self.results = []
        self.env.unwrapped.round = 1

        for episode_num in range(self.num_episodes):
            arr_temp = {"total_completion_time": 0.0, "rescheduling_count": 0.0}
            terminated = False

            self.env.reset(seed=episode_num)
            self.env.unwrapped.setup_quantum_resources()
            self.rr_index = 0
            self.greedy_index = 0
            
            # --- NEW: Initialize strategies for the episode ---
            self._initialize_episode_strategies()
            # --- END NEW ---

            while not terminated:
                if control == "greedy":
                    action = self.greedy(self.greedy_index)
                elif control == "random":
                    action = self.random()
                elif control == "round_robin":
                    action = self.round_robin()
                elif control == "greedy_error":
                    action = self.greedy_error(self.greedy_index)
                elif control == "ect":
                    action = self.earliest_completion_time()
                elif control == "sjf":
                    action = self.shortest_job_first()
                # --- NEW ALGORITHMS ---
                elif control == "luf":
                    action = self.lowest_utilization_first()
                elif control == "wrr":
                    action = self.weighted_round_robin()
                elif control == "rbn":
                    action = self.random_best_n(n=3)
                # --- END NEW ALGORITHMS ---
                
                task_to_be_scheduled = self.env.unwrapped.current_qtask
                obs, reward, terminated, done, info = self.env.step(action)
                
                if reward < 0: 
                    self.greedy_index += 1
                if reward > 0:
                    self.greedy_index = 0
                    arr_temp["total_completion_time"] += task_to_be_scheduled.waiting_time + task_to_be_scheduled.execution_time
                    arr_temp["rescheduling_count"] += task_to_be_scheduled.rescheduling_count

            sys.stdout.write("\033[F\033[K")
            print(f"[{control}] progress: {episode_num + 1}/{self.num_episodes}")
            self.env.unwrapped.qsp_env.run()
            self.results.append(arr_temp)
            
            if episode_num == self.num_episodes - 1:
                self._collect_and_summarize_metrics(control)

        self._save_to_csv(control)
        
    def greedy(self, greedy_index):
        greedy_strategy = sorted(self.env.unwrapped.qnodes, key=lambda x: x.next_available_time)
        safe_index = min(greedy_index, len(greedy_strategy) - 1)
        return self.env.unwrapped.qnodes.index(greedy_strategy[safe_index])

    def random(self):
        return self.env.action_space.sample()

    def round_robin(self):
        action = self.rr_index % self.env.unwrapped.n_qnodes
        self.rr_index += 1
        return action

    def greedy_error(self, greedy_index, g_error="Readout_assignment_error"):
        greedy_strategy = sorted(self.env.unwrapped.qnodes, key=lambda x: (x.next_available_time, x.error[g_error]))
        safe_index = min(greedy_index, len(greedy_strategy) - 1)
        return self.env.unwrapped.qnodes.index(greedy_strategy[safe_index])

    def earliest_completion_time(self):
        qtask = self.env.unwrapped.current_qtask
        qnodes = self.env.unwrapped.qnodes
        broker = self.env.unwrapped.broker
        completion_times = []
        for node in qnodes:
            processed_task, waiting_time, execution_time = broker.preprocess_qtask(qtask, node)
            if processed_task.status == TaskStatus.ERROR:
                completion_times.append(float('inf'))
            else:
                est_completion_time = self.env.unwrapped.qsp_env.now + waiting_time + execution_time
                completion_times.append(est_completion_time)
        if all(ct == float('inf') for ct in completion_times):
            return self.random()
        return np.argmin(completion_times)

    def shortest_job_first(self):
        qtask = self.env.unwrapped.current_qtask
        qnodes = self.env.unwrapped.qnodes
        broker = self.env.unwrapped.broker
        execution_times = []
        for node in qnodes:
            processed_task, _, execution_time = broker.preprocess_qtask(qtask, node)
            if processed_task.status == TaskStatus.ERROR:
                execution_times.append(float('inf'))
            else:
                execution_times.append(execution_time)
        if all(et == float('inf') for et in execution_times):
            return self.random()
        return np.argmin(execution_times)

    def lowest_utilization_first(self):
        """
        Assigns the task to the node that has been historically the least busy.
        Utilization = Total Busy Time of a Node / Total Simulation Time
        """
        qnodes = self.env.unwrapped.qnodes
        # We use getattr to safely get total_busy_time, defaulting to 0 if not present
        busy_times = [getattr(node, 'total_busy_time', 0.0) for node in qnodes]
        
        # Check for invalid tasks on each node
        valid_nodes = []
        for i, node in enumerate(qnodes):
            # A simple validity check without full preprocessing
            if self.env.unwrapped.current_qtask.qubit_number <= node.qubit_number:
                valid_nodes.append(i)
        
        if not valid_nodes:
            return self.random() # All nodes are invalid for this task

        # Find the node with the minimum busy time *among the valid ones*
        min_busy_time = float('inf')
        action = -1
        for i in valid_nodes:
            if busy_times[i] < min_busy_time:
                min_busy_time = busy_times[i]
                action = i

        return action

    def weighted_round_robin(self):
        """
        Assigns tasks in a round-robin fashion but gives more time slots
        to more powerful nodes (based on CLOPS).
        """
        if not self.wrr_list: # Failsafe if list is empty
            return self.random()
        
        action = self.wrr_list[self.wrr_index % len(self.wrr_list)]
        self.wrr_index += 1
        return action

    def random_best_n(self, n=3):
        """
        A hybrid approximation algorithm. Randomly selects N nodes and then
        applies the optimal ECT strategy to that subset.
        """
        qnodes = self.env.unwrapped.qnodes
        num_qnodes = len(qnodes)
        
        # Create a list of node indices [0, 1, 2, 3, 4]
        node_indices = list(range(num_qnodes))
        
        # If N is larger than available nodes, just use all nodes
        n = min(n, num_qnodes)
        
        # Randomly sample N unique indices
        sampled_indices = random.sample(node_indices, n)
        
        # Now, apply the ECT logic only on the sampled nodes
        qtask = self.env.unwrapped.current_qtask
        broker = self.env.unwrapped.broker
        
        best_completion_time = float('inf')
        action = -1
        
        for index in sampled_indices:
            node = qnodes[index]
            processed_task, waiting_time, execution_time = broker.preprocess_qtask(qtask, node)
            
            if processed_task.status != TaskStatus.ERROR:
                est_completion_time = self.env.unwrapped.qsp_env.now + waiting_time + execution_time
                if est_completion_time < best_completion_time:
                    best_completion_time = est_completion_time
                    action = index
        
        # If no valid node was found in the sample, fall back to random
        return action if action != -1 else self.random()

    def _save_to_csv(self, control) -> None:
        file_name = "./results/heuristics/" 
        if not os.path.exists(file_name): os.makedirs(file_name)
        file_name += control + ".csv"
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total Completion Time', 'Rescheduling Count'])
            for i in range(len(self.results)):
                writer.writerow([i, self.results[i]['total_completion_time'], self.results[i]['rescheduling_count']])
        print("CSV file saved to " + file_name)

    def _plot_results(self, paths) -> None:
        plt.figure(figsize=(14, 8))
        for path in paths:
            df1 = pd.read_csv(path['path'])
            plt.plot(df1['Episode'], df1['Total Completion Time'], ".-", color=path['color'], label=path['label'])
            self._summarize_results(df1, path['label'])
        plt.ylabel('Total Completion Time')
        plt.xlabel('Evaluation Episode')
        plt.title('Heuristic Method Performance Comparison')
        plt.legend(loc='best')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
        if not os.path.exists("./results/heuristics/"): os.makedirs("./results/heuristics/")
        plt.savefig("./results/heuristics/performance_comparison.png")
        plt.show()

    def _summarize_results(self, values, label) -> None:
        print(f"\n--- Results Summary for {label} solution ---")
        print(f"Number of Episodes: {self.num_episodes}")
        print(f"Average Total Completion Time: {sum(values['Total Completion Time']) / self.num_episodes:.2f}")
        print(f"Average Rescheduling Count: {sum(values['Rescheduling Count']) / self.num_episodes:.2f}")
        print("------------------------------------------")
    
    def _collect_and_summarize_metrics(self, control):
        serviced_tasks = self.env.unwrapped.serviced_qtasks
        serviced_tasks = self.env
        qnodes = self.env.unwrapped.qnodes
        total_sim_time = self.env.unwrapped.qsp_env.now
        if not serviced_tasks:
            print(f"\n--- No tasks were serviced for {control}. Cannot generate metrics. ---")
            return
        total_qtasks = len(serviced_tasks)
        avg_wait_time = sum(task.waiting_time for task in serviced_tasks) / total_qtasks
        throughput = total_qtasks / total_sim_time if total_sim_time > 0 else 0
        failure_threshold = 5
        failed_tasks = sum(1 for task in serviced_tasks if task.rescheduling_count >= failure_threshold)
        success_rate = (total_qtasks - failed_tasks) / total_qtasks * 100
        print(f"\n--- Detailed Metrics Summary for '{control}' ---")
        print(f"Total Simulation Time: {total_sim_time:.2f}s")
        print(f"Throughput: {throughput:.4f} tasks/sec")
        print(f"Average Wait Time per Task: {avg_wait_time:.4f}s")
        print(f"Success Rate (rescheduled < {failure_threshold} times): {success_rate:.2f}%")
        print(f"Number of Failed Tasks: {failed_tasks}")
        print("\n--- Quantum Computer Utilization ---")
        for node in qnodes:
            busy_time = getattr(node, 'total_busy_time', 0.0)
            utilization = (busy_time / total_sim_time) * 100 if total_sim_time > 0 else 0
            print(f"  - {node.name} (ID: {node.id}): {utilization:.2f}% utilization")
        print("------------------------------------------")


if __name__ == "__main__":
    env_config = {
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
    }
    env = qsimpy_env_creator(env_config)
    
    print("\n\n\n")
    
    heuristics = HeuristicSolutions(env, num_episodes=100)
    heuristics.run("sjf")
    # methods = [
    #     'greedy', 'random', 'round_robin', 'greedy_error', 
    #     'ect', 'sjf', 'luf', 'wrr', 'rbn'
    # ]


    # processes = [Process(target=heuristics.run, args=(m,)) for m in methods]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()
    print("All Processes are done!!")

    # --- UPDATE PATHS FOR PLOTTING ---
    paths = [
        {"label": "Random", "path": "./results/heuristics/random.csv", "color": "red"},
        {"label": "Round Robin", "path": "./results/heuristics/round_robin.csv", "color": "blue"},
        {"label": "Greedy (Earliest Available)", "path": "./results/heuristics/greedy.csv", "color": "black"},
        {"label": "Greedy + Error", "path": "./results/heuristics/greedy_error.csv", "color": "green"},
        {"label": "Earliest Completion Time (ECT)", "path": "./results/heuristics/ect.csv", "color": "purple"},
        {"label": "Shortest Job First (SJF)", "path": "./results/heuristics/sjf.csv", "color": "orange"},
        {"label": "Lowest Utilization First (LUF)", "path": "./results/heuristics/luf.csv", "color": "cyan"},
        {"label": "Weighted Round Robin (WRR)", "path": "./results/heuristics/wrr.csv", "color": "magenta"},
        {"label": "Random Best-3 (RBN)", "path": "./results/heuristics/rbn.csv", "color": "brown"},
    ]
    # --- END UPDATE ---
    heuristics._plot_results(paths)