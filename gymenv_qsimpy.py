"""
QSimPy 0.1.0 - Gymnasium Environment
Author: Hoa Nguyen
Sample Gymasium environment for QSimPy with the scenario of 5 IBM QNodes (washington, kolkata, hanoi, perth, lagos) 
and continuous QTasks arrival, generated from a quantum circuit dataset.
"""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from numpy.random import default_rng
from qsimpy import Broker, QTask, TaskStatus, IBMQNode, Dataset, Log
import simpy


class QSimPyEnv(gym.Env):
    MAX_ROUNDS = 999  # maximum number of rounds in the QTask dataset

    """
    Gym environment for QSimPy.

    Args:
        config (dict): Configuration parameters for the environment.
        dataset (str): Path to the QTask dataset.
        mode (str): Selection for simulation/emulation mode.

    Attributes:
        n_qtasks (int): Number of qtasks.
        n_qnodes (int): Number of qnodes.
        qtasks (list): List of qtasks.
        qnodes (list): List of qnodes.
        obs_dim (int): Dimension of the observation space.
        observation_space (gym.spaces.Box): Observation space.
        action_space (gym.spaces.Discrete): Action space.
        qtask_dataset (Dataset): QTasks dataset.
        rng (numpy.random.Generator): Random number generator.
        qsp_env (simpy.Environment): QSimPy environment.
        round (int): Current round.
        seed (int): Seed for the random number generator.
    """

    def __init__(
        self,
        config=None,
        dataset=None,
        mode="simulate"
    ):
        """
        Initialize the QSimPyEnv class.

        Args:
            config (dict): Configuration parameters for the environment.
            dataset (str): Path to the dataset.
        """
        
        super().__init__()
        if dataset is None:
            raise ValueError("Dataset is not specified")
        
        self.dataset_path = dataset

        # OBSERVATION SPACE
        # Each observation is a dict of qtask_attributes and qnode_attributes
        # QTask attrributes = [arrivaltime, qt_qubits, cl]
        # QNode attributes = [qn_qubits, d1cps, next_available_time]
        self.n_qtasks = 25
        self.n_qnodes = 5  # number of qnodes
        self.qtasks = []
        self.qnodes = []
        self.serviced_qtasks = []
        self.mode = mode
        self.obs_dim = 4 + self.n_qnodes * 3
        self.observation_space = Box(
            low=np.ones((self.obs_dim,), dtype=np.float32) * -np.inf,
            high=np.ones((self.obs_dim,), dtype=np.float32) * np.inf,
            dtype=np.float32,
        )
        # Example max values for each type of observation
        max_time = 10000000  # Max next available time
        max_qubits = 500  # Max number of qubits
        max_layers = 1000000  # Max number of layers in a circuit
        max_clops = 10000  # Max computational load
        max_rescheduling_count = 1000  # Max number of rescheduling

        # Assuming the observation consists of [arrival_time, qubit_number, circuit_layers] for tasks
        # and [qubit_number, clops, next_available_time] for each node
        task_obs_low = np.array([0, 0, 0, 0], dtype=np.float32)
        task_obs_high = np.array(
            [max_time, max_qubits, max_layers, max_rescheduling_count], dtype=np.float32
        )
        node_obs_low = np.array([0, 0, -1] * self.n_qnodes, dtype=np.float32)
        node_obs_high = np.array(
            [max_qubits, max_clops, max_time] * self.n_qnodes, dtype=np.float32
        )

        # Combine to form the complete observation space
        obs_low = np.concatenate([task_obs_low, node_obs_low]).astype(np.float32)
        obs_high = np.concatenate([task_obs_high, node_obs_high]).astype(np.float32)

        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.current_obs = None

        # ACTION SPACE
        self.action_space = Discrete(self.n_qnodes)

        # Load QTasks dataset
        self.qtask_dataset = Dataset(self.dataset_path)
        self.seed = 22
        self.rng = default_rng(seed=self.seed)
        
        # QSimPy environment
        self.qsp_env = simpy.Environment()
        self.setup_quantum_resources()

        # Round
        self.round = 1
        self.round_robin_index = 0
        self.results = [] 
        
        # Set rescheduling time, default = 0.01s
        self.rescheduling_time = 0.01

        # Check if evaluation is set
        if config is None:
            config = {}
        self.evaluation = config.get("evaluation", False)
        self.policy = config.get("policy", "UnknownPolicy")
        

    def __getstate__(self):
        """
        This method prepares the object for serialization (pickling).
        We remove all objects that cannot be pickled.
        """
        state = self.__dict__.copy()
        
        # List of attributes to remove before pickling
        unpicklable_attributes = [
            'qsp_env', 
            'broker', 
            'qnodes', 
            'rng', 
            'qtask_dataset'
        ]
        
        for attr in unpicklable_attributes:
            if attr in state:
                del state[attr]
                
        return state

    def __setstate__(self, state):
        """
        This method restores the object after serialization.
        We re-initialize the objects that we removed in __getstate__.
        """
        self.__dict__.update(state)
        
        # Re-initialize the unpicklable attributes
        self.qtask_dataset = Dataset(self.dataset_path)
        self.rng = default_rng(seed=self.seed)
        self.qsp_env = simpy.Environment()
        self.setup_quantum_resources()
    
    def _get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
            np.ndarray: The current observation, which includes the observation of the current quantum task and all quantum nodes.
        """
        # Get the current observation of quantum task
        if self.current_qtask is None:
            self.qtask_obs = np.array([0, 0, 0, 0], dtype=np.float32)
        else:
            self.qtask_obs = np.array(
                [
                    self.current_qtask.arrival_time,
                    self.current_qtask.qubit_number,
                    self.current_qtask.circuit_layers,
                    self.current_qtask.rescheduling_count,
                ],
                dtype=np.float32,
            )

        # Get the current observation of quantum nodes
        self.qnode_obs = []
        for qnode in self.qnodes:
            qnode_obs = np.array(
                [
                    qnode.qubit_number,
                    qnode.clops,
                    qnode.next_available_time,
                ],
                dtype=np.float32,
            )
            self.qnode_obs.append(qnode_obs)

        # Flatten the qnode observations and concatenate with qtask observations
        qnode_obs_flat = np.concatenate(self.qnode_obs).astype(np.float32)
        self.current_obs = np.concatenate(
            (self.qtask_obs, qnode_obs_flat), dtype=np.float32
        )
        return self.current_obs

    def setup_quantum_resources(self):
        # Create a list of 10 IBM QNodes
        qnode_ids = range(self.n_qnodes)
        qnode_names = [
            "washington",
            "kolkata",
            "hanoi",
            "perth",
            "lagos",
        ]
        self.qnodes = [
            IBMQNode.create_ibmq_node(self.qsp_env, qid, qname)
            for qid, qname in zip(qnode_ids, qnode_names)
        ]

        # Manually assign the name attribute to each node after creation
        for node, name in zip(self.qnodes, qnode_names):
            node.name = name
        
        # Create a Broker
        self.broker = Broker(self.qsp_env, self.qnodes, self.mode)

    def generate_qtasks(self):
        """Generate a list of QTasks from the QTask dataset, following Poisson distribution of arrival time."""
        # QTask IDs
        qtask_ids = list(
            f"{self.round:04d}" + f"{id:02d}" for id in range(self.n_qtasks)
        )

        # Get QTask from the subset of the dataset
        qtasks = self.qtask_dataset.get_subset_data(self.round)
        
        # Option 1: Random arrival time of qtasks (original implementation)
        n_qtasks = len(qtasks)  # number of qtasks
        qtask_arrival = self.rng.uniform(
            low=0.1 + self.round * 60, high=59.9 + self.round * 60, size=n_qtasks
        )
        # Set the arrival time of the first qtask to 0
        qtask_arrival.sort()
        
        # Option 2: Get arrival time from the datasets
        # qtask_arrival = [qtask["arrival_time"] for qtask in qtasks.values()]


        # Extract only the values (dictionaries) from qtasks
        qtask_values = list(qtasks.values())

        self.qtasks = [
            QTask(
                id=tid,
                arrival_time=arrival_time,
                qtask_data=qdata,
            )
            for tid, arrival_time, qdata in zip(qtask_ids, qtask_arrival, qtask_values)
        ]

        # Set current qtask to the first qtask in the list
        if len(self.qtasks) > 0:
            self.current_qtask = self.qtasks.pop(0)
        self.prev_qtask = None
        self.round += 1

    def submit_task_to_qnode(self, qtask, qnode_id=None):
        if qnode_id is None:
            qnode_id = self.round_robin_index % self.n_qnodes
            self.round_robin_index += 1

        # Check for task validity against the chosen node
        qtask, waiting_time, execution_time = self.broker.preprocess_qtask(
            qtask, self.qnodes[qnode_id]
        )

        if qtask.status == TaskStatus.ERROR:
            # Handle infeasible scheduling
            qtask.status = TaskStatus.QUEUED
            qtask.QNode = None
            qtask.rescheduling_count += 1
            qtask.arrival_time = self.qsp_env.now + self.rescheduling_time
            
            index = 0
            while (
                index < len(self.qtasks)
                and self.qtasks[index].arrival_time < qtask.arrival_time
            ):
                index += 1
            self.qtasks.insert(index, qtask)
            
            self.qsp_env.run(until=self.qsp_env.now + self.rescheduling_time)
            return -0.1, qtask.rescheduling_count
        
        # If valid, schedule the task
        qtask_execution = self.broker.submit_qtask_to_qnode(
            qtask, self.qnodes[qnode_id]
        )
        self.qsp_env.process(qtask_execution)
        
        # --- START OF FIX ---
        # Calculate all timing info BEFORE running the simulation
        
        # The task will start after the current time + its waiting time.
        start_time = self.qsp_env.now + waiting_time
        
        # We can store this on the object now for logging purposes.
        qtask.start_time = start_time
        
        # Calculate the exact time the simulation needs to run until.
        completion_time = start_time + execution_time
        
        # Calculate the delay from the task's very first arrival.
        delay_time = start_time - qtask.init_arrival_time
        
        # --- END OF FIX ---

        # NOW, run the simulation until this specific task is done
        self.qsp_env.run(until=completion_time)
        
        self.results.append({
            'qtask_id': qtask.id,
            'qnode_id': qnode_id,
            'waiting_time': waiting_time,
            'execution_time': execution_time,
            'rescheduling_count': qtask.rescheduling_count,
        })
        
        self.serviced_qtasks.append(qtask)
        
        # The reward is based on the total time spent by the task in the system
        total_time_in_system = delay_time + waiting_time + execution_time

        return total_time_in_system, qtask.rescheduling_count

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
        
        # Re-initialize the entire simulation environment
        self.qsp_env = simpy.Environment()
        self.rng = default_rng(seed=self.seed)
        self.setup_quantum_resources()
        self.results = []
        self.round = 1
        self.serviced_qtasks = [] 
        
        self.generate_qtasks()
        self.current_obs = self._get_obs().astype(np.float32)
        info = {}
        return self.current_obs, info
    
    def collect_results(self):
        Log.print_simulation_results(self.qnodes)
        """Collect and summarize results from the QTask executions."""
        total_qtasks = len(self.results)
        total_waiting_time = sum(result['waiting_time'] for result in self.results)
        total_execution_time = sum(result['execution_time'] for result in self.results)
        total_rescheduling_count = sum(result['rescheduling_count'] for result in self.results)

        if total_qtasks > 0:
            avg_waiting_time = total_waiting_time / total_qtasks
            avg_execution_time = total_execution_time / total_qtasks
            avg_rescheduling_count = total_rescheduling_count / total_qtasks
        else:
            avg_waiting_time = avg_execution_time = avg_rescheduling_count = 0

        summary = {
            'total_qtasks': total_qtasks,
            'total_waiting_time': total_waiting_time,
            'total_execution_time': total_execution_time,
            'avg_waiting_time': avg_waiting_time,
            'avg_execution_time': avg_execution_time,
            'avg_rescheduling_count': avg_rescheduling_count,
        }
        return summary

    def step(self, action):
        # Submit the current qtask to the selected qnode and run the simulation
        # The returned `time_reward` is the total time the task spent in the system
        time_reward, _ = self.submit_task_to_qnode(
            self.current_qtask, action
        )

        # A reward of -0.1 indicates a penalty for an invalid action
        if time_reward == -0.1:
            reward = -0.1
        else:
            # The objective is to MINIMIZE completion time, so the reward should be inverse
            # Add a small epsilon to avoid division by zero
            reward = 1.0 / (time_reward + 1e-6)

        scheduled_qtask = self.current_qtask

        # Get the next observation
        if len(self.qtasks) > 0:
            self.current_qtask = self.qtasks.pop(0)
            # Advance simulation time to the arrival of the next task if needed
            if self.qsp_env.now < self.current_qtask.arrival_time:
                self.qsp_env.run(until=self.current_qtask.arrival_time)
            terminated = False
        else:
            self.current_qtask = None
            terminated = True

        self.current_obs = self._get_obs()
        info = {"scheduled_qtask_id": scheduled_qtask.id}
        return self.current_obs, reward, terminated, False, info

    def close(self):
        # If the evaluation is set, run the environment and export the results
        if self.evaluation:
            Log.log = True
            self.qsp_env.run()
            Log.print_simulation_results(self.qnodes)
            self.collect_results()
            # Log.export_simulation_results(self.qnodes, output_file=self.policy)
        pass
