import numpy as np

# Create the evaluation environment
from env_creator import qsimpy_env_creator

eval_env = qsimpy_env_creator(
    env_config={
        "obs_filter": "rescale_-1_1",
        "reward_filter": None,
        "dataset": "qdataset/traintest/synds_200_sub_60_testset_with_sorted_arrival.csv",
        "config": {"evaluation": True, "policy": "greedy"},
    }
)

skipped_list = []


def enhanced_greedy_policy(env, skipped_list):
    current_obs = env.unwrapped.current_obs
    qnode_start_index = 4  # Adjust based on the actual qtask observation length
    qnode_obs_length = 3  # Number of values per qnode in the observation

    best_choice = None
    earliest_time = float("inf")

    # Iterate over each qnode observation to find the one with the earliest next available time
    for i in range(qnode_start_index, len(current_obs), qnode_obs_length):
        qnode_id = (i - qnode_start_index) // qnode_obs_length

        # Skip the qnode if it's in the skipped list
        if qnode_id in skipped_list:
            continue

        qnode_next_available_time = current_obs[i + 2]  # next_available_time of qnode

        # Update the best choice if this qnode's next available time is earlier
        if qnode_next_available_time < earliest_time:
            earliest_time = qnode_next_available_time
            best_choice = qnode_id

    return best_choice


# Run the evaluation
num_episodes = 1  # Set the number of episodes for evaluation

for episode in range(num_episodes):
    obs = eval_env.reset(seed=22)
    # print(f"Episode {episode} started with observation: {obs}")
    terminate = False
    episode_reward = 0

    while not terminate:
        action = enhanced_greedy_policy(eval_env, skipped_list)
        obs, reward, terminate, _, info = eval_env.step(action)
        episode_reward += reward

        # If the reward is -10, add the action (qnode_id) to the skipped list
        if reward <= -10:
            skipped_list.append(action)
        # print(
        #     f"Action: {action}, Reward: {reward}, Terminate: {terminate}, Skip list: {skipped_list}"
        # )
        if reward > -10:
            skipped_list.clear()

        if terminate:
            print(f"Episode {episode} finished with reward {episode_reward}")
            break

# Run and close the evaluation environment
eval_env.close()
