import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt  
import cv2
import os

import matplotlib.pyplot as plt

from gymnasium  import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from tqdm import tqdm  


class QuadrupedEnv(gym.Env):
    def __init__(self):
        super(QuadrupedEnv, self).__init__()
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.DIRECT) # # Use this for training
        #self.physics_client = p.connect(p.GUI) # Use this for GUI version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load robot and environment
        robots = p.loadMJCF("../ant.xml")
        print(f"QuadrupedEnv, robots= {robots}")
        self.robot_id = robots[1]
        
        num_legs=p.getNumJoints(self.robot_id)
        print(f"QuadrupedEnv, num_legs= {num_legs}")


        self.ball_id = p.loadURDF("sphere2.urdf", [15, 0, 0.5], useFixedBase=True, globalScaling=0.3)
        p.changeVisualShape(self.ball_id, -1, rgbaColor=[1, 0, 0, 1])  # Red color

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_legs,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12+1,), dtype=np.float32)#4,), dtype=np.float32)

        self.max_steps = 15000
        self.step_counter = 0
		
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
    
        # Compute distance to the ball
        distance_to_ball = np.linalg.norm(np.array(robot_pos[:2]) - np.array(ball_pos[:2]))
        self.previous_distance = distance_to_ball
        
    def reset(self, *, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)  # Ensure reproducibility
        
        # Reset the robot and environment
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.5], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.ball_id, [15, 0, 0.5], [0, 0, 0, 1])
        
        self.step_counter = 0
        obs = self._get_observation()
        # Reset the previous distance
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        self.previous_distance = np.linalg.norm(np.array(robot_pos[:2]) - np.array(ball_pos[:2]))

        return obs, {} # Fix return signature

    def step(self, action):
        # Apply action to robot's motors (simplified parametric CPG)
        for joint in range(p.getNumJoints(self.robot_id)):  
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id, jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition= action[joint] 
            )

        p.stepSimulation()
        self.step_counter += 1

        # Reward function
        reward = self._compute_reward()
        
        done = self.step_counter >= self.max_steps
        obs = self._get_observation()

        # Truncated is typically used to handle environments that terminate early
        # In this case, it can be the same as done
        truncated = False  # You can implement logic to truncate if needed (e.g., timeout or failure)
		
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        distance_to_ball = np.linalg.norm(np.array(robot_pos[:2]) - np.array(ball_pos[:2]))
                # Penalize for flipping
        up_vector = p.getMatrixFromQuaternion(robot_orn)[6:9]
        if up_vector[2] < 0 :
            truncated = True
            done=True
            reward -=2000
            print ("FLIPPED. distance_to_ball=", distance_to_ball, ", revard=", reward)
        elif distance_to_ball < 1 :
            truncated = True
            done=True
            reward +=1000
            print ("SUCCESS : Got too close and terminated. distance_to_ball=", distance_to_ball, ", revard=", reward)
            
        return obs, reward, done, truncated, {}

    def _get_observation(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        obs = np.concatenate([pos, ori, linear_velocity, angular_velocity]).astype(np.float32)
        #print(f"_get_observation, obs= {obs}")
        return obs

    
    def _compute_reward(self):
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
    
        # Compute distance to the ball
        distance_to_ball = np.linalg.norm(np.array(robot_pos[:2]) - np.array(ball_pos[:2]))
    
        # Initialize reward
        reward = 0
        
        # Strong penalty for flipping - this should be the primary concern
        up_vector = p.getMatrixFromQuaternion(robot_orn)[6:9]
        if up_vector[2] < 0.7:  # Penalize more aggressively when tilting too much
            reward -= 500 * (0.7 - up_vector[2])  # Progressive penalty based on tilt
            if up_vector[2] < 0:  # Complete flip
                reward -= 1000
                
        # Stability rewards
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        stability_reward = -np.linalg.norm(angular_velocity) * 50  # Increased penalty for rotation
        reward += stability_reward
        
        # Height check to encourage staying upright
        height = robot_pos[2]
        ideal_height = 0.5  # Adjust based on your robot's ideal standing height
        height_penalty = -abs(height - ideal_height) * 50
        reward += height_penalty
        
        # Distance and movement rewards - only if robot is stable
        if up_vector[2] > 0.7:  # Only reward movement when relatively upright
            # Base distance reward
            reward -= distance_to_ball * 2
            
            # Directional movement reward
            direction_to_ball = np.array(ball_pos[:2]) - np.array(robot_pos[:2])
            if np.linalg.norm(direction_to_ball) > 0:
                direction_to_ball /= np.linalg.norm(direction_to_ball)
                velocity_towards_ball = np.dot(np.array(linear_velocity[:2]), direction_to_ball)
                reward += velocity_towards_ball * 10
            
            # Progressive rewards for getting closer
            if distance_to_ball < 1:
                reward += 100
            elif distance_to_ball < 3:
                reward += 50
            elif distance_to_ball < 5:
                reward += 25
                
            # Distance improvement reward
            distance_improvement = self.previous_distance - distance_to_ball
            reward += distance_improvement * 20
            
        # Update previous distance for next step
        self.previous_distance = distance_to_ball
        
        return reward

    def _get_joint_forces(self):
        forces = [p.getJointState(self.robot_id, i)[3] for i in range(p.getNumJoints(self.robot_id))]
        return forces

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
		
		
def make_env():
    env = QuadrupedEnv()
    return Monitor(env)
    

class ProgressAndEvalCallback(EvalCallback):
    def __init__(self, total_timesteps, eval_env, *args, reward_log_path=None, **kwargs):
        super(ProgressAndEvalCallback, self).__init__(eval_env, *args, **kwargs)
        self.total_timesteps = total_timesteps
        self.current_timestep = 0
        self.episode_rewards = []
        self.reward_log_path = reward_log_path
        self.progress_bar = None
        self.eval_env=eval_env
        self.current_episode_reward = 0

    def _on_training_start(self):
        # Initialize the progress bar
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        # Update the progress bar
        self.current_timestep += 1
        self.progress_bar.update(1)
    
        # Accumulate rewards manually
        self.current_episode_reward += self.locals["rewards"][0]
		
        #print("self.locals=", self.locals)
        # Check if the episode ends
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Reset for the next episode
            print("DONE")
			
        # Log episode rewards if available in infos
        if self.locals.get("infos") is not None:
            for info in self.locals["infos"]:
                if "episode" in info:  # Check if the 'episode' key exists
                    episode_reward = info["episode"]["r"]
                    self.episode_rewards.append(episode_reward)
        # Check if we've reached the total timesteps
        if self.current_timestep >= self.total_timesteps:
            self.model.env.close()  # Close the environment
            print("episode_rewards=", self.episode_rewards)
            if self.reward_log_path and self.episode_rewards:
                with open(self.reward_log_path, "w") as f:
                    for reward in self.episode_rewards:
                        f.write(f"{reward}\n")
            return False  # Stop training

        return super(ProgressAndEvalCallback, self)._on_step()
    
    
    def _on_training_end(self):
        # Close the progress bar
        self.progress_bar.close()
        print("Training Ended. episode_rewards=", self.episode_rewards)
        # Save rewards to a file if specified
        if self.reward_log_path and self.episode_rewards:
            with open(self.reward_log_path, "w") as f:
                for reward in self.episode_rewards:
                    f.write(f"{reward}\n")
    


class RecordFastestRobotCallback(EvalCallback):
    def __init__(self, render_env, total_timesteps, *args, video_path="fastest_robot.mp4", **kwargs):
        super().__init__(*args, **kwargs)
        self.render_env = render_env
        self.fastest_speed = 0
        self.fastest_episode_frames = []
        self.video_path = video_path
        self.progress_bar = None    
        self.total_timesteps = total_timesteps
        self.current_timestep = 0  # Add this l


    def _on_training_start(self):
        # Initialize the progress bar
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress")


    def _on_step(self) -> bool:
        
        # Update the progress bar
        self.current_timestep += 1
        self.progress_bar.update(1)

        # Evaluate speed and save frames if this is the fastest episode
        speed = self.locals["infos"][0].get("speed", 0)
        if speed > self.fastest_speed:
            self.fastest_speed = speed
            self.fastest_episode_frames = []
            self._record_episode()

        return super()._on_step()

    def _record_episode(self):
        # Render frames for the fastest episode
        obs = self.render_env.reset()
        done = False
        while not done:
            frame = self.render_env.render(mode="rgb_array")
            self.fastest_episode_frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.render_env.step(action)

    def _on_training_end(self):
        
        # Close the progress bar
        self.progress_bar.close()

        # Save the video of the fastest episode
        if self.fastest_episode_frames:
            height, width, _ = self.fastest_episode_frames[0].shape
            out = cv2.VideoWriter(
                self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
            )
            for frame in self.fastest_episode_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
        print(f"Fastest episode video saved to {self.video_path}")



class SpeedTrackingCallback(EvalCallback):
    def __init__(self, *args, log_path="speed_log.txt", **kwargs):
        super().__init__(*args, **kwargs)
        self.speeds = []
        self.log_path = log_path
        self.previous_position = None

    def _on_step(self) -> bool:
        # Get robot position from the environment
        robot_pos, _ = p.getBasePositionAndOrientation(self.eval_env.envs[0].robot_id)
        current_position = np.array(robot_pos)

        if self.previous_position is None:
            self.previous_position = current_position
            return super()._on_step()

        # Calculate speed (assuming time_step of 1/240 seconds, which is PyBullet's default)
        time_step = 1/240
        speed = np.linalg.norm(current_position - self.previous_position) / time_step
        self.speeds.append(speed)
        self.previous_position = current_position

        # Log speed
        with open(self.log_path, "a") as log_file:
            log_file.write(f"{speed}\n")

        return super()._on_step()

    def _on_training_end(self):
        # Save speed logs to file
        with open(self.log_path, "w") as f:
            for speed in self.speeds:
                f.write(f"{speed}\n")
        print(f"Speed log saved to {self.log_path}")

        # Plot learning curve
        self.plot_learning_curve()

    def plot_learning_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.speeds, label="Speed (cm/s)")
        plt.xlabel("Trials")
        plt.ylabel("Speed (cm/s)")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid()
        plt.savefig("learning_curve.png")
        plt.show()


def plot_aggregated_learning_curves(log_files):
    all_speeds = [np.loadtxt(log_file) for log_file in log_files]
    mean_speeds = np.mean(all_speeds, axis=0)
    std_speeds = np.std(all_speeds, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_speeds, label="Average Speed (cm/s)")
    plt.fill_between(range(len(mean_speeds)),
                     mean_speeds - std_speeds,
                     mean_speeds + std_speeds,
                     alpha=0.3, label="Standard Deviation")
    plt.xlabel("Trials")
    plt.ylabel("Speed (cm/s)")
    plt.title("Aggregated Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig("aggregated_learning_curve.png")
    plt.show()

log_path = "./logs/speed_logs/"  # Specify a proper directory

# Wrap environments
env = DummyVecEnv([make_env])
render_env = DummyVecEnv([make_env])

# Define total timesteps
total_timesteps = 500000

# Create callbacks
record_callback = RecordFastestRobotCallback(total_timesteps=total_timesteps, render_env=render_env,eval_env=env,
    video_path="./logs/fastest_robot.mp4")
speed_callback = SpeedTrackingCallback(eval_env=env,
    log_path="./logs/speed_log.txt")
    
### Define total timesteps
#total_timesteps = 100000

# Define and train the model
#model = PPO("MlpPolicy", env, verbose=1) #Start with this for training.
model = PPO.load("./logs/best_model", env=env) #Use this for the next iteration of training

model.learn(total_timesteps=total_timesteps, callback=[record_callback, speed_callback])

log_files = ["./logs/speed_log.txt"]  # Use only the file we actually created
if os.path.exists(log_files[0]):  # Add error handling
    plot_aggregated_learning_curves(log_files)
else:
    print(f"Warning: {log_files[0]} not found. Skipping plot generation.")


# Use fllowing to load the best model and test it
del model
model = PPO.load("./logs/best_model", env=env)

#### Test the trained model
rewards = []
for i in range(100):
    obs = env.reset()
    total_reward = 0
    for _ in range(env.envs[0].max_steps):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    rewards.append(total_reward)

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()