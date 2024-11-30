import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


from gymnasium  import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

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
        # Reset the robot and environment
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.5], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.ball_id, [15, 0, 0.5], [0, 0, 0, 1])
        
        self.step_counter = 0
        obs = self._get_observation()
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
    
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from tqdm import tqdm

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
    
    
    
    
    
env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])  # Evaluation environment

### Define total timesteps
total_timesteps = 500000
##
##    # Create the callback with progress bar and logging
progress_and_eval_callback = ProgressAndEvalCallback(
    total_timesteps=total_timesteps,
    eval_env=eval_env,
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=3,
    deterministic=True,
    render=False,
    reward_log_path="./logs/reward_log.txt"
)
# Define and train the model
#model = PPO("MlpPolicy", env, verbose=1) #Start with this for training.
model = PPO.load("./logs/best_model", env=env) #Use this for the next iteration of training
#
model.learn(total_timesteps=total_timesteps, callback=progress_and_eval_callback)
#
#   # Plot the rewards
plt.plot(progress_and_eval_callback.episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.show()
	

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