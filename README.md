# DQN_deadcells
A DQN model to fight with the boss in dead cells (default The Concierge).
Using double DDQN with a prioritized experience buffer



https://github.com/user-attachments/assets/b4ff9013-e671-46fb-b28b-3aae9bbb224a

# Environment
- windows11
- Python 3.9.21
- python library: see requirements.txt (Anaconda env "DQN": double-click SETUP_env.bat)
- Dead Cells
- CUDA and cudnn for pytorch
- Cheat Engine for 2x game speed

# usage
- the key bind is 1 for the weapon, 2 for the shield, shift for roll, space for jump, and arrow key for move.
  - It is important because I give the weapon and shield different rewards if you want to modify it in lib/action.py
  - In my own training, I used Panchaku as the weapon and Front Line Shield as the shield.
  - For the default of scrolls I pick 15 Brutality, 7 tactics, and 7 survival.
  - I choose Combo as the mutation.
  - I change the Accessibility setting
    - ![image](https://github.com/user-attachments/assets/f6c4c409-5b4d-472b-953e-85dec8a89813)
    - ![image](https://github.com/user-attachments/assets/ff573db9-1552-4f57-a261-ae64799a4896)
    - ![image](https://github.com/user-attachments/assets/56060497-9286-4380-b790-14cfd63a344d)
    - ![image](https://github.com/user-attachments/assets/4e5375dc-139c-4f83-9c06-0cfafaaf6d67)
    - ![image](https://github.com/user-attachments/assets/0012b9d7-b52b-49bc-914d-1fd0eb4082ad)





- please make sure there is a ./checkpoints folder under the main folder because I did not do the detection for that.
- if you want to train the new model, please open Cheat Engine and use the speed hack to make Dead Cell run 2x speed
- Then stand in front of this door![image](https://github.com/user-attachments/assets/b747ebca-0bfc-4a51-ba1f-26170a7cfdfb)
- run train.py
- I hard-coded the sleep time for the agent to restart. If there is any problem, please try to adjust the sleep time in the  env.py env._reset() function
- press q to quit training and press p to stop training. Remember to stop the game process by pressing tab or esc when pressing p.

# Code structure
- Most training configuration is in train.py
- Most hyperparameters are in run.py or train.py, near line 20.
- model.py is the Model architecture
- lib/Actions.py is for sending the Action into game
- lib/env.py is for restarting the game and getting the reward for action. If you want to modify the reward strategy, you can modify the env.step function
- lib/GetHp.py is for getting hp for the agent and boss
- lib/GetScreen.py is for getting the screen as the input of the agent
- lib/SendKey.py is for sending the key.

# todo
- restructure whole code structure. Make code more clear and clean

# Acknowledgement
- https://github.com/ailec0623/DQN_HollowKnight
- https://arxiv.org/pdf/1710.02298
