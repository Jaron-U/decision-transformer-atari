import atari_py
import cv2
import matplotlib.pyplot as plt

env = atari_py.ALEInterface()
env.loadROM(atari_py.get_game_path("pong"))

env.reset_game()

state = cv2.resize(env.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
print(state.shape)

plt.imshow(state)
plt.savefig('pong.png')

available_actions = env.getMinimalActionSet()
action_dict = dict([i, e] for i, e in enumerate(available_actions))
print(action_dict)

action = 0 
total = 0

while True:
    reward = env.act(action_dict.get(action))  # here we always post 0 (NOOP)
    if reward > 0.0:
        print(f"You won ({reward})")
    elif reward < 0.0:
        print(f"You lose ({reward})")
    total += reward

    if env.game_over():
        break

print(f"The episode has ended. (Total {total})")