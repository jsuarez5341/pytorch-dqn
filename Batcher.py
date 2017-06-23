from pdb import set_trace as T
import torch as t
import numpy as np
import gym

from scipy.misc import imresize

def rgb2gray(img):
   return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def preprocess(img):
   img = rgb2gray(img)
   img = imresize(img, (110, 84), interp='bicubic')/255.0
   #img = img[13:-13, :]/255.0
   return img.astype(np.float32)

class Emulator:

   def __init__(self, game):
      self.env = gym.make(game)
      self.numActions = self.env.action_space.n

      self.env.reset()

   def next(self, action, k=4, render=False):
      states = []
      reward = 0.0
      for i in range(k):
         state, r, done, _ = self.env.step(action)
         states += [state]
         reward += r
         if done:
            self.env.reset()
            break 

      if r > 0:
         r = 1
      elif r < 0:
         r = -1

      if render:
         self.env.render()

      states += (k - len(states))*[states[0]]
      states = [preprocess(s) for s in states]
      states = np.stack(states, 0).astype(np.float32)
      return states, reward, done

