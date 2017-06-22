from pdb import set_trace as T
import torch as t
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
from Batcher import Emulator

import time
from collections import deque

#Variable wrapper
def var(xNp, volatile=False, cuda=False):
   x = Variable(t.from_numpy(xNp), volatile=volatile)
   if cuda:
      x = x.cuda()
   return x

class Net(nn.Module):

   def __init__(self, numActions):
      super(Net, self).__init__()

      self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
      self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
      
      self.fc1 = nn.Linear(32*9*9, 256)
      self.fc2 = nn.Linear(256, numActions)

   def forward(self, x):
      x = x.view(-1, 4, 84, 84)

      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)

      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      return x

def train(Q, opt, criterion, eps, 
      memory, gamma, epsilon, numActions, batchSz, cuda=False):
   replay = deque()
   s, r, _ = eps.next(0)
   s = var(s, cuda=cuda)
   for i in range(1000000):

      #Epsilon greedy
      if np.random.rand() > epsilon(i):
         q, a = Q(s).max(1)
         a = a.data[0, 0]
      else:
         a = np.random.randint(0, numActions)
         q = 'Random'
   
      #Sample a transition for memory
      sNxt, r, done = eps.next(a, render=True)
      sNxt = var(sNxt, cuda=cuda)
      replay.append((s, a, r, sNxt, done))
      s = sNxt

      #Limit replay to size memory
      if len(replay) > memory:
         replay.popleft()

      #Sample from memory (Messy stacking operations)
      inds = np.random.randint(0, len(replay), batchSz)
      ss, a, r, sNxt, done = zip(*[replay[i] for i in inds])
      ss = t.stack(ss, 0)
      sNxt = t.stack(sNxt, 0)
      r = var(np.asarray(r).astype(np.float32), cuda=cuda)
      done = var(np.asarray(done).astype(np.float32), cuda=cuda)
      a = var(np.asarray(a).astype(np.int), cuda=cuda).view(-1, 1)
      
      
      #Compute prediction and target
      y = r + (1-done)*t.max(Q(sNxt), 1)[0]*gamma
      pred = t.gather(Q(ss), 1, a)
      loss = criterion(pred, Variable(y.data))

      #Step optimizer
      loss.backward()
      opt.step()

      #Check that Q has not diverged
      if i % 100 == 0:
         print('Iter: ', i, ', Q: ', q)

if __name__ == '__main__':
   cuda = True
   game = 'Breakout-v0'
   eta = 0.00025
   memory = 1e6
   gamma = 0.99
   batchSz = 32
   def epsilon(i):
      decayOver = 1e4
      if i > decayOver:
         return 0.1
      return 1 - 0.9 * float(i)/decayOver

   eps = Emulator(game)
   numActions = eps.numActions
   criterion = nn.MSELoss()

   Q = Net(numActions).cuda()
   opt = t.optim.RMSprop(Q.parameters(), lr=eta)

   train(Q, opt, criterion, eps, 
         memory, gamma, epsilon, numActions,  batchSz=32, cuda=cuda)
