from pdb import set_trace as T
import torch as t
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
from Batcher import Emulator

from collections import deque
import sys

#Exponentially decaying average
class EDA():
   def __init__(self, k=0.999):
      self.k = k
      self.eda = 0.0

   def update(self, x):
      self.eda = (1-self.k)*x + self.k*self.eda

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
      
      self.fc1 = nn.Linear(3456, 256)
      self.fc2 = nn.Linear(256, numActions)

   def forward(self, x):
      x = x.view(-1, 4, 110, 84)

      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)

      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      return x

class ReplayMememory:
   def __init__(self, maxSamples):
      self.capacity = maxSamples   
      self.queue = deque()

   def push(self, s, a, r, sNxt, done):
      self.queue.append((s, a, r, sNxt, done))
      if len(self.queue) > self.capacity:
         self.queue.popleft()

   def sample(self, numSamples, toVar=True, cuda=False):
      inds = np.random.randint(0, len(self.queue), numSamples)
      s, a, r, sNxt, done = zip(*[self.queue[i] for i in inds])
 
      if toVar:
         s = t.stack(s, 0)
         sNxt = t.stack(sNxt, 0)
         r = var(np.asarray(r).astype(np.float32), cuda=cuda)
         done = var(np.asarray(done).astype(np.float32), cuda=cuda)
         a = var(np.asarray(a).astype(np.int), cuda=cuda).view(-1, 1)

      return s, a, r, sNxt, done

class EpsilonGreedyPolicy:
   def __init__(self, epsMin, decayOver):
      self.epsMin    = epsMin
      self.decayOver = decayOver

   def sample(self, Q, s, i, numActions):
      if np.random.rand() > self.epsilon(i):
         q, a = Q(s).max(1)
         a = a.data[0, 0]
      else:
         a = np.random.randint(0, numActions)
         q = None

      return q, a
 
   def epsilon(self, i):
      if i > self.decayOver:
         return self.epsMin

      return 1.0 -  (1 - self.epsMin)*float(i)/self.decayOver

class StatTracker:
   def __init__(self, k=0.99):
      self.edaQ = EDA()
      self.edaR = EDA()
      self.r = 0.0

   def update(self, q, r, done):
      self.r += r

      if done:
         self.edaR.update(self.r)
         self.r = 0.0
         return
      
      if q is not None:
         self.edaQ.update(q)

   def stats(self):
      return self.edaQ.eda, self.edaR.eda
 

class QLoss:
   def __init__(self, Q, gamma):
      self.Q = Q
      self.gamma = gamma
      self.criterion = nn.MSELoss()

   def __call__(self, s, a, r, sNxt, done):
      y = r + (1-done)*t.max(self.Q(sNxt), 1)[0]*self.gamma
      pred = t.gather(self.Q(s), 1, a)
      loss = self.criterion(pred, Variable(y.data))
      return loss

class DoubleQLoss:
   def __init__(self, Q, gamma):
      self.Q = Q
      self.gamma = gamma
      self.criterion = nn.MSELoss()

   def __call__(self, s, a, r, sNxt, done):
      y = r + (1-done)*t.max(self.Q(sNxt), 1)[0]*self.gamma
      pred = t.gather(self.Q(s), 1, a)
      loss = self.criterion(pred, Variable(y.data))
      return loss

class DQNAgent:
   def __init__(self, Q, env, opt, cuda=False,
            memorySz=50000, gamma=0.99, epsMin=0.1, decayOver=1e6):
      self.env = env
      self.policy = EpsilonGreedyPolicy(epsMin, decayOver)
      self.memory = ReplayMememory(memorySz)
      self.loss = QLoss(Q, gamma)
      self.tracker = StatTracker(k=0.999)

      self.Q = Q
      self.opt = opt
      self.numActions = env.numActions
      self.i = 0
      self.cuda = cuda

      s, self.r, _ = env.next(0)
      self.s = var(s, cuda=cuda)

   def interact(self):
      q, a = self.policy.sample(self.Q, self.s, self.i, self.numActions)
      sNxt, r, done = self.env.next(a, render=False)
      self.tracker.update(q.data[0,0] if q is not None else q, r, done)

      sNxt = var(sNxt, cuda=self.cuda)
      self.memory.push(self.s, a, r, sNxt, done)
      self.s = sNxt
      self.i += 1

   def learn(self, numSamples):
      ss, a, r, sNxt, done = self.memory.sample(numSamples, toVar=True, cuda=cuda)
      loss = self.loss(ss, a, r, sNxt, done)

      #Step optimizer
      self.opt.zero_grad()
      loss.backward()
      self.opt.step()

   #Test time
   def act(self):
      q, a = self.policy.sample(self.Q, self.s, 1e12, self.numActions)
      sNxt, r, done = self.env.next(a, render=True)
      sNxt = var(sNxt, cuda=self.cuda)
      self.s = sNxt
 
 
def play(agent, cuda=False):
   while True:
      agent.act()

def train(agent, maxIters, batchSz, saveName, cuda=False):

   for i in range(maxIters):
      agent.interact() 
      agent.learn(batchSz)

      #Check that Q has not diverged
      if i % 1000 == 0:
         t.save(agent.Q.state_dict(), 'saves/'+saveName)
         q, a = agent.tracker.stats()
         print('Iter: ', i, ', Q: ', q, ', V: ', a)

if __name__ == '__main__':
   name = sys.argv[1]
   validate = False
   cuda = True 
   game = 'Breakout-v0'

   eta = 0.00025
   alpha = 0.95
   eps = .01

   epsMin = 0.1
   decayOver = 1e6
     
   memorySz = 50000
   maxIters = int(1e9)
   gamma = 0.99
   batchSz = 32

   env = Emulator(game)
   Q = Net(env.numActions)
   if cuda: Q = Q.cuda()
   opt = t.optim.RMSprop(Q.parameters(), lr=eta, alpha=alpha, eps=eps)

   agent = DQNAgent(Q, env, opt, cuda, memorySz, 
         gamma, epsMin, decayOver)

   if validate:
      agent.Q.load_state_dict(t.load('saves/'+name))
      play(agent, cuda=cuda)
   else:
      train(agent, maxIters, batchSz, name, cuda=cuda)
