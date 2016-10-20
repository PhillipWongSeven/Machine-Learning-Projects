import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from itertools import product

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here     
        self.lights= ['red', 'green']
        self.waypoints =  ['forward','right','left']
        self.actions = ['forward','right','left', None]
        self.states =list(product(self.waypoints, self.lights))
        keys =list(product(self.states, self.actions))
        self.getQ  = {key:0 for key in keys}
        

        self.gamma = .5
        self.alpha =  .5  # Try something different everytime
        self.epsilon= 1   # We want our epsilon to decay expoenetially going from full exploration to exploritation. 
        

        #statistics/measures
        self.trial = 0  #number of trials
        self.utility=0
        self.completed=0
        self.success=0

        
    def reset(self, destination=(2,3)):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        print "Some useful statistics:"
        print "------------------------"
        print "The agent is in trial = {}".format(self.trial)
        print "The agent utility is ={}".format(self.utility)
        print "The agent complete on time = {}".format(self.completed)
        if self.trial !=0:
            self.success = float(self.completed)/float(self.trial)
        print "The agent success rate = {}".format(self.success)
        print
        #update epsilon for every new trial

        self.epsilon = math.exp(-.3*self.trial)
        #self.alpha   = 1- float(self.trial)/99
        #self.alpha   = .6
        #self.alpha   = math.exp( -.5*self.trial)

        self.trial+=1


    def choose_action(self, state):
        if random.random() < self.epsilon: # exploration 
            action = random.choice(self.actions) 
        else: 
            q = [self.getQ[(state, a)] for a in self.actions] 
            maxQ = max(q) 
            count = q.count(maxQ) 
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
 
            action = self.actions[i]
        return action
    def updateQ(self, current_state, action, future_state, reward):
        total = reward + self.gamma * self.getQ[(future_state, self.choose_action(future_state))]
        self.getQ[(current_state, action)] = (1 - self.alpha)*self.getQ[(current_state, action)] + self.alpha * total


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        current_state = (self.next_waypoint, inputs['light'])
        self.state = current_state
    
        # TODO: Select action according to your policy

        action = self.choose_action(current_state)

        # Execute action and get reward
        reward = self.env.act(self, action)  

        self.utility+=reward  # matric

        if reward==12:
            self.completed+=1 

        # TODO: Learn policy based on state, action, reward

        inputs=self.env.sense(self)        
        self.next_waypoint = self.planner.next_waypoint()
        future_state = (self.next_waypoint, inputs['light'])
        
        self.updateQ(current_state, action, future_state, reward)



        # self.state[0] is the current state waypoint
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, waypoints={}, reward = {}, ".format(deadline, inputs, action, self.state[0], reward)  # [debug]
   

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment() # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100) # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
