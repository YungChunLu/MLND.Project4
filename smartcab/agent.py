import logging
import random
import matplotlib.pyplot as plt
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
logging.basicConfig(filename='debug.log',level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p')

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    actions = [None, 'forward', 'right', 'left']

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.discount = 0.7 # Discount Factor
        self.trials = 1 # How many trial has been run
        self.learning = 0.5 / self.trials # Learining Rate
        self.epsilon = 0.9 ** (3 * self.trials) # To use epsilon-greedy agent        
        self.reward = 0 # Total reward got at this trial
        self.success = 0 # How many times reaching to the destination
        self.penalties = 0 # How many times incuring penalties
        self.records = [] # Record how many times incuring penalties during each trial
        self.rewards = [] # Record total reward got during each trial

        self.Q = defaultdict(int) # Q Table, State : (GPS, light, oncoming, left, action)
        
        for action in self.actions:
            # for red light
            for oncoming in ['left', 'else']:
                for left in ['forward', 'else']:
                    self.Q['right', 'red', oncoming, left, action] = 0
  
            for GPS in ['left', 'forward', None]:
                self.Q[GPS, 'red', 'else', 'else', action] = 0

            # for green light
            for oncoming in ['forward', 'else']:
                self.Q['left', 'green', oncoming, 'else', action] = 0

            for GPS in ['right', 'forward', None]:
                self.Q[GPS, 'green', 'else', 'else', action] = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.reward = 0
        self.trials += 1
        self.learning = 0.5 / self.trials
        self.epsilon = 0.9 ** (3 * self.trials)
        self.penalties = 0
        # TODO: Prepare for a new trip; reset any variables here, if required

    def best_action(self, GPS, inputs):
        if random.random() > self.epsilon:
            candidates = [(GPS, inputs['light'], inputs['oncoming'], inputs['left'], action) for action in self.actions]
            q_values = [self.Q[candidate] for candidate in candidates]
            Q = max(q_values)
            best = [i for i,v in enumerate(q_values) if v == Q]
            return candidates[random.choice(best)][-1]
        else:
            self.epsilon *= 0.9
            return GPS

    def process_inputs(self, GPS, inputs):
        # for red light
        if inputs['light'] is 'red':
            if GPS is 'right':
                # for oncoming car, we only care whether it is left or not
                if inputs['oncoming'] is not 'left':
                    inputs['oncoming'] = 'else'
                # for left car, we only care whether it is forward or not
                if inputs['left'] is not 'forward':
                    inputs['left'] = 'else' 
            else:
                inputs['oncoming'] = 'else'
                inputs['left'] = 'else'
        # for green light 
        else:
            if GPS is 'left':
                # for oncoming car, we only care whether it is forward or not
                if inputs['oncoming'] is not 'forward':
                    inputs['oncoming'] = 'else'
            else:
                inputs['oncoming'] = 'else'
                inputs['left'] = 'else'
        return inputs

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        inputs = self.process_inputs(self.next_waypoint, inputs)
        self.state = inputs
        
        # TODO: Select action according to your policy
        action = self.best_action(self.next_waypoint, inputs)

        # Naive agent
        # action = random.choice(self.actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward += reward

        # TODO: Learn policy based on state, action, reward
        q_before = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], action)
        next_GPS = self.planner.next_waypoint()
        next_inputs = self.process_inputs(next_GPS, self.env.sense(self))
        next_action = self.best_action(next_GPS, next_inputs)
        q_after = (next_GPS, next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], next_action)

        self.Q[q_before] = (1 - self.learning) * self.Q[q_before] + self.learning * (reward + self.discount * self.Q[q_after])
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=5.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

    fig, ax = plt.subplots( nrows=1, ncols=1) 
    plt.xlabel('Order of trials')
    plt.ylabel('# of incured penalties')
    plt.title('Penalties')
    ax.plot(a.records)
    fig.savefig('penalties.png')

    fig, ax = plt.subplots( nrows=1, ncols=1) 
    plt.xlabel('Order of trials')
    plt.ylabel('# of rewards')
    plt.title('Rewards')
    ax.plot(a.rewards)
    fig.savefig('rewards.png')
    

if __name__ == '__main__':
    run()

    
