import numpy as np
import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=True, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.decay_rate=0.05
        self.decay_time=0
        self.action_threshold=-0.0 #Waypoint action threshold regulates whether we want to take risky behavior or not.




    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        ###########
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        self.epsilon=self.epsilon*np.exp(-self.decay_rate*self.decay_time)
        #self.epsilon-=0.05
        self.decay_time+=1

        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon=0
            self.alpha=0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the
            environment. The next waypoint, the intersection inputs, and the deadline
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ###########
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 0

        # building current state.
        '''
        Rules of Right applied. Fighting the state space size!
        if light is red:
            possible valid actions:
                idle
                right if no upcoming traffic from left (left=forward)
        state needed to capture: light=red, left traffic

        if light is green:
            possible valid actions:
                forward
                right
                left = if not oncoming traffic

        state needed to capture: light, oncoming traffic.
        '''

        state = []

        state.append('light=' + inputs['light'])
        if inputs['light']=='red':
            state.append('left=' + str(inputs['left']))
        else:
            state.append('oncoming='+str(inputs['oncoming']))

        state=tuple(state)

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ###########
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = -np.inf
        max_action=''

        for action,Q_value in self.Q[state].items():
            if Q_value>maxQ:
                maxQ=Q_value
                max_action=action
        return maxQ

    def get_maxQ_action(self,state):
        """ The get_maxQ_action function is called when the agent is choosing action with
            maximum Q-value of all actions based on the 'state' the smartcab is in. """
        maxQ = -np.inf
        max_action=''

        for action,Q_value in self.Q[state].items():
            if Q_value>maxQ:
                maxQ=Q_value
                max_action=action

        return max_action

    def createQ(self, state,initial_values=0.0):
        """ The createQ function is called when a state is generated by the agent. """

        ###########
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning:
            if state not in self.Q:
                self.Q[state]={}
                for action in self.valid_actions:
                    self.Q[state][action]=initial_values
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()



        ###########
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if self.learning:
            if random.random()>self.epsilon:
                '''now we are acting in non-random fashion and need to understand which action to choose.
                Suggested rule is following:
                    if Q value of waypoint action is bigger than self.action_treshold, then agent takes waypoint action
                    otherwise, take the action with max Q to earn reqard without necessary reaching the goal.
                '''
                waypoint_action_Qvalue=self.Q[state][self.next_waypoint]
                if waypoint_action_Qvalue>self.action_threshold:
                    return self.next_waypoint
                else:
                    return self.get_maxQ_action(state)

        return self.env.valid_actions[random.choice([0, 1, 2, 3])]


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards
            when conducting learning. """

        ###########
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        self.Q[state][action]=self.Q[state][action] + self.alpha*(reward- self.Q[state][action])

        return


    def update(self):
        """ The update function is called when a time step is completed in the
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        if action=='none': #convert back to None value
            action=None
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return


def run():
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    #env = Environment(num_dummies=10,grid_size=(4,4))
    env = Environment(verbose=True)

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent)

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env,update_delay=0.01, log_metrics=True,display=True, optimized=True)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10)


if __name__ == '__main__':
    run()
