import gym
import numpy as np
from MarioDQNAgent import MarioDQNAgent
import tensorflow as tf

with tf.device('/device:GPU:0'):
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        
        env = gym.make('SuperMarioBros-1-1-v0')
        state_size = [84,84,10] #we will convert the images to grayscale
        gamma = 0.99 #discount
        num_episodes = 2000 #how many levels to attempt
        batch_size = 32 #SARST batch size
        
        #Given six NES buttons, and a lifetime of playing Mario, I've determined there are really on 14
        #button combinations you would ever play. you can override these using ppaquette's wrappers, but
        #i prefer to do them here so you can define different button combos
        button_maps = [
            
            #THE FIRST ACTION MUST BE NOOP
            [0,0,0,0,0,0], #0 - no button,
            
            #AFTER THAT, THE ORDER DOES NOT MATTER
            [1,0,0,0,0,0], #1 - up only (to climb vine)
            [0,1,0,0,0,0], #2 - left only
            [0,0,1,0,0,0], #3 - down only (duck, down pipe)
            [0,0,0,1,0,0], #4 - right only
            [0,0,0,0,1,0], #5 - run only
            [0,0,0,0,0,1], #6 - jump only
            [0,1,0,0,1,0], #7 - left run
            [0,1,0,0,0,1], #8 - left jump
            [0,0,0,1,1,0], #9 - right run
            [0,0,0,1,0,1], #10 - right jump
            [0,1,0,0,1,1], #11 - left run jump
            [0,0,0,1,1,1], #12 - right run jump
            [0,0,1,0,0,1], #13 - down jump
        ]
        
        agent = MarioDQNAgent(env,
                              session,
                              gamma,
                              state_size,
                              button_maps,
                              memory_capacity=10000,
                              num_episodes=num_episodes,
                              batch_size=batch_size)

        agent.train()
	for i in range(10):
		agent.play()
