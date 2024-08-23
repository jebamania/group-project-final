# https://www.imdb.com/title/tt0133093/characters/nm0915989
# Agent Smith : Never send a human to do a machine's job.

# used as starting point: https://github.com/nuno-faria/tetris-ai/blob/master/dqn_agent.py

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
import numpy as np
import random

class DQNAgentSmith:

    '''Deep Q Learning Agent

    '''

    def __init__(self, state_size):

        self.state_size = state_size
        self.model = self._build_model()
        self.epsilon = 1
        self.epsilon_stop_episode = 1500
        self.epsilon_min = 0
        self.epsilon_decay = 0.999

    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        model = Sequential()
        
        n_neurons = 24
        activations = ['relu', 'relu', 'linear']
        
        model.add(Dense(
            n_neurons,
            input_dim=self.state_size,
            activation=self.activations[0],
        ))

        model.add(Dense(n_neurons, activation=self.activations[1]))
        model.add(Dense(1, activation=self.activations[2]))

        model.compile(loss='mse', optimizer='adam')
        
        return model
    
    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        '''Random score for a certain action'''
        return random.random()


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state, verbose=0)[0]


    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)
    
    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            for state in states:
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state
    
    def train(self, batch_size=32, epochs=3):
        '''Trains the agent'''
        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= batch_size:

            batch = random.sample(self.memory, batch_size)

            # Get the expected score for the next states, in batch (better performance)
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]

            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)
            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    # Partial Q formula
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            # Fit the model to the given values
            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay


    def get_next_states(self, state):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states