import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
import matplotlib.animation as ani
import time


def read_file(filename):
    # read csv and return it as a numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data


class Maze:
    # class Maze for easy processing
    # stores the Maze as a numpy array where 0 is clear and 1 is a wall
    # by default, start is the top left corner and the goal is the bottom right
    # the starting and ending points should be clear for meaningful results
    # The program is designed for square-shaped mazes
    def __init__(self, maze, start=[0, 0]):
        self.grid = np.array(maze)
        self.start = start
        self.goal = [len(self.grid)-1, len(self.grid)-1]

    def step(self, position, action):
        # returns the position after a step taken from a given position
        if action == 0:
            # move down
            position[0] -= 1
        elif action == 1:
            # up
            position[0] += 1
        elif action == 2:
            # left
            position[1] -= 1
        else:
            # right
            position[1] += 1
        return position


class Learner:
    # class Learner stores information about the agent and its training
    def __init__(self,  maze: Maze, alpha=1, gamma=0.99):
        # the maze we are learning
        self.maze = maze
        # parameter for controlling how much the q-values change based on new information
        self.alpha = alpha
        # parameter for controlling how much a move is rewarded for possible future rewards
        self.gamma = gamma
        # our agent is initially at the start
        self.position = self.maze.start
        # The q-values for every move for every position in the grid, all initially zero.
        # This a list inside a list inside a list. The first list is the rows of the grid.
        # Every element in a row is a list of 4 elements containing the q-values for every action to be taken from the corresponding position.
        # indices 0,1,2,3 stand for actions down, up, left and right
        self.q_values = [[[0, 0, 0, 0] for _ in range(
            len(self.maze.grid))] for _ in range(len(self.maze.grid))]

    def choose_action(self, epsilon=0.1):
        # choose the next action, with probability epsilon we pick a random direction to go to,
        # otherwise we pick the direction with the highest q-value
        if default_rng().uniform(0, 1) < epsilon:
            return default_rng().integers(0, 4)
        else:
            return np.argmax(self.q_values[self.position[0]][self.position[1]])

    def consequence_of_action(self, new_position):
        # we check if the position after an action is allowed and give a reward or penalty accordingly
        # if we move into a clear space we give a small penalty to punish long routes and unnecessary loops
        # the algorithm seeks the shortest path
        valid = True  # is move valid
        reward = -0.05
        if not (0 <= new_position[0] < len(self.maze.grid) and 0 <= new_position[1] < len(self.maze.grid)):
            # if we hit the edge of the grid, we give a big penalty
            # we cannot move into the edge so the move is invalid
            valid = False
            reward = -5
        elif self.maze.grid[new_position[0]][new_position[1]] == 1:
            # we also a give a penalty for hitting the wall inside the maze
            # we cannot move into a wall so the move is invalid
            reward = -2
            valid = False
        elif new_position == self.maze.goal:
            # if we found the goal, we give a big reward
            reward = 10
        return valid, reward

    def learn(self, state, action, reward, next_state):
        # update the q-value of an action after it has been made
        # The value q(s,a) of the action a for a position or state s is:
        # q(s,a):=q(s,a)+alpha(reward+ gamma*max(q'(s',a'))-q(s,a)),
        # where max(q'(s',a')) is the highest q-value of actions for the new position that we would be in after the action is done.
        # the column of the point is state [0], row is state[1], and action is the action 0, 1, 2 or 3
        # q(s,a)
        old_value = self.q_values[state[0]][state[1]][action]
        # q'(s',a')
        next_max = np.max(self.q_values[next_state[0]][next_state[1]])
        # calculate the q-value
        new_value = old_value + self.alpha * \
            (reward + self.gamma * next_max - old_value)
        # save the q-value in the Learner object
        self.q_values[state[0]][state[1]][action] = new_value

    def train(self, episodes=100, epsilon=0.1):
        # train the agent to solve the maze, the number of episodes is how many times it goes through the maze and updates q-values
        for episode in range(episodes):

            # put agent back to starting position after previous episode
            self.position = self.maze.start
            while self.position != self.maze.goal:
                # stop when the goal is found
                current = copy.copy(self.position)
                # choose direction
                action = self.choose_action(epsilon=epsilon)
                # check the consequences
                new_position = self.maze.step(current, action)
                valid, reward = self.consequence_of_action(new_position)

                if not valid:
                    # if the move is not valid, we stay in the same position and suffer a penalty
                    self.learn(self.position, action, reward, self.position)
                else:
                    # if the move is valid, we move into the new position
                    self.learn(self.position, action, reward, new_position)
                    self.position = new_position

    def result(self):
        # after training, this is called and the agent acts according to the policies set in training
        # returns a boolean telling if we reached the goal and the path taken
        # we are initially at the starting point
        self.position = copy.copy(self.maze.start)
        success = False
        path = [self.maze.start]
        while self.position != self.maze.goal and path.count(self.position) < 2:
            # we take steps until we reach the goal or we step into a position that we have already visited, meaning that the agent is lost
            # set epsilon to zero since we don't want to explore new paths anymore
            action = self.choose_action(epsilon=0)
            # test if the move is allowed, it generally should always be if there has been any training
            test = copy.copy(self.position)
            self.maze.step(test, action)
            if self.consequence_of_action(test)[0] == True:
                # if the move is allowed, agent takes the step
                self.maze.step(self.position, action)
            # add the position to the taken path
            current = copy.copy(self.position)
            path.append(current)
        if path[-1] == self.maze.goal:
            # if the last position in the path is the goal, we were successful
            success = True
        return success, path

    def show_policies(self):
        # show the direction that has the highest q-value in every position
        n = len(self.q_values)
        # create a grid for the image
        # we fill the grid with different numbers to tell what is happening in that position
        grid = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # fill the grid with the actions 0, 1, 2 or 3 depending on which has the highest q-value
                grid[i, j] = self.q_values[i][j].index(
                    max(self.q_values[i][j]))
        for i in range(n):
            for j in range(n):
                if (self.q_values[i][j]) == [0, 0, 0, 0]:
                    # mark point in the grid if the point is not visited
                    grid[i, j] = 8
        for i in range(n):
            for j in range(n):
                if self.maze.grid[i, j] == 1:
                    # mark point if it is a wall
                    grid[i, j] = 7
        # mark the goal
        grid[n-1, n-1] = 5

        fig, ax = plt.subplots()

        # map the numbers to letters to give a clear visualisation
        # Up, Down, Left, Right, Goal, wall, Not visited
        number_to_letter = {0: 'U', 1: 'D',
                            2: 'L', 3: 'R', 5: 'G', 7: '▉', 8: 'N'}

        # add text annotations for each cell
        for i in range(n):
            for j in range(n):
                # place the letters in the image
                letter = number_to_letter[grid[i, j]]
                ax.text(j, i, letter, ha='center', va='center', fontsize=10)

        # set the x and y ticks to match the grid
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))

        # set the x and y limits to match the grid
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        # remove the tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.show()

    def animate(self, path):
        # animate the path taken by the agent, takes a list of the positions of the agent as an argument
        frames = []
        for p in path:
            # create frames to animate, add the agent to the grids
            frame = copy.copy(self.maze.grid)
            frame[p[0]][p[1]] = 2
            frame[-1][-1] = 3
            frames.append(frame)

        # for the plot
        num_frames = len(path)
        cmap = ListedColormap(['pink', 'black', 'blue', 'red'])

        # create the figure and axis
        fig, ax = plt.subplots()

        # initialize the plot
        im = ax.imshow(frames[0], cmap=cmap, interpolation='nearest')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # define the update function for the animation
        def update(frame):
            im.set_array(frames[frame])
            return [im]

        # create the animation
        animation = ani.FuncAnimation(
            fig, update, frames=num_frames, blit=True, repeat=True)
        animation.save('isoin.gif')

        plt.show()


if __name__ == '__main__':
    # the simulation is run
    # change the file name for different maze
    ma = read_file('small_maze.txt')
    # make a maze of the grid
    maze = Maze(ma)
    # initialize a learner
    learner = Learner(maze)

    # time the training
    start = time.perf_counter()
    # train, change the parameter for different training configuration
    learner.train(1000)
    end = time.perf_counter()
    training_time = round(end - start, 2)

    print('training took '+ str(round(end - start, 2)) +" s")
    success, path = learner.result()

    if success:
        print('goal was found')
        print('length:', len(path))
    else:
        print('goal was not found')

    learner.animate(path)
    # may require some zooming to see clearly
    learner.show_policies()
