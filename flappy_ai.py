from itertools import cycle
import random
import sys
import numpy as np
import os
import time
import pygame
from pygame.locals import *


# filename='Flappy_AI_log_'+time.strftime("%Y_%m_%d_%H_%M_%S")
# file=open(filename,'w')


FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range


def main(population):
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        scores = mainGame(population)

        return scores
        exit()






def mainGame(population):


    score = playerIndex = loopIter = basex = 0


    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4


    playerx = int(SCREENWIDTH * 0.2)
    playery = []
    playerVelY = []
    playerMaxVelY = []
    playerMinVelY = []
    playerAccY = []
    playerFlapAcc = []
    playerFlapped = []
    scores=[]
    crashCheck=[]
    alive=len(population)

    for i in range(len(population)):
		playery += [int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2) + random.randint(-SCREENHEIGHT/4,SCREENHEIGHT/4)]
		playerVelY += [-9]
		playerMaxVelY += [10]
		playerMinVelY += [-8]
		playerAccY += [1]
		playerFlapAcc += [-9]
		playerFlapped += [False]
		scores += [0]
		crashCheck += [False]



    clock = pygame.time.Clock()
    time_score = 0


    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
#            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
#            	for ind in range(len(population)):
#            		if not crashCheck[ind]:
#				        if playery[ind] > -2 * IMAGES['player'][0].get_height():
#				            playerVelY[ind] = playerFlapAcc[ind]
#				            playerFlapped[ind] = True



        time_passed = clock.tick(FPS)
        time_sec = time_passed / 1000.0
        time_score += time_sec


        # check for crash here

        for ind in range(len(population)):

        	if not crashCheck[ind]:

				crashTest = checkCrash({'x': playerx, 'y': playery[ind], 'index': playerIndex},
				                       upperPipes, lowerPipes)
				if crashTest[0]:
					scores[ind]=time_score
					crashCheck[ind]=True
					alive-=1

		if alive == 0:
			return scores




        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1


        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        for ind in range(len(population)):
        	if not crashCheck[ind]:
				if playerVelY[ind] < playerMaxVelY[ind] and not playerFlapped[ind]:
				    playerVelY[ind] += playerAccY[ind]
				if playerFlapped[ind]:
				    playerFlapped[ind] = False
				playerHeight = IMAGES['player'][0].get_height()
				playery[ind] += min(playerVelY[ind], BASEY - playery[ind] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)

        #draw players
        for ind in range(len(population)):
        	if not crashCheck[ind]:
        		if playery[ind]<0:
		    		playery[ind]=0
        		SCREEN.blit(IMAGES['player'][0], (playerx, playery[ind]))



        for pipe in lowerPipes:
        	if pipe['x']+52>playerx:
				pipe_pos_x, pipe_pos_y = pipe['x'], pipe['y']
				break



        for ind in range(len(population)):

			if not crashCheck[ind]:

				pposy=int(playery[ind]/32)
				pposx=int(playerx/32)

		    	x = np.array([playery[ind], pipe_pos_x, pipe_pos_y])

		    	### PREDICTION ###
		    	prediction = predict(population[ind]['Genome'],x)

		    	if prediction == 1:
		    		if playery[ind] > -2 * IMAGES['player'][0].get_height():
		    			playerVelY[ind] = playerFlapAcc[ind]
		    			playerFlapped[ind] = True



        pygame.display.update()
        FPSCLOCK.tick(FPS)







def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask









def write_file(ind):

	model=ind['Genome']

	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	file.write('W1 = [ ')
	for i in range(len(W1)):
		for j in range(len(W1[0])):
			if i==len(W1)-1 and j==len(W1[0])-1:
				file.write(str(W1[i,j]))
			else:
				file.write(str(W1[i,j]) + ', ')
	file.write(' ]\n')
	file.write('\n')
	file.write('W2 = [ ')
	for i in range(len(W2)):
		for j in range(len(W2[0])):
			if i==len(W2)-1 and j==len(W2[0])-1:
				file.write(str(W2[i,j]))
			else:
				file.write(str(W2[i,j]) + ', ')
	file.write(' ]\n')
	file.write('\n')

	file.write('b1 = [ ')
	for i in range(len(b1)):
		for j in range(len(b1[0])):
			if i==len(b1)-1 and j==len(b1[0])-1:
				file.write(str(b1[i,j]))
			else:
				file.write(str(b1[i,j]) + ', ')
	file.write(' ]\n')
	file.write('\n')
	file.write('b2 = [ ')
	for i in range(len(b2)):
		for j in range(len(b2[0])):
			if i==len(b2)-1 and j==len(b2[0])-1:
				file.write(str(b2[i,j]))
			else:
				file.write(str(b2[i,j]) + ', ')
	file.write(' ]\n')
	file.write('\n')
	file.write('Fitness: ' + str(ind['Fitness']) + '\n')
	file.write('\n')
	file.write('\n')
	file.write('%%%%%%%%%%%%%%%%%%%')
	file.write('\n')
	file.write('\n')






##### NEURAL NETWORK PREDICT #####

def predict(model, x):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)[0]



##### MUTATION #####

def mutation(model,prob):

	W1, b1, W2, b2= model['W1'], model['b1'], model['W2'], model['b2']

	rW1=np.zeros((nn_input_dim, nn_hdim1))
	rb1=np.zeros((1, nn_hdim1))
	rW2=np.zeros((nn_hdim1, nn_output_dim))
	rb2=np.zeros((1, nn_output_dim))


	for i in range(nn_input_dim):
		for j in range(nn_hdim1):
			if random.random()<=prob:
				rW1[i][j]=random.random()
			else:
				rW1[i][j]=W1[i][j]

	for j in range(nn_hdim1):
		if random.random()<=prob:
			rb1[0][j]=random.random()
		else:
			rb1[0][j]=b1[0][j]

	for i in range(nn_hdim1):
		for j in range(nn_output_dim):
			if random.random()<=prob:
				rW2[i][j]=random.random()
			else:

				rW2[i][j]=W2[i][j]

	for j in range(nn_output_dim):
		if random.random()<=prob:
			rb2[0][j]=random.random()
		else:
			rb2[0][j]=b2[0][j]




	return {'W1': rW1, 'b1': rb1, 'W2': rW2, 'b2': rb2}



##### REPRODUCTION #####

def reproduction(modelA,modelB):

	W1A, b1A, W2A, b2A = modelA['W1'], modelA['b1'], modelA['W2'], modelA['b2']
	W1B, b1B, W2B, b2B= modelB['W1'], modelB['b1'], modelB['W2'], modelB['b2']

	rW1=np.zeros((nn_input_dim, nn_hdim1))
	rb1=np.zeros((1, nn_hdim1))
	rW2=np.zeros((nn_hdim1, nn_output_dim))
	rb2=np.zeros((1, nn_output_dim))

	for i in range(nn_input_dim):
		for j in range(nn_hdim1):
			if random.randint(0,1)==1:
				rW1[i][j]=W1A[i][j]
			else:
				rW1[i][j]=W1B[i][j]

	for j in range(nn_hdim1):
		if random.randint(0,1)==1:
			rb1[0][j]=b1A[0][j]
		else:
			rb1[0][j]=b1B[0][j]

	for i in range(nn_hdim1):
		for j in range(nn_output_dim):
			if random.randint(0,1)==1:
				rW2[i][j]=W2A[i][j]
			else:
				rW2[i][j]=W2B[i][j]

	for j in range(nn_output_dim):
		if random.randint(0,1)==1:
			rb2[0][j]=b2A[0][j]
		else:
			rb2[0][j]=b2B[0][j]


	return {'W1': rW1, 'b1': rb1, 'W2': rW2, 'b2': rb2}





########## START ##########


np.random.seed(0)

### SETTINGS ###
population_size=50
population=[]

max_fitnesses = 0
best_ind_fitness=0

mutation_prob=0.5
survivors_prob=0.1

nn_input_dim = 3
nn_hdim1 = 5
nn_output_dim = 2
################




for generation in range(10000):

	if generation==0:
		# Generate models
		for j in range(population_size):
		 	# POPULATION
			W1 = np.random.randn(nn_input_dim, nn_hdim1) / np.sqrt(nn_input_dim)
			b1 = np.random.randn(1, nn_hdim1) / np.sqrt(nn_input_dim)
			W2 = np.random.randn(nn_hdim1, nn_output_dim) / np.sqrt(nn_hdim1)
			b2 = np.random.randn(1, nn_output_dim) / np.sqrt(nn_hdim1)
		   	model = { 'Generation': generation, 'Ind': j, 'Genome': {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, 'Origin': 'Genesis', 'Fitness': 'und'}
		   	population+=[model]

	else:

		### DEATH ###

		survivors= random.randint(2,population_size*survivors_prob)

		new_population=[]

		for survivor in range(survivors):
			target=max(fitnesses)
			population_len=len(population)
			for index in range(population_len):
				ind=population[index]
				if ind['Fitness']==target:
					new_population+=[ind]
					population.pop(index)
					fitnesses.pop(index)
					break

		population=new_population



		### NEW INDS ###

		# POPULATION
		pre_population = population

		size_pre_population = len(pre_population)

		nr_new_inds = population_size - size_pre_population

		for x in range(nr_new_inds):
			if random.randint(0,1)==1:

				### MUTATION ###

				prev_genome = pre_population[random.randint(0,size_pre_population-1)]['Genome']
				new_genome = mutation(prev_genome,mutation_prob)
				population += [{ 'Population': 1, 'Generation': generation, 'Ind': (size_pre_population)+x, 'Genome': new_genome, 'Origin': 'Mutation', 'Fitness': 'und'}]

			else:

				### REPRODUCTION ###
				indAB = random.sample(pre_population,2)
				new_genome=reproduction(indAB[0]['Genome'],indAB[1]['Genome'])
				population += [{ 'Population': 1, 'Generation': generation, 'Ind': (size_pre_population)+x, 'Genome': new_genome, 'Origin': 'Reproduction', 'Fitness': 'und'}]



##
	print 'Generation: ' + str(generation)
##


	### TEST MODELS / FITNESS ###


	### PLAY GAME ###
	fitnesses=main(population)
	###

	for ind in range(population_size):
		population[ind]['Fitness']=fitnesses[ind]



	#if generation%10==0:
	best_ind_index=np.argmax(fitnesses)
	best_ind = population[best_ind_index]
	# write_file(best_ind)

	print(max(fitnesses))
