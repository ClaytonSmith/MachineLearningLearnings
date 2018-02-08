from __future__ import print_function
import pygame
import sys
import time
import math
import numpy as np
from itertools import chain


# Stuff for the thoughts

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 1 pixel = 1 epoch
# Fake sin driver

# # Canvas Opt
caption            = 'Test'


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE  = (  0,   0, 255)

#                 R    G    B
BACKGROUND  = (   9,  33,  64)
GRID        = (   2,  73,  86)
WAVE1       = GREEN #( 229,  70,  97)
WAVE2       = ( 255, 166,  68)
WAVE3       = ( 153, 138,  47)
WAVE4       = (  44,  89,  79)
TRANSPARENCY =(1, 1, 1)
COLORKEY=(127, 127, 0)

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1  = nn.RNNCell(1, 61)
        self.lstm2  = nn.RNNCell(61, 61)
        self.linear = nn.Linear(61, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t   = Variable(torch.zeros(input.size(0), 61).double(), requires_grad=False)
        c_t   = Variable(torch.zeros(input.size(0), 61).double(), requires_grad=False)
        h_t2  = Variable(torch.zeros(input.size(0), 61).double(), requires_grad=False)
        c_t2  = Variable(torch.zeros(input.size(0), 61).double(), requires_grad=False)
        
    
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):            
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            #h_t = self.lstm1(input_t, h_t)
            #h_t2 = self.lstm2(h_t, h_t2)
            
            output = self.linear(h_t2)
            outputs += [output]
            
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            #h_t      = self.lstm1(output, h_t)
            #h_t2     = self.lstm2(h_t, h_t2)
            output   = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs

def wave_moment(amp, freq, epoc):
    return  np.sin( freq *  epoc).astype('float64') #* amp


# parent => surface
def generate_subsurface( parent, ratio=(1,1)):
    s = pygame.Surface( tuple( int( l * r) for l, r in zip(parent.get_size(), ratio)))#, pygame.SRCALPHA, 32)
    #s = s.convert_alpha()
    s.set_colorkey((1,1,1))
    return s


# args should all be [] but can accept surface
def merge_surfaces(base, *args):
    result = reduce( lambda x,y: c.blit(y), list(chain.from_iterable( [ arg if isinstance( arg, list) else [ arg] for arg in args ] )), base) 
    return result



window_size = 300
######## THIS WILL BE TRANSPLANTED #######
buffer = 5

# Base wave
basew_f = 4 # Frequency
basew_a = 40 # amplitude
basew_s = .05  # Speed 

# compound wave
subw_f = 2
subw_a = 40
subw_s = .05

when = 2/3
pygame.init()
pygame.display.set_caption("Sine Wave")

# Some config width height settings
base_surface_size   = (base_surface_width, base_surface_heigth) = ( 1000, 800)

# Make a screen to see
screen          = pygame.display.set_mode(base_surface_size)

# Make a surface to draw on
base_surface    = pygame.Surface( base_surface_size )


# XY Scalar
wave_container_ratio  = ( 1,   1)
wave_surface_ratio    = ( 1,   1)   # in respect to container 
grid_surface_ratio    = ( 1,   1)
pred_surface_ratio    = ( 1 - when,   1)   # Make sure to 

# From base_surface
wave_container  = generate_subsurface( base_surface,   wave_container_ratio)
wave_surface    = generate_subsurface( wave_container, wave_surface_ratio)
grid_surface    = generate_subsurface( wave_container, grid_surface_ratio)
pred_surface    = generate_subsurface( wave_container, pred_surface_ratio)
act_surface     = generate_subsurface( wave_container, pred_surface_ratio)

# Mind the pred_surface when changing now
now             = int( wave_surface.get_width() * when)
future          = int( wave_surface.get_width() - now ) 

base_surface.fill(BACKGROUND)
wave_surface.fill(TRANSPARENCY)
act_surface.fill( TRANSPARENCY)
pred_surface.fill(TRANSPARENCY)
#grid_surface.fill(TRANSPARENCY)
grid_surface.fill(BACKGROUND)

# #Draw grid
for moment in range( now, grid_surface.get_width()):
    if not (now - moment) % 30:
        pygame.draw.line( grid_surface, GRID, (moment,0), (moment, grid_surface.get_height()), 1)

for moment in range( now, 0, -1):
    if not (now - moment) % 30:
        pygame.draw.line( grid_surface, GRID, (moment,0), (moment, grid_surface.get_height()), 1)    

# - Draw last due to overlap
pygame.draw.line( grid_surface, WHITE, ( now, 0), ( now, grid_surface.get_height()), 3)

# Init the AI
np.random.seed(0)
torch.manual_seed(0)
    
# build the model
seq = Sequence()
seq.double()
criterion = nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
#begin to train

# Generate this later
#stream = np.array([ wave_moment( basew_a, basew_f, basew_s * x) for x in range(51) ] )

#training data
T = 10
L = window_size # Was 1000
N = 20 # was 5

tx = np.zeros((N, L), 'int64')

tx[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
training_data    = np.sin( tx / 1.0 / T ).astype('float64')
training_input   = Variable(torch.from_numpy( training_data[ 0:, :-1 ]), requires_grad=False)
training_target  = Variable(torch.from_numpy( training_data[ 0:,   1:]), requires_grad=False)


#print( training_input )
#print( training_target) 

# Simple main loop
running = True
#linspace = np.linspace(50, 255, num=(grid_surface.get_width() - now), endpoint=True, dtype=int)

frame_count = 0
stream      = np.array([])
        
while running:
#    screen.fill(pygame.color.Color('white'))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    def drawpred(yi, psurface, place, place_in, color):

        # Always show
        wave_container.blit(grid_surface, (0, 0)) # height is 1/3 of total height. Centered   
        wave_container.blit(wave_surface, (0, 0))
        
        if yi is not None and psurface is not None: 
            for i, y in enumerate(yi):
                
                y = int( ( subw_a * y) + ( psurface.get_height() / 2))
                if y < 0:
                    y = 0
                elif y > psurface.get_height():
                    y = psurface.get_height()
                pygame.draw.circle(psurface, color, (i, y), 2)
                
            place_in.blit( psurface, place)
            
        #always show
        base_surface.blit(wave_container, (0, 0))
        screen.blit(base_surface, (0, 0))         # Layer surfaces onto screen
        pygame.display.flip()                     # Show it.
        
    wave_surface.scroll(-1,0)
    
    yb = wave_moment( basew_a, basew_f, basew_s * frame_count )
    pygame.draw.circle( wave_surface, WHITE, (now, int( ( yb  * basew_a) + (wave_surface.get_height()/2))), 2)
    
    #y = yb + wave_moment( subw_a, subw_f, subw_s * frame_count )
    #pygame.draw.circle( wave_surface, WAVE2, (now, int( wave_surface.get_height()/2 + yb * subw_a)), 2)

    active_wave = yb
    # Draw future
    ##########################################

    # ~~~ AI thoughts here ~~~
    # Waiting to build up training data
    if stream.size == window_size+1: #+1 for offset
        stream  = np.append(stream[1:], active_wave) # stay upto date 
        test_input   = Variable(torch.from_numpy(np.array([stream[ :-1 ]])), requires_grad=True)
        test_target  = Variable(torch.from_numpy(np.array([stream[   1:]])), requires_grad=False)
        
        def closure():
            optimizer.zero_grad()
            
            out   = seq(training_input)#, future = future) #seq(input)
            loss  = criterion(out, training_target)
            yt    = out.data.numpy()

            #out   = seq(test_input)#, future = future) #seq(input)
            #loss  = criterion(out, test_target)
            #yt    = out.data.numpy()
           
            # THINKING
            print('THINKING - loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        
        print('Thinking on itteration: ', frame_count)
        
        optimizer.step(closure)
       
        # begin to predict
        
        pred = seq(test_input, future = future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.data.numpy()[0])
        yap = pred.data.numpy()
        
        #print( len( y[0][future:  ))
        yp = int((yap[0][future:][0] * basew_a) + (wave_surface.get_height()/2) )
        if yp < 0:
            yp = 0
        elif yp > wave_surface.get_height():
            yp = wave_surface.get_height()
            
        pygame.draw.circle( wave_surface, GREEN, (now, yp), 2)
        drawpred( yap[0][future:], act_surface, (now, 0), wave_container, GREEN)
        act_surface.fill( TRANSPARENCY)
    else:
        stream  = np.append( stream, active_wave)
        drawpred( None, None, None, None, None)
        
        
    # scrolling
    
    # Needed or things get smudgy  



    frame_count += 1
