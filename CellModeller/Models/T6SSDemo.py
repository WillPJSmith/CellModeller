"""
T6SSDemo.py
===========

A CellModeller model file to demonstrate use of T6SS functionality.
This file specifies the parameters that govern cell behaviour in the agent-based bacterial model.
It also specifies rules for how to initiate and update the states of cells as the simulation progresses.  
Created 16.11.19 by WPJS.

"""

import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math


"""
Input parameters.
"""
# set strain colors: 
# Type0 (T6SS+, Random attacker) = blue,
# Type1 (T6SS-, Unarmed) = green,
# Type2 (Lysing) = red
cell_colors = {0:[0.0, 0.0, 1.0],\
			   1:[0.0, 1.0, 0.0],\
			   2:[1.0, 0.0, 0.0]}
			   
# T6SS cost, firing and response parameters
kFiring0 = 50.0											# The T6SS firing rate used by Type0 cells [firings per cell per h]
HitThresh = 1											# The number of T6SS hits required to kill a Type1 cell
LysisDelay = 5											# The time delay between the hit threshold being reached and target cell lysis [timesteps]
CostPerRate = 0.001										# The weapon cost per unit firing rate: 0.001 * 50.0 = 5% growth rate reduction

# T6SS numerical parameters
max_population = 2**15									# The largest cell population for which we'll allocate memory
number_of_types = 2										# The number of living cell types for which we'll separately compute hit tables
CDKParams = dict(max_cells=max_population,
			 max_needles=max_population * 16, 
			 max_sqs=64**2,
			 grid_spacing=5.0,
			 rand_seed=None,
			 num_types=2)


def setup(sim):
	"""
	This function is called when the simulation is initiated.
	It specifies which modules are to be used in the simulation, how cells are initially placed, and when to generate output.
	"""

	global TimeStep

	# Use biophys, regulation and T6SS modules
	biophys = CLBacterium(sim, jitter_z=False, max_cells=max_population)
	regul = ModuleRegulator(sim)					
	sim.init(biophys, regul, None, None, CDKParams)

	# Specify initial cell locations in the simulation
	sim.addCell(cellType=0, pos=(-10,0,0)) 
	sim.addCell(cellType=1, pos=(10,0,0)) 

	# Add some objects to render the cells in the GUI
	therenderer = Renderers.GLBacteriumRenderer(sim)
	sim.addRenderer(therenderer)
	TimeStep = sim.dt
	
	# print output (pickle) files every 10 timesteps
	sim.pickleSteps = 10


def init(cell):
	"""
	This function specifies states of cells when they are first created.
	"""
	cell.targetVol = 2.5 + random.uniform(0.0,0.5)		# volume cells must reach before dividing [microns cubed]
	cell.growthRate = 1.0								# initial growth rate in h^-1
	cell.countdown = LysisDelay							# steps remaining before lysis, once lysis countdown is triggered
	cell.num_hits = [0]*number_of_types					# table of T6SS hits taken from different cell types
	cell.num_firings = 0								# number of T6SS firings this step
	cell.color = cell_colors[cell.cellType]				# cell's color when rendered on the GUI


def update(cells):
	"""
	This function is called each simulation step, and specifies how to update cells' states.
	""" 
	global TimeStep

	# go through the list of cells
	for (id, cell) in cells.iteritems():

		# set growth rates (type 0 cells pay a cost; dead cells cannot grow)
		if cell.cellType==2:
			cell.growthRate = 0.0
		else:
			f = (cell.num_firings / float(TimeStep)) * CostPerRate
			cell.growthRate = max(1-f, 0.0)

		# type 1 cells (Unarmed) are vulnerable to type 0 cells (T6SS+ Random attackers)
		if cell.num_hits[0] >= HitThresh and cell.cellType==1:
			cell.cellType = 2
			cell.growthRate = 0
			cell.divideFlag = False
			cell.color = cell_colors[cell.cellType]

		# cells intoxicated by T6SS lyse after a fixed countdown
		if cell.cellType==2:
			if cell.countdown < 1:
				cell.deathFlag = True
			else:
				cell.countdown-=1

		# Non-lysing cells can divide symmetrically once they reach a volume threshold
		if cell.volume > cell.targetVol and not cell.cellType==2:
			cell.asymm = [1,1]
			cell.divideFlag = True


def divide(parent, d1, d2):
	"""
	This function specifies how to change daughter (d1,d2) cell states when a parent cell divides.
	""" 
	d1.targetVol = 2.5 + random.uniform(0.0,0.5)
	d2.targetVol = 2.5 + random.uniform(0.0,0.5)


def kill(cell):
	"""
	This function specifies how to change a cell's state when it dies.
	""" 
	# define what happens to a cell's state when it dies
	cell.growthRate = 0.0   	# dead cells can't grow any more
	cell.divideFlag = False		# dead cells can't divide
    

def FiringProtocol(cells):
	"""
	This function controls which cells should fire T6SS needles this step.
	"""
	global TimeStep

	# the two cell types fire with independent expectation rates
	Lambda0 = float(kFiring0 * TimeStep)
	for (id, cell) in cells.iteritems():
		type = cell.cellType
		if type==0:
			num_firings = numpy.random.poisson(Lambda0,1)[0]
		else:
			num_firings = 0
		cell.num_firings = num_firings

