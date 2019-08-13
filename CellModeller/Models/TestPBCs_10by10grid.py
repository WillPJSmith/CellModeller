"""
Simple model for testing PBCs (2-D, x-y).
Adapted from Examples/ex1a_simpleGrowth2D.py.

W.P.J.S. August 2019
"""
import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
import numpy
import math

def setup(sim):
    # Set biophysics, signalling, and regulation models
    # 
    biophys = CLBacterium(sim, jitter_z=False, periodic=True, N_x = 10, N_y = 10, max_cells = 2**13, max_contacts = 16)

    # add some planes to coral cells
    planeWeight = 1.0
    #biophys.addPlane((0,0,0), (+1,0,0), planeWeight) 		    # left side
    #biophys.addPlane((0,0,0), (0,+1,0), planeWeight) 			# front side
    #biophys.addPlane((15.0, 0, 0), (-1,0,0), planeWeight) 		# right side
    #biophys.addPlane((0, 15.0, 0), (0,-1,0), planeWeight) 		# rear side
    #biophys.addPlane((0,0,0), (0,0,+1), planeWeight) 			# base

    # use this file for reg too
    regul = ModuleRegulator(sim)	
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)


    # Specify the initial cell and its location in the simulation
    sim.addCell(cellType=0, pos=(15, 25, 0)) 
    sim.addCell(cellType=1, pos=(35, 25, 0)) 

    if sim.is_gui:
        # Add some objects to draw the models
        from CellModeller.GUI import Renderers
        therenderer = Renderers.GLBacteriumRenderer(sim)
        sim.addRenderer(therenderer)

    sim.pickleSteps = 10

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 3.5 + random.uniform(0.0,0.5)
    # Specify growth rate of cells
    cell.growthRate = 1.0

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.iteritems():
        cell.color = [cell.cellType*0.6+0.1, 1.0-cell.cellType*0.6, 0.3]
        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 2.5 + random.uniform(0.0,0.5)
    d2.targetVol = 2.5 + random.uniform(0.0,0.5)

