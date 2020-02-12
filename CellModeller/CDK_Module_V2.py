"""
CDK_Module_V2.py
=============

A prototype class structure for modelling contact-dependent killing and inhibion (CDK) using the CellModeller colony simulator.
Here, we are primarily concerned with modelling interference competition using the bacterial T6SS (Type-VI secretion system).
However, similar code could later be adapted to study other contact-dependent phenomena e.g. toxin-on-stick; conjugation etc.

Created by W.P.J.S on 02.06.17 (V0.0)
Updated by W.P.J.S on 20.10.17 (V0.5)

Moved to development copy for merge with PBC system
W.P.J.S on 04.11.19 (V?.?)

"""

import numpy
import math
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import vec
import vtk


class CDK_Module:

	def __init__(self, CDKParams):
		"""
		intialise the module
		I'll hard-wire some of the key parameters for now - later we'll take these parameters from the model file.
		"""
		print 'Initialising CDK module (V2.0)'

		# array parameters	
		self.max_cells = CDKParams['max_cells']
		self.max_needles = CDKParams['max_needles']
		self.max_sqs = CDKParams['max_sqs']
		self.grid_spacing = CDKParams['grid_spacing']
		self.num_types = CDKParams['num_types']
	
		# setup
		self.context = None
		self.queue = None
		self.program = None
		self.saveNeedles = False
		self.n_needles = 0
		self.n_cells = 0

		# if specified, seed random generator for repeatability
		if 'rand_seed' in CDKParams:
			numpy.random.seed(seed=CDKParams['rand_seed'])

		# if specified, set periodic mode. 
		if 'periodic' in CDKParams:
			self.periodic = CDKParams['periodic']
		else:
			self.periodic = False
			
		# periodic mode: lock the grid to the dimensions specified (N_x by N_y)	
		if self.periodic:
			if 'N_x' and 'N_y' in CDKParams:
				self.N_x = CDKParams['N_x']
				self.N_y = CDKParams['N_y']
				self.set_periodic_grid()
			else:
				print "Error: please specify grid dimensions (N_x>0, N_y>0) within CDKParams, in model file."
				exit(0)


	def init_cl(self):
		"""
		initialise OpenCL program object.
		/!\ assumes queue and context have already been set by simulator.
		"""
		
		# intialise kernels
		kernel_path = '/anaconda2/envs/cellmodeller/lib/python2.7/site-packages/CellModeller/CDK_Module_V2.cl'
		kernel_src = open(kernel_path, 'r').read()
		self.program = cl.Program(self.context, kernel_src).build(cache_dir=False)

	
	def init_data(self):
		"""
		initialise data arrays for frontend and backend (dev=device).
		/!\ assumes queue and context have already been set by simulator.
		"""
		
		# array shapes
		cell_geom = (self.max_cells,)
		needle_geom = (self.max_needles,)
		sq_geom = (self.max_sqs,)
		
		# LOCAL
		self.P1s = numpy.zeros((self.max_needles,3), numpy.float32)
		self.P2s = numpy.zeros((self.max_needles,3), numpy.float32)
		
		# DEVICE and LOCAL
		# cell config data
		self.cell_centroids = numpy.zeros(cell_geom, vec.float4)
		self.cell_centroids_dev = cl_array.zeros(self.queue, cell_geom, vec.float4)
		self.cell_dirs = numpy.zeros(cell_geom, vec.float4)
		self.cell_dirs_dev = cl_array.zeros(self.queue, cell_geom, vec.float4)
		self.cell_lens = numpy.zeros(cell_geom, numpy.float32)
		self.cell_lens_dev = cl_array.zeros(self.queue, cell_geom, numpy.float32)
		self.cell_rads = numpy.zeros(cell_geom, numpy.float32)
		self.cell_rads_dev = cl_array.zeros(self.queue, cell_geom, numpy.float32)

		# needle config_data
		self.needle_centroids = numpy.zeros(needle_geom, vec.float4) 
		self.needle_centroids_dev = cl_array.zeros(self.queue, needle_geom, vec.float4)
		self.needle_dirs = numpy.zeros(needle_geom, vec.float4) 
		self.needle_dirs_dev = cl_array.zeros(self.queue, needle_geom, vec.float4)
		self.needle_lens = numpy.zeros(needle_geom, numpy.float32)
		self.needle_lens_dev = cl_array.zeros(self.queue, needle_geom, numpy.float32)

		# gridding data
		self.cell_sqs = numpy.zeros(cell_geom, numpy.int32)
		self.cell_sqs_dev = cl_array.zeros(self.queue, cell_geom, numpy.int32)
		self.needle_sqs = numpy.zeros(needle_geom, numpy.int32)
		self.needle_sqs_dev = cl_array.zeros(self.queue, needle_geom, numpy.int32)
		self.sorted_needle_ids = numpy.zeros(needle_geom, numpy.int32)
		self.sorted_needle_ids_dev = cl.array.zeros(self.queue, needle_geom, numpy.int32)
		self.needle_sq_inds = numpy.zeros(sq_geom, numpy.int32)
		self.needle_sq_inds_dev = cl.array.zeros(self.queue, sq_geom, numpy.int32)
		
		# firing protocols
		self.cell_firings = numpy.zeros(cell_geom, numpy.int32)	
		self.firing_pattern = numpy.zeros(cell_geom, numpy.int32)	
		
		# new stuff which I'll categorise later
		self.needle_hits_axis = numpy.zeros(needle_geom, vec.float4)
		self.needle_hits_axis_dev = cl_array.zeros(self.queue, needle_geom, vec.float4)
		self.needle_hits_surf = numpy.zeros(needle_geom, vec.float4)
		self.needle_hits_surf_dev = cl_array.zeros(self.queue, needle_geom, vec.float4)

		self.needle_to_cell_idx = numpy.zeros(needle_geom, numpy.int32)
		self.cell_hits = numpy.zeros(cell_geom, numpy.int32)
		self.cell_hits_dev = cl_array.zeros(self.queue, cell_geom, numpy.int32)
		self.needle_hits = numpy.zeros(needle_geom, numpy.int32)
		self.needle_hits_dev = cl_array.zeros(self.queue, needle_geom, numpy.int32)
		
		self.needle_to_cell = numpy.zeros(needle_geom, numpy.int32)
		self.needle_to_cell_dev = cl_array.zeros(self.queue, needle_geom, numpy.int32)
		self.cell_types = numpy.zeros(cell_geom, numpy.int32)


	def init_data_periodic(self):
		"""
		additional arrays for periodic simulations
		"""
		# Connectivity of periodic grid
		self.needle_sq_neighbour_inds = numpy.zeros((self.n_sqs*9,), numpy.int32)
		self.needle_sq_neighbour_inds_dev = cl_array.zeros(self.queue, (self.n_sqs*9,), numpy.int32)
		self.needle_sq_neighbour_offset_inds = numpy.zeros((self.n_sqs*9,), numpy.int32)
		self.needle_sq_neighbour_offset_inds_dev = cl_array.zeros(self.queue, (self.n_sqs*9,), numpy.int32)

		# offset vectors for computing cell images
		self.offset_vecs = numpy.zeros((9,), vec.float4)
		self.offset_vecs_dev = cl_array.zeros(self.queue, (9,), vec.float4)


	def set_periodic_grid(self):
		"""
		Initiate a grid for domains with fixed dimensions. 
		This is a faster alternative to calling update_grid() every substep. 
		"""
		print "Setting up periodic domain now..."

		# Periodic grids are user-specified; don't need to compute a grid to enclose all cells
		self.min_x_coord = 0.0
		self.min_y_coord = 0.0
		self.max_x_coord = self.N_x * self.grid_spacing
		self.max_y_coord = self.N_y * self.grid_spacing

		self.grid_x_min = 0
		self.grid_x_max = self.N_x

		self.grid_y_min = 0
		self.grid_y_max = self.N_x

		self.n_sqs = self.N_x * self.N_y
		print "		##  working with a fixed %i-by-%i grid (%i squares)..." % (self.N_x, self.N_y, self.n_sqs)


	def set_periodic_grid_connectivity(self):
		"""
		Define the perodic Moore neighbourhood of each grid square
		"""
		print "		##  defining grid connectivity..."
		
		# index the grid squares
		m = self.N_y; n = self.N_x
		M = numpy.arange(m*n).reshape((m, n))
		
		# create an expanded grid with periodic border
		Mb = numpy.zeros((m+2,n+2), numpy.int32);
		Mb[1:m+1,1:n+1] = M;
		Mb[1:m+1,0] = M[:,n-1];
		Mb[1:m+1,n+1] = M[:,0];
		Mb[0,1:n+1] = M[m-1,:];
		Mb[m+1,1:n+1] = M[0,:];
		Mb[0,0] = M[m-1,n-1];
		Mb[0,n+1] = M[m-1,0];
		Mb[m+1,0] = M[0,n-1];
		Mb[m+1,n+1] = M[0,0];
		
		print 'Mb is:', Mb

		# produce an analagous grid containing offset vector indices
		O = 4*numpy.ones((m,n), numpy.int32)

		# add a periodic border to O
		Ob = numpy.zeros((m+2,n+2), numpy.int32);
		Ob[1:m+1,1:n+1] = O;
		Ob[:,0] = 3;
		Ob[:,n+1] = 5;
		Ob[0,:] = 1;
		Ob[m+1,:] = 7;
		Ob[0,n+1] = 2;
		Ob[m+1,0] = 6;
		Ob[0,0] = 0;
		Ob[m+1,n+1] = 8; 
		
		# for each square in the grid, extract an ordered list of neighbour squares and a list of the corresponding offsets
		Ns = numpy.zeros((m*n,9), numpy.int32);
		Os = numpy.zeros((m*n,9), numpy.int32);
		c = 0;
		for i in range(1,m+1):
			for j in range(1,n+1):
				Ns[c,:] = [Mb[i-1,j-1],Mb[i-1,j],Mb[i-1,j+1],\
						   Mb[i,  j-1],Mb[i,  j],Mb[i,  j+1],\
						   Mb[i+1,j-1],Mb[i+1,j],Mb[i+1,j+1]]
				   
				Os[c,:] = [Ob[i-1,j-1],Ob[i-1,j],Ob[i-1,j+1],\
						   Ob[i,  j-1],Ob[i,  j],Ob[i,  j+1],\
						   Ob[i+1,j-1],Ob[i+1,j],Ob[i+1,j+1]]
				c+=1;

		self.needle_sq_neighbour_inds = Ns.flatten()
		self.needle_sq_neighbour_offset_inds = Os.flatten()
		self.needle_sq_neighbour_inds_dev.set(self.needle_sq_neighbour_inds)
		self.needle_sq_neighbour_offset_inds_dev.set(self.needle_sq_neighbour_offset_inds)
		
		
	def set_periodic_offsets(self):
		"""
		Precompute cell offset vectors for a given cuboidal domain.
		"""
		print "		##  Setting up offsets..."
		L_x = self.max_x_coord - self.min_x_coord
		L_y = self.max_y_coord - self.min_y_coord
		self.offset_vecs[0] = (-1.0*L_x, -1.0*L_y, 0.0, 0.0)
		self.offset_vecs[1] = (0.0, -1.0*L_y, 0.0, 0.0)
		self.offset_vecs[2] = (+1.0*L_x, -1.0*L_y, 0.0, 0.0)
		self.offset_vecs[3] = (-1.0*L_x, 0.0, 0.0, 0.0)
		self.offset_vecs[4] = (0.0, 0.0, 0.0, 0.0)
		self.offset_vecs[5] = (+1.0*L_x, 0.0, 0.0, 0.0)
		self.offset_vecs[6] = (-1.0*L_x, +1.0*L_y, 0.0, 0.0)
		self.offset_vecs[7] = (0.0, +1.0*L_y, 0.0, 0.0)
		self.offset_vecs[8] = (+1.0*L_x, +1.0*L_y, 0.0, 0.0)
		self.offset_vecs_dev.set(self.offset_vecs)


	def step(self, cell_states):
		"""
		Carry out ckd procedure for a simulation step
		"""
		# extra step: clear everything
		self.init_data()
		
		# Load current cell states to module
		self.LoadFromCellstates(cell_states)
		
		# Read number of firings off for each cell
		self.GetCellFirings(cell_states)
		
		# carry out firings that have been requested, then run hit detection
		if self.n_needles > 0:

			self.ExecuteFiringProtocol()
			self.GetCellHits(cell_states)
			
			
	def step_withRetaliation(self, cell_states):
		"""
		Carry out ckd procedure for a simulation step, with consecutive firing rounds for constituitive and retaliatory firing strategies 
		"""
		### ROUND ONE ###
		# extra step: clear everything and load the current cell config
		self.init_data()		
		self.LoadFromCellstates(cell_states)
		
		# Read number of firings off for each cell
		FiringType = 0
		self.GetCellFirings_RoundBased(cell_states, FiringType)
		
		# carry out firings that have been requested, then run hit detection
		if self.n_needles > 0:

			# carry out firing
			self.ExecuteFiringProtocol()
			
			# log hits including data for retaliation (written to cell states during hit updates)
			self.GetCellHits_withRetaliation(cell_states)
			
		# copy over hit data and then clear arrays
		self.needle_to_cell_old = self.needle_to_cell					# which needles hit which cells (n.idx -> c.idx)
		self.needle_hits_old = self.needle_hits							# which needles hit 			(n.idx -> bool)
		self.needle_hits_axis_old = self.needle_hits_axis				# axial hit points 				(n.idx -> float4)
		self.needle_hits_surf_old = self.needle_hits_surf				# surface hit points			(n.idx -> float4)

		### ROUND TWO ###
		# again, load current config
		self.init_data()
		self.LoadFromCellstates(cell_states)

		FiringType = 1
		self.GetCellFirings_RoundBased(cell_states, FiringType)

		if self.n_needles > 0: # then execute firing protocol
		
			self.GetNeedlesForCells_Retaliatory()
			#self.GetNeedlesForCells()
			self.UpdateNeedleData()
			self.UpdateCentroidGrid()
			self.BinCellsAndNeedles()
			self.SortNeedles()
			self.FindCellNeedleContacts()
			
			# log hits including data for retaliation (written to cell states during hit updates)
			self.GetCellHits_withRetaliation(cell_states)
			
		# after retaliation, reset num_firings to 0 for retaliator cells
		# this is necessary to exclude [accidental] counter-retaliation, for which we do not yet pass enough data to manage.
		# TODO: fix this to allow for accidental counter-retaliation (i.e. Type 2 cells shoot in response to hits by Type 2s.)
		for (cid,cs) in cell_states.items():
			if cs.cellType == 1:
				cs.num_firings_old = cs.num_firings
				cs.num_firings = 0
				
			
		
	def GetCellFirings(self, cell_states):
		"""
		Read firing counts from cell_states dict, writing to our local array
		"""
		for (cid,cs) in cell_states.items():
			i = cs.idx
			self.cell_firings[i] = cs.num_firings
			self.firing_pattern[i] = cs.cellType
			
		# count number of needles for this round		
		self.n_needles = sum(self.cell_firings)
		
		
	def GetCellFirings_RoundBased(self, cell_states, firingType):
		"""
		Read firing counts from cell_states dict, writing to our local array
		This time we filter cells by their type, allowing firing only if type matches the firing type for this round.
		"""
		for (cid,cs) in cell_states.items():
			i = cs.idx
			# not currently used
			self.firing_pattern[i] = cs.cellType
			
			# if cellType matches firing type, count their firings
			if cs.cellType == firingType:
				self.cell_firings[i] = cs.num_firings
			else:
				self.cell_firings[i] = 0
				
		# count number of needles for this round		
		self.n_needles = sum(self.cell_firings)
		
		
	def GetCellHits(self, cell_states):
		"""
		Write hit counts to cell_states dict from our local array
		"""
		
		# extract a complete list of cells that got hit
		cell_hit_list = self.needle_to_cell[self.needle_hits==1]
		
		# extract the indices of the needles responsible for those hits
		hit_to_needle_index = numpy.nonzero(self.needle_hits==1)[0]

		# go through the population and update hit counts from cells of each type
		for (cid,cs) in cell_states.items():
			i = cs.idx
			if cs.cellType == -1:
				continue
			for type in range(0,self.num_types):
				num_hits_by_type = sum(self.cell_types[self.needle_to_cell_idx[hit_to_needle_index[cell_hit_list==i]]]==type)
				cs.num_hits[type] += num_hits_by_type				
				#print 'cell %i (type %i) hit %i times by type %i cells' % (i, cs.cellType, num_hits_by_type, type)
				
				
	def GetCellHits_withRetaliation(self, cell_states):
		"""
		Write hit counts to cell_states dict from our local array
		Also, update firing counts for retaliator cells
		"""
		
		# extract a complete list of cells that got hit
		cell_hit_list = self.needle_to_cell[self.needle_hits==1]
		
		# extract the indices of the needles responsible for those hits
		hit_to_needle_index = numpy.nonzero(self.needle_hits==1)[0]

		# go through the population and update hit counts from cells of each type
		for (cid,cs) in cell_states.items():
			i = cs.idx
			total_hits = 0
			for type in range(0,self.num_types):
				num_hits_by_type = sum(self.cell_types[self.needle_to_cell_idx[hit_to_needle_index[cell_hit_list==i]]]==type)
				cs.num_hits[type] += num_hits_by_type
				total_hits += num_hits_by_type
			
			# for retaliatory cell, set number of firings to be the number of hits sustained this round
			if cs.cellType == 1:
				#cs.num_firings = total_hits	
				cs.num_firings = 2*total_hits # 2 tits per tat	
				

	def ExecuteFiringProtocol(self):
		"""
		Draw needles according to current firing protocol, and compute all resulting needle-cell contacts.
		"""
		# Compute needles according to current firing protocol, and set these data
		self.GetNeedlesForCells()
		self.UpdateNeedleData()

		# fit a grid around these points, reporting results (non-periodic mode only)
		if not self.periodic:
			self.UpdateCentroidGrid()

		# Bin cells using current grid
		self.BinCellsAndNeedles()

		# Sort cells by square, so we can look up square contents
		self.SortNeedles()

		# Check for hits between cells and needles (write to cell_hits array)
		self.FindCellNeedleContacts()


	def UpdateNeedleData(self):
		"""
		Using current copy of P1s, P2s, update needle centroid and dir data
		"""
		# go through needle list and write to our arrays
		for i in range(0,self.n_needles):
			P1 = self.P1s[i]
			P2 = self.P2s[i]
			
			pos = 0.5*(P1 + P2)
			dir = P2 - P1
			len = numpy.linalg.norm(dir)
			dir /= float(len)
			
			self.needle_centroids[i][0] = pos[0]
			self.needle_centroids[i][1] = pos[1]
			self.needle_centroids[i][2] = pos[2]
			
			self.needle_dirs[i][0] = dir[0]
			self.needle_dirs[i][1] = dir[1]
			self.needle_dirs[i][2] = dir[2]
			
			self.needle_lens[i] = len
			
		# Copy data from local to device
		self.SetNeedles()
			
			
	def SetCells(self):
		"""
		Copy local cell data over to device
		"""
		# set cell data
		self.cell_centroids_dev.set(self.cell_centroids)
		self.cell_dirs_dev.set(self.cell_dirs)
		self.cell_rads_dev.set(self.cell_rads)
		self.cell_lens_dev.set(self.cell_lens)
		
		
	def SetNeedles(self):
		"""
		Copy local needle data over to device
		"""
		# set needle data
		self.needle_centroids_dev.set(self.needle_centroids)
		self.needle_dirs_dev.set(self.needle_dirs)
		self.needle_lens_dev.set(self.needle_lens)

	"""
	HIT DETECTION FUNCTIONS
	"""	
		
	def FindCellNeedleContacts(self):
		"""
		Based on current grid and centroid allocation, concurrently check each cell for needle contacts.
		Limit contact search to needles in a focal cell's Moore neighbourhood, performing a bounding spheres test to exclude geometrically impossible contacts
		"""
		# set spatial sorting data on device 
		self.cell_sqs_dev.set(self.cell_sqs)
		self.needle_sqs_dev.set(self.needle_sqs)
		self.sorted_needle_ids_dev.set(self.sorted_needle_ids)
		self.needle_sq_inds_dev.set(self.needle_sq_inds)
		
		# clear the hit count from last step
		self.cell_hits_dev = cl.array.zeros_like(self.cell_hits_dev)
		self.needle_hits_dev = cl.array.zeros_like(self.needle_hits_dev)
		self.needle_to_cell_dev = cl.array.zeros_like(self.needle_to_cell_dev)
		self.needle_hits_axis_dev = cl.array.zeros_like(self.needle_hits_axis_dev)
		self.needle_hits_surf_dev = cl.array.zeros_like(self.needle_hits_surf_dev)

		# call hit detection kernel
		if self.periodic:
			self.program.check_hits_periodic(self.queue,
								   (int(self.n_cells),),
								   None,
								   numpy.int32(self.n_needles),
								   numpy.int32(self.grid_x_min),
								   numpy.int32(self.grid_x_max),
								   numpy.int32(self.grid_y_min),
								   numpy.int32(self.grid_y_max),
								   numpy.int32(self.n_sqs),
								   self.cell_sqs_dev.data,
								   self.needle_sqs_dev.data,
								   self.sorted_needle_ids_dev.data,
								   self.needle_sq_inds_dev.data,
								   self.needle_sq_neighbour_inds_dev.data,
								   self.needle_sq_neighbour_offset_inds_dev.data,
								   self.offset_vecs_dev.data,
								   self.cell_centroids_dev.data,
								   self.cell_dirs_dev.data,
								   self.needle_centroids_dev.data,
								   self.needle_dirs_dev.data,
								   self.needle_lens_dev.data,
								   self.cell_lens_dev.data,
								   self.cell_rads_dev.data,
								   self.cell_hits_dev.data,
								   self.needle_hits_dev.data,
								   self.needle_to_cell_dev.data,
								   self.needle_hits_axis_dev.data,
								   self.needle_hits_surf_dev.data).wait()
		else:
			self.program.check_hits(self.queue,
								   (int(self.n_cells),),
								   None,
								   numpy.int32(self.n_needles),
								   numpy.int32(self.grid_x_min),
								   numpy.int32(self.grid_x_max),
								   numpy.int32(self.grid_y_min),
								   numpy.int32(self.grid_y_max),
								   numpy.int32(self.n_sqs),
								   self.cell_sqs_dev.data,
								   self.needle_sqs_dev.data,
								   self.sorted_needle_ids_dev.data,
								   self.needle_sq_inds_dev.data,
								   self.cell_centroids_dev.data,
								   self.cell_dirs_dev.data,
								   self.needle_centroids_dev.data,
								   self.needle_dirs_dev.data,
								   self.needle_lens_dev.data,
								   self.cell_lens_dev.data,
								   self.cell_rads_dev.data,
								   self.cell_hits_dev.data,
								   self.needle_hits_dev.data,
								   self.needle_to_cell_dev.data,
								   self.needle_hits_axis_dev.data,
								   self.needle_hits_surf_dev.data).wait()
					   
		# retrieve hit counts from device
		self.cell_hits = self.cell_hits_dev.get()
		self.needle_hits = self.needle_hits_dev.get()
		self.needle_to_cell = self.needle_to_cell_dev.get()
		
		# retrieve hit data
		self.needle_hits_axis = self.needle_hits_axis_dev.get()
		self.needle_hits_surf = self.needle_hits_surf_dev.get()

		
	def UpdateCentroidGrid(self):
		"""Update our grid_(x,y)_min, grid_(x,y)_max, and n_sqs.
		Assumes that our copy of cell_centers is current.
		/!\ only works in two dimensions at the moment. 
		"""
		cell_coords = self.cell_centroids.view(numpy.float32).reshape((self.max_cells, 4))
		needle_coords = self.needle_centroids.view(numpy.float32).reshape((self.max_needles, 4))
		grid_spacing = self.grid_spacing

		# check cell_coords
		x_coords = cell_coords[:,0]
		grid_x_min_cell = int(math.floor(x_coords.min() / grid_spacing))
		grid_x_max_cell = int(math.ceil(x_coords.max() / grid_spacing))
		y_coords = cell_coords[:,1]
		grid_y_min_cell = int(math.floor(y_coords.min() / grid_spacing))
		grid_y_max_cell = int(math.ceil(y_coords.max() / grid_spacing))
	
		# check needle coords
		x_coords = needle_coords[:,0]
		grid_x_min_needle = int(math.floor(x_coords.min() / grid_spacing))
		grid_x_max_needle = int(math.ceil(x_coords.max() / grid_spacing))
		y_coords = needle_coords[:,1]
		grid_y_min_needle = int(math.floor(y_coords.min() / grid_spacing))
		grid_y_max_needle = int(math.ceil(y_coords.max() / grid_spacing))
	
		# determine maximum and minimum grid square numbers
		self.grid_x_min = min(grid_x_min_cell, grid_x_min_needle)
		self.grid_x_max = max(grid_x_max_cell, grid_x_max_needle)
		self.grid_y_min = min(grid_y_min_cell, grid_y_min_needle)
		self.grid_y_max = max(grid_y_max_cell, grid_y_max_needle)
	
		# if everything's in just one grid row or column, make sure there's at least one row or column
		if self.grid_x_min == self.grid_x_max:
			self.grid_x_max += 1

		if self.grid_y_min == self.grid_y_max:
			self.grid_y_max += 1

		self.n_sqs = (self.grid_x_max-self.grid_x_min)*(self.grid_y_max-self.grid_y_min)
			
			
	def BinCellsAndNeedles(self):
		"""
		Given grid measurements, use an OpenCL kernel to assign cells and needles to grid squares
		"""
		# call the kernel individually for cell centroids
		self.program.bin_centroids(self.queue,
							   (self.n_cells,),
							   None,
							   numpy.int32(self.grid_x_min),
							   numpy.int32(self.grid_x_max),
							   numpy.int32(self.grid_y_min),
							   numpy.int32(self.grid_y_max),
							   numpy.float32(self.grid_spacing),
							   self.cell_centroids_dev.data,
							   self.cell_sqs_dev.data).wait()
					   
		# call the kernel individually for needle centroids
		self.program.bin_centroids(self.queue,
							   (self.n_needles,),
							   None,
							   numpy.int32(self.grid_x_min),
							   numpy.int32(self.grid_x_max),
							   numpy.int32(self.grid_y_min),
							   numpy.int32(self.grid_y_max),
							   numpy.float32(self.grid_spacing),
							   self.needle_centroids_dev.data,
							   self.needle_sqs_dev.data).wait()
	
		# retrieve data from device
		self.cell_sqs = self.cell_sqs_dev.get()
		self.needle_sqs = self.needle_sqs_dev.get()			   
		
		
	def SortNeedles(self):
		"""
		Modified version of CM4 sort_cells: create lists of needles held in each grid square
		"""
		self.sorted_needle_ids = numpy.zeros_like(self.sorted_needle_ids)
		self.sorted_needle_ids.put(numpy.arange(self.n_needles), numpy.argsort(self.needle_sqs[:self.n_needles]))

		# find the start of each sq in the list of sorted cell ids
		sorted_sqs = numpy.sort(self.needle_sqs[:self.n_needles])
		
		# clear the needle square data, then write to that array
		self.needle_sq_inds = numpy.zeros_like(self.needle_sq_inds)
		self.needle_sq_inds.put(numpy.arange(self.n_sqs), numpy.searchsorted(sorted_sqs, numpy.arange(self.n_sqs), side='left'))
		

	"""
	FUNCTIONS FOR DRAWING NEEDLES
	"""

	def GetNeedlesForCells(self):
		"""
		Go through the cell configuration and draw needles according to the firing protocol
		"""

		# reformat vector data so we can easily get at it
		centers = self.cell_centroids.view(numpy.float32).reshape((self.max_cells, 4))
		dirs = self.cell_dirs.view(numpy.float32).reshape((self.max_cells, 4))
		
		# go through cells and draw appropriate numbers of vectors per cell
		write_ind = 0
		for i in range(0,self.n_cells):
			
			p_i = centers[i,0:3]
			a_i = dirs[i,0:3]
			l_i = self.cell_lens[i]
			r_i = self.cell_rads[i]
			N = self.cell_firings[i]
			
			if N > 0:
			
				# draw needles according to a particular firing strategy: 
				# 1) random firing
				#if self.firing_pattern[i] == 1:
				# I think actually we need to distinguish between initial and response firing at a higher stage,
				# so that we can do the two parts consecutively each simulation step.
				(this_P1s, this_P2s) = self.DrawFiringVectorsForCapsule_Random(p_i, a_i, l_i, r_i, N)	
				
				# write the points associated with these needles to array
				self.P1s[write_ind:write_ind+N,:] = this_P1s
				self.P2s[write_ind:write_ind+N,:] = this_P2s
				
				# log the cell that produced these needles
				self.needle_to_cell_idx[write_ind:write_ind+N] = i
				
				# update the write index
				write_ind += N


	def GetNeedlesForCells_Retaliatory(self):
		"""
		Go through the cell configuration and draw needles according to the firing protocol
		# TODO: merge this with the standard GetNeedlesForCells function
		# NOTE: for this to work, need to save copies
		needle_to_cell_old
		needle_hits_old
		needle_hits_axis_old
		needle_hits_surf_old
		"""

		# reformat vector data so we can easily get at it
		centers = self.cell_centroids.view(numpy.float32).reshape((self.max_cells, 4))
		dirs = self.cell_dirs.view(numpy.float32).reshape((self.max_cells, 4))
		
		# collect a list of cell hits from the previous round of firing
		cell_hit_list = self.needle_to_cell_old[self.needle_hits_old==1]
		hit_to_needle_index = numpy.nonzero(self.needle_hits_old==1)[0]

		# go through cells and draw appropriate numbers of vectors per cell
		write_ind = 0
		for i in range(0,self.n_cells):
			
			# always needed
			r_i = self.cell_rads[i]
			N = self.cell_firings[i]
			
			# needed if retaliation strategy is random
			p_i = centers[i,0:3]
			a_i = dirs[i,0:3]
			l_i = self.cell_lens[i]

			
			if N > 0:
			
				# work out which hit locations correspond to this cell
				my_needle_inds = hit_to_needle_index[cell_hit_list==i]
				p_axis = self.needle_hits_axis_old[my_needle_inds]
				p_surf = self.needle_hits_surf_old[my_needle_inds]

				# draw needles according to a TFT firing strategy:
				# OPTION 1. spatial sensing
				(this_P1s, this_P2s) = self.DrawFiringVectorsForCapsule_Retaliator(r_i, p_axis, p_surf, N)
				
				# OPTION 2: random
				#(this_P1s, this_P2s) = self.DrawFiringVectorsForCapsule_Random(p_i, a_i, l_i, r_i, N)
				
				# write the points associated with these needles to array
				self.P1s[write_ind:write_ind+N,:] = this_P1s
				self.P2s[write_ind:write_ind+N,:] = this_P2s
				
				# log the cell that produced these needles
				self.needle_to_cell_idx[write_ind:write_ind+N] = i
				
				# update the write index
				write_ind += N
		
		
	def DrawFiringVectorsForCapsule_Retaliator(self, r_i, p_axis, p_surf, N):
		"""
		Prototype for spatial-sensing retaliator.
		Given a non-empty set of M axis and surface points (p_axis, p_surface), randomly choose N of them 
		"""
				
		# Preallocate space for N firing vectors in 3D. 
		P1s = numpy.zeros((N,3))
		P2s = numpy.zeros((N,3))
		M = len(p_axis)	
	
		# prepare a randomised selection of these points, using each point at least once (if N >= M) OR using a random subselection (N < M)
		rand_inds = numpy.tile(numpy.random.permutation(M),(N / M)+1)[0:N]
	
		for i in range(0,N):
		
			# draw axial and surface points
			axis_point = numpy.array([p_axis[rand_inds[i]][j] for j in range(3)])
			surf_point = numpy.array([p_surf[rand_inds[i]][j] for j in range(3)])
		
			# get needle start and end points
			needle_len = r_i
			a_r = (surf_point - axis_point)
			a_r /= numpy.linalg.norm(a_r)
		
			# get needle start and end points
			P1 = surf_point 
			P2 = surf_point + needle_len*a_r
	
			# log this information
			P1s[i] = P1
			P2s[i] = P2

		return P1s, P2s	

		
			
	def DrawFiringVectorsForCapsule_Random(self, p_i, a_i, l_i, r_i, N):
		"""
		Return a set of N T6SS needle vectors (P1s,P2s) for a single cell, drawing needle baseplate sites randomly over the cell's surface area.
		We'll use the convention that P1 is the surface point.
		"""
		# /!\ TODO: would be more efficient to call for Nc cap points, Nb cyl points, and run cap and cyl functions once each for Nc, Nb needles.

		# Testing: lock rng here so that each cell always draws the same set of needles
		# Seed = 1
# 		numpy.random.seed(Seed)

		# Preallocate space for N firing vectors in 3D. 
		P1s = numpy.zeros((N,3))
		P2s = numpy.zeros((N,3))

		for i in range(0,N):
		
			# randomly decide if this is a cap point, or a cylinder point
			p_cyl = float(l_i) / float(l_i + 2*r_i)
			#p_cyl = -1 # /!\ TODO temporarily force cap points for testing
			dice = numpy.random.uniform(0,1,1)
			if dice > p_cyl:
				(p_a, p_r) = self.DrawACapPoint(p_i, a_i, l_i)
			else:
				(p_a, p_r) = self.DrawACylPoint(p_i, a_i, l_i)
	
			# convert from axis point to surface point
			P1 = p_a + r_i*p_r;
	
			# compute location of needle tip
			P2 = p_a + p_r;

			# log this information
			P1s[i] = P1
			P2s[i] = P2

		return P1s, P2s


	def DrawACapPoint(self, p_i, a_i, l_i):
		"""
		Draws a vector with axial origin p_a and radial direction p_r from one of a capsule's caps.
		The capsule's configuration is defined by position p_i, unit axis a_i, and length l_i.
		"""
	
		r = 1.0 # cap radius
	
		# draw point angles at random, then convert to cartesial coordinates
		theta = 2.0*numpy.pi*numpy.random.uniform(0,1,1)                 	 # draw a polar angle
		phi = numpy.arcsin(2.0*numpy.random.uniform(0,1,1)-1.0)              # draw an azimuthal angle
		x_c = numpy.multiply(r*numpy.cos(phi), numpy.cos(theta))
		y_c = numpy.multiply(r*numpy.cos(phi), numpy.sin(theta))
		z_c = r*numpy.sin(phi)
		
		# decide if this vector is in the top cap or in the bottom cap
		if z_c > 0:
			p_a = (p_i + 0.5*l_i*a_i).T      	# axial position corresponding to the top cell pole
		else:
			p_a = (p_i - 0.5*l_i*a_i).T      	# axial position corresponding to the bottom cell poll

		# rotate the vector into the lab's frame using quaternions
		k_hat = numpy.array([0,0,1.0])			# unit z vector
		a_hat = numpy.array(a_i)				# unit cell axis
		p_c = numpy.array([x_c,y_c,z_c])		# radial vector in capsule frame
		q = self.GetRotationQuaternion(k_hat, a_hat) # quaternion to achieve rotation
		p_r = self.QuatRotate(q, p_c)				# rotate radial vector into lab frame
	
		return p_a, p_r


	def DrawACylPoint(self, p_i, a_i, l_i):
		"""
		Draw a vector with axial origin p_a and radial direction p_r from a capsule's cylindrical portion.
		The capsule's configuration is defined by position p_i, unit axis a_i, and length l_i.
		"""
		# draw axial position
		d_c = (numpy.random.uniform(0,1,1)-0.5)*l_i
		p_a = p_i + d_c*a_i # normally transposed
	
		# draw axial position
		phi = 2.0*numpy.pi*numpy.random.uniform(0,1,1)
		x_c = numpy.cos(phi)
		y_c = numpy.sin(phi)
		z_c = 0.0
	
		# rotate the vector into the lab's frame using quaternions
		k_hat = numpy.array([0,0,1.0])			# unit z vector
		a_hat = numpy.array(a_i)				# unit cell axis
		p_c = numpy.array([x_c,y_c,z_c])		# radial vector in capsule frame
		q = self.GetRotationQuaternion(k_hat, a_hat) # quaternion to achieve rotation
		p_r = self.QuatRotate(q, p_c)				# rotate radial vector into lab frame
	
		return p_a, p_r
	

	def GetRotationQuaternion(self, u1, u2):
		"""
		Compute Quaternion q transforming vector u1 into vector u2
		"""
		u1n = u1 / numpy.linalg.norm(u1)
		u2n = u2 / numpy.linalg.norm(u2)
		q = numpy.zeros((4,));
		q[0] = 1 + numpy.dot(u2n, u1n);          
		q[1:] = numpy.cross(u2n, u1n); 
	
		return q / numpy.linalg.norm(q)  


	def QuatRotate(self, q, r):
		"""
		Rotate vector r by quaternion q
		"""
		qin = q / numpy.linalg.norm(q)
		dcm = numpy.zeros((3,3));
		dcm[0][0] = numpy.power(qin[0],2) + numpy.power(qin[1],2) - numpy.power(qin[2],2) - numpy.power(qin[3],2)
		dcm[0][1] = 2*(numpy.multiply(qin[1], qin[2]) + numpy.multiply(qin[0], qin[3]))
		dcm[0][2] = 2*(numpy.multiply(qin[1], qin[3]) - numpy.multiply(qin[0], qin[2]))
		dcm[1][0] = 2*(numpy.multiply(qin[1], qin[2]) - numpy.multiply(qin[0], qin[3]))
		dcm[1][1] = numpy.power(qin[0],2) - numpy.power(qin[1],2) + numpy.power(qin[2],2) - numpy.power(qin[3],2)
		dcm[1][2] = 2*(numpy.multiply(qin[2], qin[3]) + numpy.multiply(qin[0], qin[1]))
		dcm[2][0] = 2*(numpy.multiply(qin[1], qin[3]) + numpy.multiply(qin[0], qin[2]))
		dcm[2][1] = 2*(numpy.multiply(qin[2], qin[3]) - numpy.multiply(qin[0], qin[1]))
		dcm[2][2] = numpy.power(qin[0],2) - numpy.power(qin[1],2) - numpy.power(qin[2],2) + numpy.power(qin[3],2)
	
		return numpy.dot(dcm,r).T
	
	
	"""
	AUXILLIARY FUNCTIONS FOR FILE I/O
	"""
	
	def LoadFromCellstates(self, cell_states):
		"""
		Fill cell data arrays directly from a cell_state dictionary
		"""
		# clear 
		#self.is_deleted = numpy.zeros(cell_geom, numpy.int32)

		# go through cell states dict and write entries to our arrays
		max_idx = 0
		for (cid,cs) in cell_states.items():
			i = cs.idx
			self.cell_centroids[i] = tuple(cs.pos)+(0,)
			self.cell_dirs[i] = tuple(cs.dir)+(0,)
			self.cell_rads[i] = cs.radius
			self.cell_lens[i] = cs.length
			self.cell_types[i] = cs.cellType
			if i > max_idx:
				max_idx = i
		#self.n_cells = len(cell_states) #debug
		self.n_cells = max_idx+1 # e.g. if max_idx = 144, min_idx = 0, then we have 144 + 1 idxs to search
		#print 'Found %i idxs, of which %i are living' % (self.n_cells, len(cell_states))
		
		# copy data from local to device
		self.SetCells()

	
	def WriteNeedlesToVTK(self, fileName):
		""" 
		Write needle vectors to vtk (.vtp) format
		"""
		# merge needle positions and vectors into arrays
		pos = self.P1s				# vector origins = surface points
		vec = self.P2s-self.P1s			# vector orientations and lengths

		# create a set of points
		Points = vtk.vtkPoints()
		for p in pos:
			Points.InsertNextPoint(p)

		# create needle data
		Needles = vtk.vtkDoubleArray()
		Needles.SetNumberOfComponents(3)
		Needles.SetName("NeedleVecs")
		for v in vec:
			Needles.InsertNextTuple(v)
		
		# append points and vectors to a vtk polydata object
		polydata = vtk.vtkPolyData()
		polydata.SetPoints(Points)
		polydata.GetPointData().SetVectors(Needles)
		polydata.Modified()

		# setup writer
		if vtk.VTK_MAJOR_VERSION <= 5:
			polydata.Update()
		writer = vtk.vtkXMLPolyDataWriter()
		writer.SetFileName(fileName)
		if vtk.VTK_MAJOR_VERSION <= 5:
			writer.SetInput(polydata)
		else:
			writer.SetInputData(polydata)

		# write to file
		print 'Writing needles to file...'
		writer.Write()
		
		
	def WritePointsToVTK(self, pos, fileName):
		"""
		Prototype writer for point data (e.g. collision points)
		"""		
		# create a set of points
		Points = vtk.vtkPoints()
		for p in pos:
			Points.InsertNextPoint(numpy.array([p[0],p[1],p[2]]))
			
		# append points and vectors to a vtk polydata object
		polydata = vtk.vtkPolyData()
		polydata.SetPoints(Points)
		polydata.Modified()

		# setup writer
		if vtk.VTK_MAJOR_VERSION <= 5:
			polydata.Update()
		writer = vtk.vtkXMLPolyDataWriter()
		writer.SetFileName(fileName)
		if vtk.VTK_MAJOR_VERSION <= 5:
			writer.SetInput(polydata)
		else:
			writer.SetInputData(polydata)

		# write to file
		print 'Writing points to file...'
		writer.Write()


