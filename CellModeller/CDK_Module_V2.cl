#define EPSILON 0.1f
#define MARGIN 0.01f

// find the point where a needle hit a cell's surface
// /!\ NOTE: only run this on needles that definitely hit the focal cell
void contact_point(const float4 r_a,  // center of first segment
				   const float4 r_b,  // center of second segment
				   const float4 d_a,  // direction of first segment (unit)
				   const float4 d_b,  // direction of second segment (unit)
				   const float len_a, // length of first segment
				   const float len_b, // length of second segment
				   const float rad,   // cell radius
				   float* s_star,     // (return) parametric coordinate of intercept WRT cell axis
				   float* t_star,	  // (return) parametric coordinate of intercept WRT needle axis
				   float4* E0,		  // (return) cartesian coordinates of axial intercept point
				   float4* E1)		  // (return) cartesian coordinates of surface intercept point
{   
   // compute segment end points
   float hlen_a = len_a / 2.f;
   float hlen_b = len_b / 2.f;
   float4 P0 = r_a - hlen_a * d_a;	
   float4 P1 = r_a + hlen_a * d_a;
   float4 Q0 = r_b - hlen_b * d_b;
   float4 Q1 = r_b + hlen_b * d_b;
   
   // intermediate vectors
   float4 U = P1 - P0;
   float4 V = Q1 - Q0;
   float4 W0 = P0 - Q0;
   
   // intermediate scalars
   float a = dot(U, U);
   float b = -2.f * dot(U, V);
   float c = dot(V, V);
   float d = 2.f * dot(W0, U);
   float e = -2.f * dot(W0, V);
   float f = dot(W0, W0) - pow(rad, 2);
   
   // check 1: get intercept with left sphere, if one exists
   float sL = 0;
   float tL = INFINITY;
   float disc_L = pow(e + b*sL, 2) - 4.f*c*(a*pow(sL,2) + d*sL + f);
   if (disc_L >= 0) {
      tL = (-1.f*(e + b*sL) - sqrt(disc_L)) / (2.f*c);
   }
   
   // check 2: get intercept with left sphere, if one exists
   float sR = 1;
   float tR = INFINITY;
   float disc_R = pow(e + b*sR, 2) - 4.f*c*(a*pow(sR,2) + d*sR + f);
   if (disc_R >= 0) {
      tR = (-1.f*(e + b*sR) - sqrt(disc_R)) / (2.f*c);
   }
   
   // check 3: intercept with cylindrical midsection, if one exists
   float c1 = (c - (pow(b,2) / (4.f*a)));
   float c2 = -1.f * (((b*d) / (2*a)) - e); 
   float c3 = (f - (pow(d,2) / (4.f*a)));
   float tC = (-1.f*c2 - sqrt(pow(c2,2) - 4.f*c1*c3)) / (2.f*c1);
   float sC = -0.5*(d+(tC*b)) / a;
   if (!(sC > 0 && sC < 1)) {   
       tC = INFINITY;
   }
	
   // choose the intercept coordinates closest to the needle's base (lowest t)   
   float this_s_star = sL;
   float this_t_star = tL;
   if (tR < this_t_star) {
      this_s_star = sR;
      this_t_star = tR;
   }
   if (tC < this_t_star) {
      this_s_star = sC;
      this_t_star = tC;
   }
   
   // return parameteric coordinates of intercept
   *s_star = this_s_star;
   *t_star = this_t_star;   
   
   // compute and return cartesian coords of axial and surface points corresponding to intercept
   *E0 = P0 + this_s_star*U;
   *E1 = Q0 + this_t_star*V;


}

// find the closest points on two line segments
void closest_points_on_segments(const float4 r_a,  // center of first segment
                                const float4 r_b,  // center of second segment
                                const float4 a,    // direction of first segment (unit)
                                const float4 b,    // direction of second segment (unit)
                                const float len_a, // length of first segment
                                const float len_b, // length of second segment
                                float4* p_a,       // (return) point on first segment
                                float4* p_b,       // (return) point on second segment
                                float4* p_a2,      // (return) 2nd point on first segment
                                float4* p_b2,      // (return) 2nd point on second segment
                                int* two_pts)      // (return) were two points picked?
{
  float hlen_a = len_a / 2.f;
  float hlen_b = len_b / 2.f;
  float4 r = r_b - r_a;
  float a_dot_r = dot(a, r);
  float b_dot_r = dot(b, r);
  float a_dot_b = dot(a, b);
  float denom = 1.f - a_dot_b * a_dot_b;

  float t_a = 0.f;
  float t_b = 0.f;

  *two_pts = 0;
  float t_a2 = 0.f;
  float t_b2 = 0.f;

  if (sqrt(denom) > EPSILON) {
    // non-parallel lines

    // closest points on the same lines if they were infinitely long
    float t_a0 = (a_dot_r - b_dot_r * a_dot_b) / denom;
    float t_b0 = (a_dot_r * a_dot_b - b_dot_r) / denom;

    // there are a few different cases we have to handle...
    bool on_a = fabs(t_a0) < hlen_a;
    bool on_b = fabs(t_b0) < hlen_b;
    if (!on_a && !on_b) {
      // the corner
      float c_a = copysign(hlen_a, t_a0);
      float c_b = copysign(hlen_b, t_b0);

      // signs of partials at the corner
      float dd_dt_a = 2.f*(c_a - a_dot_b*c_b - a_dot_r);
      float dd_dt_b = 2.f*(c_b - a_dot_b*c_a + b_dot_r);

      if (sign(dd_dt_a) == sign(c_a)) {
        // on the other edge
        t_b = c_b;
        t_a = clamp(t_b*a_dot_b + a_dot_r, -hlen_a, hlen_a);
      } else {
        t_a = c_a;
        t_b = clamp(t_a*a_dot_b - b_dot_r, -hlen_b, hlen_b);
      }
    } else if (on_a && !on_b) {
      t_b = copysign(hlen_b, t_b0);  // +/- end of b?
      t_a = clamp(t_b*a_dot_b + a_dot_r, -hlen_a, hlen_a);
    } else if (!on_a && on_b) {
      t_a = copysign(hlen_a, t_a0);  // +/- end of a?
      t_b = clamp(t_a*a_dot_b - b_dot_r, -hlen_b, hlen_b);
    } else {
      t_a = t_a0;
      t_b = t_b0;
    }
  } else {
    // lines are roughly parallel, this case is degenerate
    // start off assuming the lines are in the same direction
    // project a onto b

    // use the same _dot_r for each for consistency
    float x_dot_r = copysign(min(a_dot_r, b_dot_r), a_dot_r);

    // project the ends of a into b coordinates
    float a_l = -x_dot_r - hlen_a;
    float a_r = -x_dot_r + hlen_a;

    // find the intersection of the two on b
    float i_l = max(a_l, -hlen_b);
    float i_r = min(a_r, hlen_b);

    if (i_l > i_r) {
      // they don't intersect
      if (a_l < -hlen_b) {
        t_a = hlen_a;
        t_b = -hlen_b;
      } else {
        t_a = -hlen_a;
        t_b = hlen_b;
      }
    } else {
      // the segments intersect, pick two points
      *two_pts = 1;
      t_b = i_l;
      t_a = t_b + x_dot_r;
      t_b2 = i_r;
      t_a2 = t_b2 + x_dot_r;
    }

    // if they weren't in the same direction, negate
    if (a_dot_b < 0.f) {
      t_b = -t_b;
      t_b2 = -t_b2;
    }
  }
  *p_a = r_a + t_a*a;
  *p_b = r_b + t_b*b;
  if (two_pts) {
    *p_a2 = r_a + t_a2*a;
    *p_b2 = r_b + t_b2*b;
  }
}


// Set the sq of a centroid (cell or needle) based on its position.
__kernel void bin_centroids(const int grid_x_min,
                        const int grid_x_max,
                        const int grid_y_min,
                        const int grid_y_max,
                        const float grid_spacing,
                        __global const float4* centroids,
                        __global int* sqs)
{
  int i = get_global_id(0);
  int x = (int)floor(centroids[i].x / grid_spacing) - grid_x_min;
  int y = (int)floor(centroids[i].y / grid_spacing) - grid_y_min;
  sqs[i] = y*(grid_x_max-grid_x_min) + x;
}


// Find all needles touching a given cell
__kernel void check_hits(const int n_needles,
                            const int grid_x_min,
                            const int grid_x_max,
                            const int grid_y_min,
                            const int grid_y_max,
                            const int n_sqs,
                            __global const int* cell_sqs,
                            __global const int* needle_sqs,
                            __global const int* needle_sorted_ids,
                            __global const int* needle_sq_inds,
                            __global const float4* cell_centroids,
                            __global const float4* cell_dirs,
                            __global const float4* needle_centroids,
                            __global const float4* needle_dirs,
                            __global const float* needle_lens,
                            __global const float* cell_lens,
                            __global const float* cell_rads,
                            __global int* cell_hits,
                            __global int* needle_hits,
                            __global int* needle_to_cell,
                            __global float4* needle_hits_axis,
                            __global float4* needle_hits_surf)

{
  // id of focal cell
  int i = get_global_id(0);
  
  // what square are we in?
  int grid_x_range = grid_x_max-grid_x_min;
  int grid_y_range = grid_y_max-grid_y_min;
  int sq_row = cell_sqs[i] / grid_x_range; // square row
  int sq_col = cell_sqs[i] % grid_x_range; // square col

  // loop through our square and the eight squares surrounding it
  // (fewer squares if we're on an edge)
  for (int row = max(0, sq_row-1); row < min((int)(sq_row+2), grid_y_range); row++) {
  		    
    for (int col = max(0, sq_col-1); col < min((int)(sq_col+2), grid_x_range); col++) {

      // what square is this?
      int sq = row*grid_x_range + col;

      // loop through all the cell ids in the current square (but
      // don't go past the end of the list)
      for (int n = needle_sq_inds[sq]; n < (sq < n_sqs-1 ? needle_sq_inds[sq+1] : n_needles); n++) {

        int j = needle_sorted_ids[n]; // idx of neighboring needle

		// bounding spheres test: are these objects close enough to touch, irrespective of orientation?
        if (length(cell_centroids[i] - needle_centroids[j]) > 0.5*(cell_lens[i] + 2.0*cell_rads[i] + needle_lens[j]) - MARGIN)
        {
          // if not...
			continue;
        }
        
        // shortest distance test
        float4 pi, pj; // pi is point on cell segment, pj point on needle segment
        float4 pi2, pj2; // optional second points  (not used)
        int two_pts = 0; // are there two contacts? (not used)
        closest_points_on_segments(cell_centroids[i], needle_centroids[j],
                                   cell_dirs[i], needle_dirs[j],
                                   cell_lens[i], needle_lens[j],
                                   &pi, &pj, &pi2, &pj2, &two_pts);

        float4 v_ij = pj-pi; // vector between closest points
        float dist = length(v_ij) - cell_rads[i];
		
		// if we scored a hit...
		if (dist < -1.0*MARGIN)
        {
			//...increment hit count on this cell
			cell_hits[i]++;  // moving to different system
			needle_hits[j]=1;
			needle_to_cell[j]=i;
			
			// call the contact point finder, and demo passing data
			float s_star, t_star;
			float4 E0, E1;
			contact_point(cell_centroids[i], needle_centroids[j],
                                   cell_dirs[i], needle_dirs[j],
                                   cell_lens[i], needle_lens[j],
                                   cell_rads[i],
                                   &s_star, &t_star,
                                   &E0, &E1);
                                   
            // write hit points to needle-shaped arrays
            needle_hits_axis[j] = E0;
            needle_hits_surf[j] = E1;                                                      
		}  
      }
    }
  }
}

// Find all needles touching a given cell, using periodic boundary conditions
__kernel void check_hits_periodic(const int n_needles,
                            const int grid_x_min,
                            const int grid_x_max,
                            const int grid_y_min,
                            const int grid_y_max,
                            const int n_sqs,
                            __global const int* cell_sqs,
                            __global const int* needle_sqs,
                            __global const int* needle_sorted_ids,
                            __global const int* needle_sq_inds,
                            __global const int* needle_sq_neighbour_inds,
                            __global const int* needle_sq_neighbour_offset_inds,
                            __global const float4* offset_vecs,
                            __global const float4* cell_centroids,
                            __global const float4* cell_dirs,
                            __global const float4* needle_centroids,
                            __global const float4* needle_dirs,
                            __global const float* needle_lens,
                            __global const float* cell_lens,
                            __global const float* cell_rads,
                            __global int* cell_hits,
                            __global int* needle_hits,
                            __global int* needle_to_cell,
                            __global float4* needle_hits_axis,
                            __global float4* needle_hits_surf)

{
  // id of focal cell
  int i = get_global_id(0);
  
  // focal cell's square
  int sq_f = cell_sqs[i];
  
  // loop through the cell's 9 neighbours
  for (int neigh_index = 0; neigh_index < 9; neigh_index++) {
  
    // look up the index of this neighbour square
	int sq = needle_sq_neighbour_inds[sq_f*9 + neigh_index];  		    

	// offset vector for needles in this neighbour square
	int offset_ind = needle_sq_neighbour_offset_inds[sq_f*9 + neigh_index];	
	float4 offset = offset_vecs[offset_ind];

	// loop through all the needles listed in this neighbour square
    for (int n = needle_sq_inds[sq]; n < (sq < n_sqs-1 ? needle_sq_inds[sq+1] : n_needles); n++) {

      int j = needle_sorted_ids[n]; // idx of neighboring needle

	  // bounding spheres test: are these objects close enough to touch, irrespective of orientation?
      if (length(cell_centroids[i] - needle_centroids[j] - offset) > 0.5*(cell_lens[i] + 2.0*cell_rads[i] + needle_lens[j]) - MARGIN)
      {
          // if not...
			continue;
      }
        
      // shortest distance test
      float4 pi, pj; // pi is point on cell segment, pj point on needle segment
      float4 pi2, pj2; // optional second points  (not used)
      int two_pts = 0; // are there two contacts? (not used)
      closest_points_on_segments(cell_centroids[i], (needle_centroids[j]+offset),
                                   cell_dirs[i], needle_dirs[j],
                                   cell_lens[i], needle_lens[j],
                                   &pi, &pj, &pi2, &pj2, &two_pts);

      float4 v_ij = pj-pi; // vector between closest points
      float dist = length(v_ij) - cell_rads[i];
		
	  // if we scored a hit...
	  if (dist < -1.0*MARGIN)
      {
			//...increment hit count on this cell
			cell_hits[i]++;  // moving to different system
			needle_hits[j]=1;
			needle_to_cell[j]=i;
			
			// call the contact point finder, and demo passing data
			float s_star, t_star;
			float4 E0, E1;
			contact_point(cell_centroids[i], (needle_centroids[j]+offset),
                                   cell_dirs[i], needle_dirs[j],
                                   cell_lens[i], needle_lens[j],
                                   cell_rads[i],
                                   &s_star, &t_star,
                                   &E0, &E1);
                                   
            // write hit points to needle-shaped arrays
            needle_hits_axis[j] = E0;
            needle_hits_surf[j] = E1;                                                      
	  }  
    }
  }
}