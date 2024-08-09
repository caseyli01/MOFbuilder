import numpy as np
import re 

from place_bbs import superimpose

def PBC3DF_sym(vec1, vec2):

	dX,dY,dZ = vec1 - vec2
			
	if dX > 0.5:
		s1 = 1.0
		ndX = dX - 1.0
	elif dX < -0.5:
		s1 = -1.0
		ndX = dX + 1.0
	else:
		s1 = 0.0
		ndX = dX
				
	if dY > 0.5:
		s2 = 1.0
		ndY = dY - 1.0
	elif dY < -0.5:
		s2 = -1.0
		ndY = dY + 1.0
	else:
		s2 = 0.0
		ndY = dY
	
	if dZ > 0.5:
		s3 = 1.0
		ndZ = dZ - 1.0
	elif dZ < -0.5:
		s3 = -1.0
		ndZ = dZ + 1.0
	else:
		s3 = 0.0
		ndZ = dZ

	sym = np.array([s1,s2,s3])

	return np.array([ndX,ndY,ndZ]), sym

def newno_fxnx(f_ex,nx,sc_unit_cell,no):
	f_nx = np.dot(np.linalg.inv(sc_unit_cell), nx)
	fdist_vec,sym = PBC3DF_sym(f_ex, f_nx) 
	f_no = np.dot(np.linalg.inv(sc_unit_cell), no)
	new_no=np.dot(sc_unit_cell,f_no + sym)
	return new_no

def adjust_edges(placed_edges, placed_nodes, sc_unit_cell):
	#find closet x to adjust placed_edge pos
	adjusted_placed_edges = []
	adjusted_placed_edges_extend = adjusted_placed_edges.extend
	adjusted_placed_OXedges = []
	adjusted_placed_OXedges_extend = adjusted_placed_OXedges.extend

	placed_edges = np.asarray(placed_edges)
	edge_labels = set(map(int, placed_edges[:,-1]))

	edge_dict = dict((k,[]) for k in edge_labels)

	#node_connection_x = [list(map(float,i[1:4])) for i in placed_nodes if re.sub('[0-9]','',i[5]) == 'X']
	node_connection_x = np.asarray([i for i in placed_nodes if re.sub('[0-9]','',i[5]) == 'X'])
	nx_elems = node_connection_x[:,0]
	node_connection_points  = [list(map(float,i)) for i in node_connection_x[:,1:4]]
	nx_charges = node_connection_x[:,4]
	nx_cp = node_connection_x[:,5]
	#o_ty = node_oxy[:,6]


	node_oxy = np.asarray([i for i in placed_nodes if re.sub('[0-9]','',i[5]) == 'O'])
	no_elems = node_oxy[:,0]
	node_oxy_points = [list(map(float,i)) for i in node_oxy[:,1:4]]
	no_charges = node_oxy[:,4]
	no_cp = node_oxy[:,5]
	#no_ty = node_oxy[:,6]

	'''look for two nearest Oxys for every X'''
	X_Opair = []
	X_Opair_append = X_Opair.append
	for i in range(len(node_connection_points)):
		cdist_xos = []
		cdist_xos_sort = []
		cdist_xos_append=cdist_xos.append
		cdist_xos_sort_append=cdist_xos_sort.append
		fvec_x= node_connection_points[i]
		for j in range(len(node_oxy_points)):
			fvec_o= node_oxy_points[j]
			fvec_xo = np.asarray(fvec_o)-np.asarray(fvec_x) 	
			fdist_xo = np.dot(np.linalg.inv(sc_unit_cell), fvec_xo)
			cdist_xo = np.linalg.norm(np.dot(sc_unit_cell, fdist_xo))
			cdist_xos_append(cdist_xo)
			cdist_xos_sort_append(cdist_xo)
		cdist_xos_sort.sort()
		cdist_xos_sort3rd=cdist_xos_sort[2]
		#print(f"cdist_xos_sort3rd{cdist_xos_sort3rd},\ncdist_xos_sort{cdist_xos_sort}")
		opair=[index for index,value in enumerate(cdist_xos) if value < cdist_xos_sort3rd]
		X_Opair_append(('X'+str(i),node_connection_points[i],(opair),[node_oxy_points[i] for i in opair]))
	
	'''cleave placed_nodes remove X_Opair, but future need add dummy atom(can be applied in node cif file )'''
	stacked_opairs = np.asarray([i[3] for i in X_Opair])
	opairs_vec= stacked_opairs.reshape(-1, stacked_opairs.shape[-1])
	xs_vec = np.asarray([i[1] for i in X_Opair])
	xos_vec = np.vstack((opairs_vec,xs_vec))
	cleaved_placed_nodes = [i for i in placed_nodes if list(map(float,i[1:4])) not in xos_vec ]

	for edge in placed_edges:
		ty = int(edge[-1])
		edge_dict[ty].append(edge)

	for k in edge_dict:

		edge = np.asarray(edge_dict[k])
		elems = edge[:,0]
		evecs = [list(map(float,i)) for i in edge[:,1:4]]
		charges = edge[:,4]
		cp = edge[:,5]
		ty = edge[:,6]

		xvecs = [list(map(float,i)) for (i,j) in zip(evecs,cp) if re.sub('[0-9]','',j) == 'X']
		relevant_node_xvecs = []
		relevant_node_xvecs_append = relevant_node_xvecs.append
		#fixed_node_connection_points = [list(map(float,i[1:4])) for i in placed_nodes if re.sub('[0-9]','',i[5]) == 'X']
		corr_opair=[]
		corr_opair_append = corr_opair.append
		for count in range(len(xvecs)):
			ex = xvecs[count]
			min_dist = (1e6, [], 0)

			f_ex = np.dot(np.linalg.inv(sc_unit_cell), ex)
			for i in range(len(node_connection_points)):
				nx = node_connection_points[i]
				f_nx = np.dot(np.linalg.inv(sc_unit_cell), nx)

				fdist_vec,sym = PBC3DF_sym(f_ex, f_nx) 
				cdist = np.linalg.norm(np.dot(sc_unit_cell, fdist_vec))

				if cdist < min_dist[0]:
					min_dist = (cdist, np.dot(sc_unit_cell,f_nx + sym), i)
					target_nx = nx
			#idx_x_fixed = [index for index, value in enumerate(fixed_node_connection_points) if value == target_nx]
			#node_connection_points.pop(min_dist[2]) #(once find one then remove it to speed up loop)
			relevant_node_xvecs_append(min_dist[1])
			idx_nx = min_dist[2]
			corresponding_o_vec = [newno_fxnx(f_ex,target_nx,sc_unit_cell,no) for no in X_Opair[idx_nx][3]]
			corresponding_x_vec = [newno_fxnx(f_ex,target_nx,sc_unit_cell,target_nx)]
			#corresponding_o_vec = np.dot(X_Opair[idx_nx][2],sc_unit_cell)
			pairo_indices = X_Opair[idx_nx][2]
			elems_nopair = [no_elems[i] for i in pairo_indices]
			charges_nopair = [no_charges[i] for i in pairo_indices]
			cp_nopair = [no_cp[i] for i in pairo_indices]
			ty_nopair =[ty[0]] *len(pairo_indices)

			corresponding_o=np.c_[elems_nopair,corresponding_o_vec,charges_nopair,cp_nopair,ty_nopair]
			corresponding_x=np.c_[[nx_elems[min_dist[2]]],corresponding_x_vec,[nx_charges[min_dist[2]]],[nx_cp[min_dist[2]]],[ty[0]]]
			#print(f"edgeX{count}\npop{min_dist[2]}\ncorresponding_o{corresponding_o}\nX_Opair[idx_nx]{X_Opair[idx_nx]}")
			corr_opair_append(corresponding_o)
			corr_opair_append(corresponding_x)
		ecom = np.average(xvecs, axis=0)
		rnxcom = np.average(relevant_node_xvecs, axis=0)

		evecs = np.asarray(evecs - ecom)
		xvecs = np.asarray(xvecs - ecom)
		relavent_node_xvecs = np.asarray(relevant_node_xvecs)

		trans = rnxcom
		min_dist,rot,tran = superimpose(xvecs,relavent_node_xvecs)
		adjusted_evecs = np.dot(evecs,rot) + trans

		adjusted_edge_in = np.column_stack((elems,adjusted_evecs,charges,cp,ty))
		adjusted_edge_opair = np.vstack(corr_opair)
		adjusted_OXedge = np.vstack((adjusted_edge_opair,adjusted_edge_in))
		#print(f"trans{trans}")
		adjusted_placed_edges_extend(adjusted_edge_in)
		adjusted_placed_OXedges_extend(adjusted_OXedge)

	return adjusted_placed_edges,adjusted_placed_OXedges,cleaved_placed_nodes,X_Opair		
