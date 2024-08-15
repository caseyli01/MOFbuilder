import numpy as np

##Frame
def Carte_points_generator(xyz_num):
        x_num, y_num, z_num = xyz_num
        """this function is to generate a group of 3d points(unit=1) defined by user for further grouping points"""
        unit_dx, unit_dy, unit_dz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # add x layer
        points = np.array([0, 0, 0])
        for i in range(0, x_num + 1):
            points = np.vstack((points, i * unit_dx))
        # add y layer
        points_x = points
        for i in range(0, y_num + 1):
            points = np.vstack((points, points_x + i * unit_dy))
        # add z layer
        points_xy = points
        for i in range(0, z_num + 1):
            points = np.vstack((points, points_xy + i * unit_dz))
        points = np.unique(points, axis=0)
        return points

def tric_points(Carte_points,unit_cell):
        #Carte_points = Frame.Carte_points_generator(
        #    self.x_num,
        #    self.y_num,
        #    self.z_num,
        #)
        if len(Carte_points)>0:
            supercell_tric_points = np.round(np.dot(Carte_points, unit_cell),3)
            return supercell_tric_points
        else:
            return np.empty((0,3))
        

def cleave_supercell_boundary(supercell, supercell_boundary):
    #print(
    #    f"supercell {supercell.shape[0]}, supercell_boundary {supercell_boundary.shape[0]}"
    #)
    supercell_inside_list = [
        i for i in supercell.tolist() if i not in supercell_boundary.tolist()
    ]
    #print(f"cleaved supercell(not in latter but in former) {len(supercell_inside_list)}")
    return np.array(supercell_inside_list)

class Frame():
    def __init__(self,supercell_xyz,unit_cell):
         self.supercell_xyz = supercell_xyz
         self.unit_cell = unit_cell
         
    def supercell_generator(self):
        super_cell_x, super_cell_y, super_cell_z = self.supercell_xyz
        supercell_Carte = Carte_points_generator(self.supercell_xyz)
        self.supercell_Carte = supercell_Carte
        # print(f"supercell_Carte{supercell_Carte}")
        outer_supercell_Carte = Carte_points_generator(
            super_cell_x + 2, super_cell_y + 2, super_cell_z + 2
        )-np.array([1,1,1])
        # print(f"outer_Carte{outer_supercell_Carte}")

        supercell_boundary_Carte = supercell_Carte[
            (supercell_Carte[:, 0] == super_cell_x)
            | (supercell_Carte[:, 0] == 0)
            | (supercell_Carte[:, 1] == super_cell_y)
            | (supercell_Carte[:, 1] == 0)
            | (supercell_Carte[:, 2] == super_cell_z)
            | (supercell_Carte[:, 2] == 0)
        ]


        supercell_inside_Carte = cleave_supercell_boundary(supercell_Carte,supercell_boundary_Carte)
        layer_out_supercell_Carte = cleave_supercell_boundary(outer_supercell_Carte,supercell_Carte)

        boundary_Carte_x  = supercell_boundary_Carte[supercell_boundary_Carte[:,0]==super_cell_x]
        boundary_Carte_x_ = supercell_boundary_Carte[supercell_boundary_Carte[:,0]==0]
        boundary_Carte_y  = supercell_boundary_Carte[supercell_boundary_Carte[:,1]==super_cell_y]
        boundary_Carte_y_ = supercell_boundary_Carte[supercell_boundary_Carte[:,1]==0]
        boundary_Carte_z  = supercell_boundary_Carte[supercell_boundary_Carte[:,2]==super_cell_z]
        boundary_Carte_z_ = supercell_boundary_Carte[supercell_boundary_Carte[:,2]==0]

        layer_out_Carte_x  = layer_out_supercell_Carte[layer_out_supercell_Carte[:,0]>super_cell_x]
        layer_out_Carte_x_ = layer_out_supercell_Carte[layer_out_supercell_Carte[:,0]<0]
        layer_out_Carte_y  = layer_out_supercell_Carte[layer_out_supercell_Carte[:,1]>super_cell_y]
        layer_out_Carte_y_ = layer_out_supercell_Carte[layer_out_supercell_Carte[:,1]<0]
        layer_out_Carte_z  = layer_out_supercell_Carte[layer_out_supercell_Carte[:,2]>super_cell_z]
        layer_out_Carte_z_ = layer_out_supercell_Carte[layer_out_supercell_Carte[:,2]<0]

        self.supercell=tric_points(supercell_Carte,self.unit_cell)
        self.boundary_supercell     = tric_points(supercell_boundary_Carte, self.unit_cell)
        self.inside_supercell       = tric_points(supercell_inside_Carte,   self.unit_cell)
        self.layer_out_supercell    = tric_points(layer_out_supercell_Carte,self.unit_cell)
        self.boundary_supercell_x   = tric_points(boundary_Carte_x , self.unit_cell)
        self.boundary_supercell_x_  = tric_points(boundary_Carte_x_, self.unit_cell)
        self.boundary_supercell_y   = tric_points(boundary_Carte_y , self.unit_cell)
        self.boundary_supercell_y_  = tric_points(boundary_Carte_y_, self.unit_cell)
        self.boundary_supercell_z   = tric_points(boundary_Carte_z , self.unit_cell)
        self.boundary_supercell_z_  = tric_points(boundary_Carte_z_, self.unit_cell)
        self.layer_out_supercell_x  = tric_points(layer_out_Carte_x ,self.unit_cell)
        self.layer_out_supercell_x_ = tric_points(layer_out_Carte_x_,self.unit_cell)
        self.layer_out_supercell_y  = tric_points(layer_out_Carte_y ,self.unit_cell)
        self.layer_out_supercell_y_ = tric_points(layer_out_Carte_y_,self.unit_cell)
        self.layer_out_supercell_z  = tric_points(layer_out_Carte_z ,self.unit_cell)
        self.layer_out_supercell_z_ = tric_points(layer_out_Carte_z_,self.unit_cell)

