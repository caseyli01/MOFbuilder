import numpy as np
from place_bbs import superimpose
import networkx as nx

def termpdb(filename):
        inputfile = str(filename)
        with open(inputfile, "r") as fp:
            content = fp.readlines()
            #linesnumber = len(content)
        data = []
        for line in content:
            line = line.strip()
            if len(line)>0: #skip blank line
                if line[0:6] == "ATOM" or line[0:6] == "HETATM":
                    value_atom = line[12:16].strip()  # atom_label
                    #resname
                    #value2 = 'MOL'  # res_name

                    value_x = float(line[30:38])  # x
                    value_y = float(line[38:46])  # y
                    value_z = float(line[46:54])  # z
                    value_charge = float(line[61:66]) 
                    value_note = line[67:80].strip() # atom_note
                    #resnumber
                    try:
                        value_res_num = int(line[22:26])
                    except ValueError:
                        value_res_num = 1 
                    data.append([value_atom,value_x,value_y,value_z,value_charge,value_note,value_res_num,'TERM'])
        return data

def Xpdb(filename,X):
        data=termpdb(filename)
        X_term=[s for s in data if X in s[5]]
        return X_term

def convert_to_tuple(array):
    return (tuple(np.round(array[0],3).flatten()), tuple(np.round(array[1],3).flatten()),tuple(np.round(array[2],3).flatten()))

def is_list_A_in_B(A, B):
    # Convert both A and B into sets of tuple representations
    A_tuples = set(convert_to_tuple(a) for a in A)
    B_tuples = set(convert_to_tuple(b) for b in B)
    # Check if all elements in A are in B
    return A_tuples.issubset(B_tuples)

def add_terminations(term_file,ex_node_cxo_cc):
    tG=nx.Graph()
    terms=[]
    node_oovecs_record=[]
    terms_append = terms.append
    node_oovecs_record_append = node_oovecs_record.append

    term_data=termpdb(term_file)
    term_info = [i[0:1]+i[4:] for i in term_data]
    term_coords = [s[1:4] for s in termpdb(term_file)]
    xterm=Xpdb(term_file,'X')
    oterm=Xpdb(term_file,'Y')
    term_xvecs= [np.asarray([l[1],l[2],l[3]]) for l in xterm]
    term_ovecs= [np.asarray([l[1],l[2],l[3]]) for l in oterm]
    term_ovecs_c = np.mean(np.asarray(term_ovecs),axis=0)
    term_coords = [s[1:4] for s in termpdb(term_file)]-term_ovecs_c
    term_xoovecs =term_xvecs+term_ovecs
    term_xoovecs = term_xoovecs-term_ovecs_c

    for ex in range(len(ex_node_cxo_cc)):
        node_x = ex_node_cxo_cc[ex][3]
        node_opair=ex_node_cxo_cc[ex][5]
        node_opair_c = np.mean(np.asarray(node_opair),axis=0)
        node_xoo_vecs = [(i-node_opair_c).astype('float') for i in (node_opair+[node_x])]

        indices = [index for index, value in enumerate(node_oovecs_record) if is_list_A_in_B(node_xoo_vecs,value[0])]
        if len(indices)==1: 
            #find index of node_oo_vecs in record 
            #print(f"found one,{indices}")
            rot = node_oovecs_record[indices[0]][1]
        else:
            _,rot,_, = superimpose(term_xoovecs,node_xoo_vecs)
            node_oovecs_record_append((node_xoo_vecs,rot))
        adjusted_term_vecs = np.dot(term_coords,rot) + node_opair_c
        adjusted_term = np.hstack((np.asarray(term_info),np.full((len(term_info),1), ex),adjusted_term_vecs))
        terms_append(adjusted_term)
    return terms