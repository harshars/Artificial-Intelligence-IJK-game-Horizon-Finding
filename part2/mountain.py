#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))

    filtered_y = zeros(grayscale.shape)

    filters.sobel(grayscale,0,filtered_y)

    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image



# main program
(input_filename, gt_row, gt_col) = sys.argv[1:]
# load in image
input_image = Image.open(input_filename)
# compute edge strength mask
edge_strength = np.array(edge_strength(input_image),dtype='float64')

imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
ridge =np.argmax(edge_strength,axis=0)
edge_strength[edge_strength==0] = 1
# ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1] ###Initial ridge given by David
boundary_within_which_transition_probability_same=4
probability_within_the_boundary=0.1

def transitionprob(i,j):
    if abs(j-i)<=boundary_within_which_transition_probability_same :
        prob=probability_within_the_boundary

    if  abs(i) <=boundary_within_which_transition_probability_same and abs(j-i)>=boundary_within_which_transition_probability_same+1 :
        prob= (1-((abs(i)*probability_within_the_boundary)+((boundary_within_which_transition_probability_same+1)*probability_within_the_boundary)))/(len(edge_strength)-i-boundary_within_which_transition_probability_same-1)

    if  abs(i-len(edge_strength)) <boundary_within_which_transition_probability_same+1 and abs(j-i)>=boundary_within_which_transition_probability_same+1  :
        prob=(1-(abs(i-len(edge_strength))*probability_within_the_boundary+(boundary_within_which_transition_probability_same+1)*probability_within_the_boundary))/(len(edge_strength)-abs(i-len(edge_strength))-boundary_within_which_transition_probability_same-1)

    if abs(j-i) >=boundary_within_which_transition_probability_same+1 and abs(i) >=boundary_within_which_transition_probability_same+1 and abs(i-len(edge_strength)) >=boundary_within_which_transition_probability_same+1:
        prob=(1- (2*boundary_within_which_transition_probability_same+1)*probability_within_the_boundary)/(len(edge_strength)-(2*boundary_within_which_transition_probability_same+1))

    return(prob)


def emmissionprob(i,j):
    emmprob=log(edge_strength[i][j])-log(np.sum(edge_strength,axis=0)[j])
    return(emmprob)

def emmissionprob2(row_coord, col_coord,i,j):
    if col_coord == j and row_coord == i:
        emmprob1=log(0.999)
    if col_coord==j and row_coord!=i:
        emmprob1=log((1-0.999)/(len(edge_strength)-1))
    else:
        emmprob1=emmissionprob(i,j)
    return(emmprob1)


######Viterbi Algorithm

def Viterbi_alg(method):
    mat=np.zeros(shape=(len(edge_strength),	len(edge_strength[0])),dtype='float64')
    mat1= np.zeros(shape=(len(edge_strength),len(edge_strength[0])))
    initial_probability=1/len(edge_strength)
    for i in range(0,len(edge_strength)):
        mat[i][0]=emmissionprob(i,0)+log(initial_probability)
        mat1[i][0]=i
    for i in range(1,len(edge_strength[0])):
        for j in range(0,len(edge_strength)):
            a=[]
            for k in range(0,len(edge_strength)):
                if method=="Human_Input":
                    intermediate=mat[k][i-1]+log(transitionprob(k,j))+emmissionprob2(gt_row,gt_col,j,i)
                if method=="Viterbi":
                    intermediate=mat[k][i-1]+log(transitionprob(k,j))+emmissionprob(j,i)
                a.append(intermediate)
            mat[j][i]=max(a)
            mat1[j][i]=a.index(max(a))
    last_column_row=np.argmax(mat,axis=0)[len(edge_strength[0])-1]
    k1=list(np.argmax(mat,axis=0))
    k1.pop(0)
    k1.append(last_column_row)
    return(k1)
    # k1=[last_column_row]
    # for ik in range(len(edge_strength[0])-1,0,-1):
    #     ab=mat1[last_column_row][ik]
    #     k1.append(ab)
    # k1.reverse()
    # return(k1)

#output answer
imageio.imwrite("output.jpg", draw_edge(input_image, Viterbi_alg("Human_Input"), (0, 255, 0), 5))
imageio.imwrite("output1.jpg", draw_edge(input_image, Viterbi_alg("Viterbi"), (255, 0, 0), 5))
imageio.imwrite("output2.jpg", draw_edge(input_image, ridge, (0, 0, 255), 5))
