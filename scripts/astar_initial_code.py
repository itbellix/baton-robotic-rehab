# Code originally from KukaPath_20210211.py (author Micah Prendergast)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math as m
import csv
import os

########################
import matplotlib
# matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle, PathPatch, ConnectionPatch
from warnings import warn
import heapq
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import time

############################


global waypoints
waypoints = []

global numpnts
numpnts = 4

global wait
wait = True


saveTransform = True
drawTransform = True
viewplots = True
viewspeed = 50
directory = "/home/itbellix/Desktop/GitHub/baton-robotic-rehab"
name = "newpath"
rawpathname = "rawpath"
pointsfile = "pathpoints"
waypointsfile = "waypoints"
pathname = directory+name
EE_Down = False                         #Set to True for Down facing EE
autotraj = True                         #Set to False for manual trajectory
AxialRotFixed = 24

################################ Patient Specific Data ##################################################

L_arm = 0.35                            #currently patient arm - 0.06 .4 for Stephan .35 for Micah
pose_x = -0.3                           #Shoulder position in robot base frame X
pose_y = 0.1                            #Shoulder position in robot base frame Y
pose_z = 0.07                           #Shoulder position in robot base frame Z #0.05 for Micah .18 for stephan

# Intia
PatientOrientation = np.array([0,90,0]) #[0, 90, 90] default


def onclick(event):
    global wait
    wait = True
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
        ax5.plot(event.xdata,event.ydata, 'o', color = 'white')
        x = event.xdata
        y = event.ydata
        waypoints.append((int(x),int(y)))
    if len(waypoints) > numpnts:
        wait = False

# Credit for Base Astar: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None, strain=None):
        self.parent = parent
        self.position = position
        self.strain = strain

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return "{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(maze, start, end, strain, sweight, allow_diagonal_movement=True, allow_3D_map=False):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """
    meanstrain = np.mean(strain)
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = 2*(len(maze[0]) * len(maze) // 2)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    if allow_3D_map:
        max_iterations = (len(maze[0]) * len(maze) * len(maze[2]) // 2)
        adjacent_squares = ((0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                            (-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, 0, -1), (1, 0, 1), (-1, 0, 1), (1, 0, -1),
                            (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1),
                            (1, 0, 0), (-1, 0, 0))


    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            warn("giving up on pathfinding too many iterations")
            return return_path(current_node)

            # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []

        for new_position in adjacent_squares:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Get Node Strain
            node_strain = strain[node_position[0]][node_position[1]]  #swapped indices here.. wtf? check other things.

            # Create new node
            new_node = Node(current_node, node_position, node_strain)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 20*child.strain #180*meanstrain*child.strain # current_node.g + 1
            child.h = (meanstrain* ((child.position[0] - end_node.position[0])) ** 2) + (meanstrain*((child.position[1] - end_node.position[1])) ** 2)
            child.f = child.g + child.h * (child.strain**2)

            # Child is already in the open list
            if len([open_node for open_node in open_list if
                    child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None

######################################################################################




def Rx(th):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(th), -m.sin(th)],
                      [0, m.sin(th), m.cos(th)]])

def Ry(phi):
    return np.matrix([[m.cos(phi), 0, m.sin(phi)],
                      [0, 1, 0],
                      [-m.sin(phi), 0, m.cos(phi)]])

def Rz(psi):
    M = np.matrix([[m.cos(psi), -m.sin(psi), 0],
                      [m.sin(psi), m.cos(psi), 0],
                      [0, 0, 1]])
    return M

# Compute rotation matrix given some euler angles
def Rxyz(theta, phi, psi):
    R = Rz(psi * np.pi / 180) * Ry(phi * np.pi / 180) * Rx(theta * np.pi / 180)
    return R

def Rxzz(theta, phi, psi):
    R = Rz(psi * np.pi / 180) * Rz(phi * np.pi / 180) * Rx(theta * np.pi / 180)
    return R

# Compute new vector after some rotation
def MatRot(R, v):

    return np.asarray(np.matmul(R, v)).flatten()

# Create rotation matrix and rotate a vector
def VecRot(v, theta, phi, psi):
    R = Rz(psi*np.pi/180) * Ry(phi*np.pi/180) * Rx(theta*np.pi/180)
    return np.asarray(np.matmul(R, v)).flatten()

class frame:
    def __init__(self, pose):

        R = Rxyz(pose[3], pose[4], pose[5])

        origin = np.array([pose[0], pose[1], pose[2]])

        trans = np.array([[pose[0], pose[1], pose[2]]])
        tf = np.concatenate((R,trans.T), axis=1)
        tf = np.concatenate((tf,np.array([[0,0,0,1]])), axis=0)

        xax = MatRot(R, np.array([1, 0, 0]))
        yax = MatRot(R, np.array([0, 1, 0]))
        zax = MatRot(R, np.array([0, 0, 1]))

        xend = np.array([xax[0] + origin[0], xax[1] + origin[1], xax[2] + origin[2]])
        yend = np.array([yax[0] + origin[0], yax[1] + origin[1], yax[2] + origin[2]])
        zend = np.array([zax[0] + origin[0], zax[1] + origin[1], zax[2] + origin[2]])

        self.orig = origin
        self.xend = xend
        self.yend = yend
        self.zend = zend
        self.pose = pose
        self.Rot = R
        self.tf = tf
        self.parent = self
        self.w = 1
        self.len = 0.1

    def copy(self):
        cp = frame(np.array([0,0,0,0,0,0]))
        cp.orig = self.orig
        cp.xend = self.xend
        cp.yend = self.yend
        cp.zend = self.zend
        cp.pose = self.pose
        cp.Rot = self.Rot
        cp.parent = self.parent
        cp.w = self.w
        cp.len = self.len

        return cp

    def transformRT(self, R, t):

        self.orig = MatRot(R, self.orig) + t
        self.xend = MatRot(R, self.xend) + t
        self.yend = MatRot(R, self.yend) + t
        self.zend = MatRot(R, self.zend) + t
        self.Rot = R

    def moveframe(self, pose, ref):

        R1 = Rxyz(pose[3], pose[4], pose[5])
        # R2 = ref.Rot
        # R1 = np.matmul(R1, R2)

        o = MatRot(R1, self.orig-ref.orig)
        x = MatRot(R1, self.xend-ref.orig)
        y = MatRot(R1, self.yend-ref.orig)
        z = MatRot(R1, self.zend-ref.orig)

        #self.orig = np.array([o[0] + pose[0] + ref.orig[0], o[1] + pose[1] + ref.orig[1], o[2] + pose[2] + ref.orig[2]])
        self.orig = np.array([o[0] + pose[0] + ref.orig[0], o[1] + pose[1] + ref.orig[1], o[2] + pose[2] + ref.orig[2]])
        self.xend = np.array([x[0] + pose[0] + ref.orig[0], x[1] + pose[1] + ref.orig[1], x[2] + pose[2] + ref.orig[2]])
        self.yend = np.array([y[0] + pose[0] + ref.orig[0], y[1] + pose[1] + ref.orig[1], y[2] + pose[2] + ref.orig[2]])
        self.zend = np.array([z[0] + pose[0] + ref.orig[0], z[1] + pose[1] + ref.orig[1], z[2] + pose[2] + ref.orig[2]])

        # self.Rot = R
        # self.parent = ref
        # self.pose = pose



        self.Rot = np.matmul(R1, self.Rot)
        self.parent = ref
        self.pose = pose

        trans = np.array([[self.orig[0], self.orig[1], self.orig[2]]])
        tf = np.concatenate((self.Rot,trans.T), axis=1)
        tf = np.concatenate((tf,np.array([[0,0,0,1]])), axis=0)

        self.tf = tf




    def drawframe(self, ax, col):
        orig = self.orig
        endx = self.xend-orig
        endy = self.yend-orig
        endz = self.zend-orig

        axes = np.array([[orig[0], orig[1], orig[2], endx[0], endx[1], endx[2]],
                          [orig[0], orig[1], orig[2], endy[0], endy[1], endy[2]],
                          [orig[0], orig[1], orig[2], endz[0], endz[1], endz[2]]])

        if (col == 'rgb'):
            ax.quiver(axes[0][0], axes[0][1], axes[0][2], axes[0][3], axes[0][4], axes[0][5],
                      pivot='tail', length=self.len, arrow_length_ratio=0.3 / 1, color='red', linewidth=self.w)
            ax.quiver(axes[1][0], axes[1][1], axes[1][2], axes[1][3], axes[1][4], axes[1][5],
                      pivot='tail', length=self.len, arrow_length_ratio=0.3 / 1, color='green', linewidth=self.w)
            ax.quiver(axes[2][0], axes[2][1], axes[2][2], axes[2][3], axes[2][4], axes[2][5],
                      pivot='tail', length=self.len, arrow_length_ratio=0.3 / 1, color='blue', linewidth=self.w)

        else:
            ax.quiver(axes[0][0], axes[0][1], axes[0][2], axes[0][3], axes[0][4], axes[0][5],
                      pivot='tail', length=self.len, arrow_length_ratio=0.3 / 1, color=col, linewidth=self.w)
            ax.quiver(axes[1][0], axes[1][1], axes[1][2], axes[1][3], axes[1][4], axes[1][5],
                      pivot='tail', length=self.len, arrow_length_ratio=0.3 / 1, color=col, linewidth=self.w)
            ax.quiver(axes[2][0], axes[2][1], axes[2][2], axes[2][3], axes[2][4], axes[2][5],
                      pivot='tail', length=self.len, arrow_length_ratio=0.3 / 1, color=col, linewidth=self.w)


    def connectframes(self, ax, frame, col):
        x1 = self.orig[0]
        y1 = self.orig[1]
        z1 = self.orig[2]
        x2 = frame.orig[0]
        y2 = frame.orig[1]
        z2 = frame.orig[2]

        mag = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

        ax.quiver(x1,y1,z1, x2-x1, y2-y1, z2-z1,
                  pivot='tail', length=mag, arrow_length_ratio=0.1 / 1, color=col, linewidth=3)
        # print(mag)



    def transformFrame(self, tf):
        R = np.matrix(np.asarray(tf)[0:3,0:3])
        t = np.asarray(tf)[0:3,3]

        self.orig = MatRot(R, self.orig) + t
        self.xend = MatRot(R, self.xend) + t
        self.yend = MatRot(R, self.yend) + t
        self.zend = MatRot(R, self.zend) + t

    def rotkuka(self, Current, End, Speed, ref, transforms):

        #Speed in deg/s
        #Current is current orientation
        #End is desired final orientation offset from current orientation
        #ref is the reference frame (shoulder, base etc.)
        #transforms it the array of transforms we are appending to

        freq = 200 #frequency of reference commands to kuka

        xR_0 = Current[0]
        yR_0 = Current[1]
        zR_0 = Current[2]

        steps = int((np.max(np.abs(End))*freq)/Speed)
        sz = End/steps


        for i in range(1, steps+1):
            xR = xR_0 + (i * sz[0])
            yR = yR_0 + (i * sz[1])
            zR = zR_0 + (i * sz[2])
            kukEE3 = self.copy()
            kukEE3.moveframe(np.array([0.0, 0.0, 0.0, -xR, -yR, zR]), ref)
            transforms.append(kukEE3.tf)

        return np.array([xR, yR, zR]), transforms #return the last pose that is set

    def Shouldermove(self, sh, pose, ref):

        freq = 200 #frequency of reference commands to kuka
        Speed = 0.05 #m/s

        x_0 = self.orig[0]
        y_0 = self.orig[1]
        z_0 = self.orig[2]

        steps = int((np.max(np.abs(pose)*freq)/Speed))
        print(steps)
        sz = pose/steps
        print(sz)
        print(pose)

        for i in range(1, steps+1):
            x = (i*sz[0])
            y = (i*sz[1])
            z = (i*sz[2])
            # print(x, y, z)
            kukEE3 = self.copy()
            # kukEE3.moveframe(np.array([x, y, z, 0.0, 0.0, 0.0]), ref)
            kukEE3.moveframe(np.array([x, y, z, 0.0, 0.0, 0.0]), ref)
            # sh.moveframe(np.array([x, y, z, 0.0, 0.0, 0.0]), ref)
            transforms.append(kukEE3.tf)
            # print(kukEE3.tf)

        sh.moveframe(np.array([pose[0], pose[1], pose[2], 0, 0, 0]), ref)
        self.moveframe(np.array([pose[0], pose[1], pose[2], 0, 0, 0]), ref)

        return transforms #return the last pose that is set
####################################################################################################################
strainmap_graph = np.load(os.path.join(directory, 'Musculoskeletal Models','Strain Maps', 'Passive', 'All_0_min2AR.npy'))
theta = np.radians(np.arange(-20, 160, 4)).tolist() # Plane Elevation -85 to 180
phi = np.radians(np.arange(0, 144, 4)).tolist() # Shoulder Elevation -44 to 144
psi= np.radians(np.arange(-90, 100, 4)).tolist() # Axial Rotation -90 to 100

thetaM = np.outer(theta, np.ones(36))
phiM = np.outer(phi, np.ones(45)).T

# here we can set a maximum strain that will never be exceeded
# A* will plan only in the graph composed by the nodes whose strain is lower than this
max_strain = 2.2
Slayer = strainmap_graph
Barrier = np.where(Slayer > max_strain)
SMap = np.zeros(shape=Slayer.shape, dtype=int)
SMap[Barrier] = 1

# list the waypoints that we want to traverse
waypoints = [(10, 10), (20, 31), (35, 22), (10, 10)]

# Below the original implementation that allows to actually select the waypoints on the strain map
# ############################ Clickable Figure ###########################

# fig5 = plt.figure()
# ax5 = fig5.add_subplot(111)
# ax5.cla()
# ax5.grid()

# map = 'hot'
# ax5.imshow(Slayer.T, cmap=map, interpolation='nearest')

# plt.gca().set_position([0, 0, 1, 1])
# plt.xlim(0, 45)
# plt.ylim(0, 36)
# ax5.grid(False)
# ax5.set_xticks([])
# ax5.set_yticks([])
# ax5.set_aspect('auto')

# #################waypoints
# implot = ax5.imshow(Slayer.T, cmap=map, interpolation='nearest')

# cid = fig5.canvas.mpl_connect('button_press_event', onclick)

# ax5.set_xticks([])
# ax5.set_yticks([])
# ax5.set_aspect('auto')

# #########################################################################################

# ############################# Wait for desired points to be selected ####################
# while(wait):
#     plt.show()
#     fig5.canvas.draw()
#     fig5.canvas.flush_events()

# plt.show()
# fig5.canvas.draw()
# fig5.canvas.flush_events()

# time.sleep(0.5)
# fig5.canvas.mpl_disconnect(cid)
# ax5.cla()
# plt.close()

############################################################################################

############################# Now begin path planning ######################################

print_maze=True
print("starting")

maze = SMap.tolist()
strain = Slayer.tolist()
wt = 0 #delete this once you fix astar function
paths = []

for pnt in range(0,len(waypoints)-1):
    start = waypoints[pnt]
    end = waypoints[pnt+1]

    print(start)
    print(end)
    temppath = astar(maze, start, end, strain, wt)

    if pnt == 0:
        fullpath = temppath
    else:
        fullpath = np.concatenate((fullpath, temppath), axis=0)

    paths.append(temppath)
    print('path '+ str(pnt+1) +' done!')

pathpnts = ([t[0] for t in fullpath], [t[1] for t in fullpath])
x = pathpnts[0]
y = pathpnts[1]

np.save(pointsfile, fullpath)
np.save(waypointsfile , waypoints)


############################################# 3D Plot ###########################################

fig1 = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot_surface(thetaM, phiM, strainmap_graph, rstride=1, cstride=1, cmap='hot', edgecolor='none', linewidth=0, antialiased=True)

plt.draw()
plt.show()
fig1.canvas.draw()
fig1.canvas.flush_events()

xth = [theta[i] for i in x]
yth = [phi[i] for i in y]
# ax1.plot3D(xth,yth, mstrain2[24][x,y]+1, color = 'white',linewidth=3,markevery=1)

#plot3D above works well on older matplotlib, but adding these patches is necessary on the newer version.

for i in range(0,len(xth)-1):
    p = Circle((xth[i], yth[i]), .03, ec='none', fc="white")
    xy2 = (xth[i],yth[i])
    xy1 = (xth[i+1],yth[i+1])
    C = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                          axesA=ax1, color='white', lw = 3)
    ax1.add_patch(p)
    ax1.add_patch(C)
    art3d.pathpatch_2d_to_3d(C, z=np.mean((strainmap_graph[x[i], y[i]], strainmap_graph[x[i+1], y[i+1]])), zdir="z")
    art3d.pathpatch_2d_to_3d(p, z=strainmap_graph[x[i],y[i]], zdir="z")

plt.draw()
plt.show()
fig1.canvas.draw()
fig1.canvas.flush_events()

##################################################################################################


fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.cla()
ax3.grid()

map = 'hot'

mstrain3 = strainmap_graph[:][:].T

totstrain = 0.0
pos_strain = 0.0
maxstrain = 0.0

for i in fullpath:
    pos_strain = mstrain3[i[1]][i[0]]
    if pos_strain > maxstrain:
        maxstrain = pos_strain

    totstrain = totstrain + pos_strain

print(totstrain)
print(maxstrain)

#Do we want nearest or bilinear interpolation?

# ax.imshow(mstrain3, cmap=map, interpolation='nearest')
ax3.imshow(mstrain3, cmap=map, interpolation='bilinear')
ax3.plot(x, y, linewidth = 3, color = 'white')

for pnt in range(0, len(waypoints)):
    ax3.plot(waypoints[pnt][0], waypoints[pnt][1], marker='o', markersize = 12, color = 'green', markeredgewidth=2)

plt.gca().set_position([0, 0, 1, 1])
plt.xlim(0, 45)
plt.ylim(0, 36)
ax3.grid(False)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_aspect('auto')

plt.draw()
plt.show()
fig3.canvas.draw()
fig3.canvas.flush_events()
time.sleep(1)

############################################ End of Path Planning ################################################


path2d = fullpath

#AxialRotFixed = 24
arvect = np.ones((len(path2d),1),dtype=int)*AxialRotFixed
pathtofollow = np.concatenate((arvect, path2d),axis=1)



## Kuka End Effector when Pointing to the Right on Startup

EEstart = np.matrix([[-0.0049,  0.0000, -1.0   ],
                     [-0.7052, -0.7090,  0.0035],
                     [-0.7090,  0.7052,  0.0035]])

#EEorig = np.array([-0.2831, -0.5645, 0.3407])
#EEorig = np.array([-0.0831, -0.5645, 0.3407])
EEorig = np.array([-0.1749, -0.5664, 0.3299])

##########################################################


## Kuka End Effector when Pointing Down on Startup

EEstartDown = np.matrix([[0.0,  -1.0, 0.0 ],
                         [-1.0, 0.0,  0.0 ],
                         [0.0,  0.0,  -1.0]])

EEorigDown = np.array([0.0, -0.5657, 0.0590])

##########################################################


# Plane Elevation -85 to 180
# Shoulder Elevation -44 to 144
# Axial Rotation -90 to 100

pe_min = -20
se_min = 0
ar_min = -90

pe_res = 4
se_res = 4
ar_res = 4


armposes = []
for p in range(0, len(pathtofollow)):
    ar = ar_min + ar_res * pathtofollow[p][0]
    se = se_min + se_res * pathtofollow[p][2]
    pe = pe_min + pe_res * pathtofollow[p][1]
    armposes.append(np.array([ar, se, pe]))

armrotations = []
prevRot = PatientOrientation  #np.array([0,90,0]) #[0, 90, 90] default
for p in range(0, len(armposes)):
    currRot = armposes[p]
    armrotations.append(currRot-prevRot)
    prevRot = currRot

#then we just need the difference between each point in the path and that will be our motion commands

###########################################################

origin = frame(np.array([0,0,0,0,0,0]))
# sh = frame(np.array([-0.2831-.3,-0.5645,0.3407,0,0,-180]))
origin.w = 2
origin.len = 0.13

origin2 = origin.copy()
origin2.moveframe(np.array([0,0,0,0,-45,0]), origin)


kukEE4 = frame(np.array([-0.1749, -0.5664, 0.3299, 90.0, 45, -90.0]))
kukEE2 = frame(np.array([0.0,0.0,0.0,0.0,0.0, 0.0]))
origin3 = kukEE2.copy()


#### For Pointing to the right on Startup ######
if EE_Down == False:
    kukEE2.transformRT(EEstart, EEorig)
    sh = frame(np.array([-0.1706-L_arm, -0.5664, 0.3299,0,0,-180]))

#### For Pointing down on startup ##############
else:
    kukEE2.transformRT(EEstartDown, EEorigDown)
    sh = frame(np.array([0.0-L_arm, -0.5657, 0.0590-0.07,0,0,-180]))

###############################################
kukEE3 = kukEE2.copy()


transforms = []

transforms = kukEE2.Shouldermove(sh, np.array([pose_x, pose_y, pose_z]), origin) #Set user shoulder position here
initloops = len(transforms)
#
# origin3 = kukEE2.copy()
kukEE3 = kukEE2.copy()

Cur = np.array([0,0,0])
# Rotations of the form [axial rotation, shoulder elevation, planar elevation]
# [0, 0, 0] corresponds to -90 degrees axial rotation, 90 degrees shoulder elevation, and 90 degrees planar elevation
# from OpenSim

if (autotraj == True):
    for eachrotation in armrotations:
        print(eachrotation)
        if np.max(np.abs(eachrotation)) != 0:    #Need to check for zero motion because of duplicate start/end for waypoints
            Cur, transforms = kukEE2.rotkuka(Cur, eachrotation, 5, sh, transforms)

else:
    # Cur, transforms = kukEE2.rotkuka(Cur, np.array([-45, 0, 0]), 5, sh, transforms)
    Cur, transforms = kukEE2.rotkuka(Cur, np.array([0,0,60]), 5, sh, transforms)
    Cur, transforms = kukEE2.rotkuka(Cur, np.array([0,-30, 0]), 5, sh, transforms)
    Cur, transforms = kukEE2.rotkuka(Cur, np.array([0, 0, -60]), 5, sh, transforms)
    Cur, transforms = kukEE2.rotkuka(Cur, np.array([0, 30, 0]), 5, sh, transforms)
    # Cur, transforms = kukEE2.rotkuka(Cur, np.array([0,0,25]), 5, sh, transforms)
# Cur, transforms = kukEE2.rotkuka(Cur, np.array([0,0,-25]), 5, sh, transforms)

totloops = len(transforms)

# #now test out the transforms to make sure they work!

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

origin.drawframe(ax,'rgb')
kukEE4.drawframe(ax,'rgb')

xvals = []
yvals = []
zvals = []


if (drawTransform == True):

    ku = origin.copy()
    filename = pathname + ".csv"
    file = open(filename,"w+")

    for k in range(0,len(transforms), viewspeed):
        ax.clear()
        ku = origin3.copy()
        ku.transformFrame(transforms[k])        #transform each current frame to the next pose
        ku.drawframe(ax,'red')                  #draw the frame
        ku.connectframes(ax,origin, 'red')      #draw the connection


        #draw and other reference geometry/frames we may want
        origin.drawframe(ax, 'rgb')                #draw origin (shoulder frame)
        sh.drawframe(ax, 'rgb')                 #default zero position of elbow (based on opensim zero point)
        # origin.connectframes(ax, kukEE, 'red')     #connect initial position of elbow to origin (shoulder)
        #
        xvals.append(ku.orig[0])
        yvals.append(ku.orig[1])
        zvals.append(ku.orig[2])

        ax.scatter(xvals, yvals, zvals, marker='.')
        # ax.plot3D(xvals, yvals, zvals, marker='.', markevery=200, markersize=1)


        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([-0.2, 1.2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #
        fig.canvas.draw()
        fig.canvas.flush_events()

Firstline = str(initloops) + ", " + str(totloops) + "\n"
file.writelines(Firstline)


if (saveTransform == True):

    for k in transforms:

        ## Now save transforms to a .csv file for parsing in C++
        arr = np.asarray(k).flatten()

        outline = str(arr[0]) + ", " + str(arr[1]) + ", " + str(arr[2]) + ", " + str(arr[3]) + ", " + \
                  str(arr[4]) + ", " + str(arr[5]) + ", " + str(arr[6]) + ", " + str(arr[7]) + ", " + \
                  str(arr[8]) + ", " + str(arr[9]) + ", " + str(arr[10]) + ", " + str(arr[11]) + ", " + \
                  str(arr[12]) + ", " + str(arr[13]) + ", " + str(arr[14]) + ", " + str(arr[15]) + "\n"

        file.writelines(outline)

    file.close()            # Make sure to close the .csv file


########################################################

## may want to save file in a different place later
# cwd = os.getcwd()
# imucwd = cwd + "/imu0.csv"
# csvpath = DatPath + "/imu0.csv"
# copyfile(imucwd, csvpath)

## Come back to these functions

    #
    # def sphereRot(self, pose, ref):
    #     psi = pose[5]
    #     phi = pose[3]
    #     theta = pose[4]
    #     R1 =  Rz(phi * np.pi / 180) * Ry(theta * np.pi / 180) * Rz(psi * np.pi / 180)
    #     # R1 = Rxyz(pose[3], pose[4], pose[5])
    #     # R2 = ref.Rot
    #     # R = np.matmul(R1, R2)
    #
    #     o = MatRot(R1, self.orig - ref.orig)
    #     x = MatRot(R1, self.xend - ref.orig)
    #     y = MatRot(R1, self.yend - ref.orig)
    #     z = MatRot(R1, self.zend - ref.orig)
    #
    #     self.orig = np.array([o[0] + pose[0] + ref.orig[0], o[1] + pose[1] + ref.orig[1], o[2] + pose[2] + ref.orig[2]])
    #     self.xend = np.array([x[0] + pose[0] + ref.orig[0], x[1] + pose[1] + ref.orig[1], x[2] + pose[2] + ref.orig[2]])
    #     self.yend = np.array([y[0] + pose[0] + ref.orig[0], y[1] + pose[1] + ref.orig[1], y[2] + pose[2] + ref.orig[2]])
    #     self.zend = np.array([z[0] + pose[0] + ref.orig[0], z[1] + pose[1] + ref.orig[1], z[2] + pose[2] + ref.orig[2]])
    #
    #     trans = np.array([[self.orig[0], self.orig[1], self.orig[2]]])
    #     tf = np.concatenate((R1,trans.T), axis=1)
    #     tf = np.concatenate((tf,np.array([[0,0,0,1]])), axis=0)
    #
    #     self.Rot = R1
    #     self.parent = ref
    #     self.pose = pose
    #     self.tf = tf
    #
    # def movekuka3(self, ref, angle, ax, col):
    #     ka = self.copy()
    #     # hd = hdframe0.copy()
    #     # ka = kuka.copy()
    #     # el.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     # hd.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     # ka.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), ka)
    #     ka.moveframe(np.array([0,0,0,angle[0], angle[1], angle[2]]), ref)
    #
    #     # el.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     # hd.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     # ka.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #
    #     # el.drawframe(ax)
    #     # hd.drawframe(ax)
    #     ka.drawframe(ax, 'rgb')
    #
    #     # el.connectframes(ax, ref, col)
    #     # hd.connectframes(ax, el, col)
    #     ka.connectframes(ax, ref, col)
    #
    #     return ka
    #
    #
    # def movekuka2(self, ref, angle, ax, col):
    #     ka = self.copy()
    #     # hd = hdframe0.copy()
    #     # ka = kuka.copy()
    #     # el.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     # hd.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     # ka.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), ka)
    #     ka.sphereRot(np.array([0,0,0,angle[0], angle[1], angle[2]]), ref)
    #
    #     # el.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     # hd.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     # ka.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #
    #     # el.drawframe(ax)
    #     # hd.drawframe(ax)
    #     ka.drawframe(ax, 'rgb')
    #
    #     # el.connectframes(ax, ref, col)
    #     # hd.connectframes(ax, el, col)
    #     ka.connectframes(ax, ref, col)
    #
    #     return ka
    #
    # def movekuka(self, ref, angle, ax, col):
    #     ka = self.copy()
    #     # hd = hdframe0.copy()
    #     # ka = kuka.copy()
    #     # el.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     # hd.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     ka.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), ka)
    #     # ka.sphereRot(np.array([0,0,0,angle[0], angle[1], angle[2]]), ref)
    #
    #     # el.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     # hd.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     ka.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #
    #     # el.drawframe(ax)
    #     # hd.drawframe(ax)
    #     ka.drawframe(ax, 'rgb')
    #
    #     # el.connectframes(ax, ref, col)
    #     # hd.connectframes(ax, el, col)
    #     ka.connectframes(ax, ref, col)
    #
    #     return ka
    #
    # def framepath(self, prevframe):
    #
    #     prevInv = np.linalg.inv(prevframe.Rot)
    #     relative_transform = np.matmul(prevInv, self.Rot)
    #
    #     print(relative_transform)

    # def movearm(self, hdframe0, ref, angle, ax, col):
    #     el = self.copy()
    #     hd = hdframe0.copy()
    #     el.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     hd.moveframe(np.array([0, 0, 0, 0, 0, -angle[2]]), el)
    #     el.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     hd.moveframe(np.array([0, 0, 0, 0, angle[1], -angle[0]]), ref)
    #     # hd.moveframe(np.array([0, 0, 0, 0, 0, angle[2]]), el)
    #     el.drawframe(ax, 'rgb')
    #     hd.drawframe(ax, 'rgb')
    #     el.connectframes(ax, ref, col)
    #     hd.connectframes(ax, el, col)