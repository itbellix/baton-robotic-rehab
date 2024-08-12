# Collection of utility functions that reimplement Simbody methods not available through the OpenSim Python API

import opensim as osim
import numpy as np


def shiftFromMassCenter(I, p, mass):
    """
    Assuming that the current inertia I is a central inertia (that is, it is
    inertia about the body center of mass CF), shift it to some other point p
    measured from the center of mass. This produces a new inertia I' about
    the point p given by I' = I + Ip where Ip is the inertia of a fictitious
    point mass of mass mtot (the total body mass) located at p, about CF.

    INPUTS:
        I: current inertia to be shifted (expected type simbody.Inertia)
        p: point to shift I to
        mass: mass of the body

    Re-implemented from: https://github.com/simbody/simbody/blob/252dfb5957b853e181555c63a546854a57ad267e/SimTKcommon/Mechanics/include/SimTKcommon/internal/MassProperties.h#L343
    """
    return sumInertias(I, pointMassAt(p, mass))

def pointMassAt(p, m):
    """
    Create an Inertia matrix (of type simbody.Inertia) for a point of a given mass, located at 
    a given location measured from the origin of the implicit F frame.
    INPUTS:
        p: location of the point-mass (it is a 3D vector)
        m: mass of the body

    Re-implemented from: https://github.com/simbody/simbody/blob/252dfb5957b853e181555c63a546854a57ad267e/SimTKcommon/Mechanics/include/SimTKcommon/internal/MassProperties.h#L458
    """
    mp = m*p
    mxx = mp[0]*p[0]
    myy = mp[1]*p[1]
    mzz = mp[2]*p[2]
    nmx = -mp[0]
    nmy = -mp[1]
    return osim.simbody.Inertia( myy+mzz,  mxx+mzz,  mxx+myy, nmx*p[1], nmx*p[2], nmy*p[2])


def sumInertias(I1, I2):
    """
    Add in another inertia matrix. Frames and reference point must be the same but
    we can't check.

    INPUTS:
        I1: first inertia  (expected type simbody.Inertia)
        I2: second inertia (expected type simbody.Inertia)

    Re-implemented from: https://github.com/simbody/simbody/blob/252dfb5957b853e181555c63a546854a57ad267e/SimTKcommon/Mechanics/include/SimTKcommon/internal/MassProperties.h#L288
    """
    mxx_1, myy_1, mzz_1 = I1.getMoments().to_numpy()
    mxy_1, mxz_1, myz_1 = I1.getProducts().to_numpy()

    mxx_2, myy_2, mzz_2= I2.getMoments().to_numpy()
    mxy_2, mxz_2, myz_2 = I2.getProducts().to_numpy()
    
    mxx = mxx_1 + mxx_2
    myy = myy_1 + myy_2
    mzz = mzz_1 + mzz_2
    mxy = mxy_1 + mxy_2
    mxz = mxz_1 + mxz_2
    myz = myz_1 + myz_2

    return osim.simbody.Inertia(mxx, myy, mzz, mxy, mxz, myz)


def rotateInertia(I, R):
    """
    Rotate an inertia matrix to express it into another frame, that has the same origin location as the original one.
    The formula applied is I_new_frame = R x I_old_frame x R^T, where R is the rotation matrix that transforms the
    original frame in the new one.

    INPUTS:
        I: inertia matrix to be transformed, expressed with respect to frame "old" (expected type simbody.Inertia)
        R: rotation matrix applied to obtain frame "new" starting from frame "old" (expected type simbody.Mat33, from simbody.Rotation using method toMat33())
    """
    I_numpy = np.zeros((3,3))
    I_numpy[0,0], I_numpy[1,1], I_numpy[2,2] = I.getMoments().to_numpy()
    I_numpy[0,1], I_numpy[0,2], I_numpy[1,2] = I.getProducts().to_numpy()
    I_numpy[1,0] = I_numpy[0,1]
    I_numpy[2,0] = I_numpy[0,2]
    I_numpy[2,1] = I_numpy[1,2]

    R_numpy = np.zeros((3,3))
    R_numpy[0,0] = R.get(0,0)
    R_numpy[1,1] = R.get(1,1)
    R_numpy[2,2] = R.get(2,2)
    R_numpy[0,1] = R.get(0,1)
    R_numpy[0,2] = R.get(0,2)
    R_numpy[1,0] = R.get(1,0)
    R_numpy[1,2] = R.get(1,2)
    R_numpy[2,0] = R.get(2,0)
    R_numpy[2,1] = R.get(2,1)
    
    I_new_numpy = np.matmul(R_numpy, np.matmul(I_numpy, np.transpose(R_numpy)))

    I_new_simbody = osim.simbody.Inertia(I_new_numpy[0,0], I_new_numpy[1,1], I_new_numpy[2,2], I_new_numpy[0,1], I_new_numpy[0,2], I_new_numpy[1,2])

    return I_new_simbody


def fromMat33toNumpyArray(R):
    """
    Transforms a rotation matrix expressed as a simbody.Mat33 object into a numpy array.
    """
    R_numpy = np.zeros((3,3))
    R_numpy[0,0] = R.get(0,0)
    R_numpy[1,1] = R.get(1,1)
    R_numpy[2,2] = R.get(2,2)
    R_numpy[0,1] = R.get(0,1)
    R_numpy[0,2] = R.get(0,2)
    R_numpy[1,0] = R.get(1,0)
    R_numpy[1,2] = R.get(1,2)
    R_numpy[2,0] = R.get(2,0)
    R_numpy[2,1] = R.get(2,1)

    return R_numpy


def fromInertiaToNumpyArray(I):
    """
    Transforms a simbody.Inertia matrix into a 3x3 numpy array
    """
    I_numpy = np.zeros((3,3))
    I_numpy[0,0], I_numpy[1,1], I_numpy[2,2] = I.getMoments().to_numpy()
    I_numpy[0,1], I_numpy[0,2], I_numpy[1,2] = I.getProducts().to_numpy()
    I_numpy[1,0] = I_numpy[0,1]
    I_numpy[2,0] = I_numpy[0,2]
    I_numpy[2,1] = I_numpy[1,2]

    return I_numpy


def convertAngAccInParentToBodyXYZDotDot(cosxy, sinxy, oocosy, qdot, b_PB):
    """
    Calculate second time derivative qdotdot of body-fixed XYZ Euler 
    angles q given sines and cosines of the Euler angles, the first  
    derivative qdot and the angular acceleration b_PB of child B in  
    parent P, expressed in P.
    
    Re-implemented from: https://github.com/simbody/simbody/blob/0d671660c4e97f26566da040eaa57089fab81192/SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L1041
    """
    s1 = sinxy[1]
    c1 = cosxy[1]
    q0 = qdot[0]
    q1 = qdot[1]
    q2 = qdot[2]

    Nb = multiplyByBodyXYZ_N_P(cosxy, sinxy, oocosy, b_PB)

    q1oc1 = q1*oocosy
    NDotw = np.array([(q0*s1-q2)*q1oc1, q0*q2*c1,(q2*s1-q0)*q1oc1])

    return Nb+NDotw


def multiplyByBodyXYZ_N_P(cosxy, sinxy, oocosy, w_PB):
    """
    This is the fastest way to form the product qdot=N_P*w_PB for a 
    body-fixed XYZ sequence where angular velocity of child in parent is 
    expected to be expressed in the parent. Here we assume you have
    previously calculated sincos(qx), sincos(qy), and 1/cos(qy).
    """
    s0 = sinxy[0]
    s1 = sinxy[1]
    c0 = cosxy[0]
    w0 = w_PB[0]
    w1 = w_PB[1]
    w2 = w_PB[2]

    t = (s0*w1-c0*w2)*oocosy

    return np.array([w0 + t*s1, c0*w1 + s0*w2, -t])
