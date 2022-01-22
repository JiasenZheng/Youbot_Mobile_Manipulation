import modern_robotics as mr
import numpy as np
from numpy import pi,sin,cos
import matplotlib.pyplot as plt
import logging
"""
To execute the code, in the code file, run the following in the terminal:
    python3 code/code.py
"""
# Initialize log file
LOG_FILENAME = 'results/newTask/newTask.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

def NextState(C_current,V,dt,v_max):
    """
    This function simulates the kinematics of the youBot
    The function NextState is based on a simple first-order Euler step, i.e.,
    new arm joint angles = (old arm joint angles) + (joint speeds) * Δt
    new wheel angles = (old wheel angles) + (wheel speeds) * Δt
    new chassis configuration is obtained from odometry
    Inputs:
        C_current: A 12-vector representing the current configuration of the robot 
        (3 variables for the chassis configuration, 5 variables for the arm configuration, 
        and 4 variables for the wheel angles)
        V: A 9-vector of controls indicating the arm joint speeds (5 variables) and the wheel speeds u (4 variables).
        dt: A timestep Δt
        v_max: A positive real value indicating the maximum angular speed of the arm joints and the wheels.
    Outputs: 
        C_next: A 12-vector representing the configuration of the robot time Δt later

    """
    # Restrict speeds
    for i in range(len(V)):
        if abs(V[i]) > v_max:
            V[i] = (V[i]/abs(V[i])) * v_max
    # Extract from current configs
    chassis_current = C_current[:3]
    arm_current = C_current[3:8]
    wheel_current = C_current[8:]
    # Extract from speeds
    thetadot = V[:5]
    u = V[5:]
    # Calculate the arm and wheel configs at next timestep
    arm_next = arm_current + thetadot*dt
    wheel_next = wheel_current + u*dt
    # Find the F for the mecanum wheel chassis according to formula 13.33
    r = 0.0475
    w = 0.3/2
    l = 0.47/2
    F = r/4 * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                        [1,1,1,1],
                        [-1,1,-1,1]])
    
    Vb = F.dot((u*dt).T).T
    omega_bz = Vb[0]
    v_bx = Vb[1]
    v_by = Vb[2]
    
    if omega_bz == 0:
        dqb = np.array([0, v_bx, v_by])
    else:
        dqb = np.array([omega_bz,
                        (v_bx*sin(omega_bz)+v_by*(cos(omega_bz)-1))/omega_bz,
                        (v_by*sin(omega_bz)+v_bx*(1-cos(omega_bz)))/omega_bz])
    # transforming dq in {b} frame to dq in {s} frame using the chassis angle k
    k = chassis_current[0]
    dqs = np.array([[1,0,0],[0,cos(k),-sin(k)],[0,sin(k),cos(k)]]).dot(dqb)
    chassis_next = chassis_current + dqs
    C_next = np.concatenate((chassis_next,arm_next,wheel_next),axis=None)

    return C_next
    

def TrajectoryGenerator(Tsei,Tsci,Tscf,Tceg,Tces,k):
    """
    The function generates configurations of trajectory points for a gripper grab an object from 
    its inital position to a goal position
    Inputs:
        Tsei: the initial configuration of the end-effector in the reference trajectory
        Tsci: the cube's initial configuration
        Tscf: the cube's desired final configuration
        Tceg: the end-effector's configuration relative to the cube when it is grasping the cube
        Tces: the end-effector's standoff configuration above the cube, before and after grasping, 
        relative to the cube
        k: the number of trajectory reference configurations per 0.01 seconds
    Output:
        L_Tse: the list of end-effector configurations in the reference frame
    """
    L_Tse = []

    # Stage1: A trajectory to move the gripper from its initial configuration to a "standoff" configuration a few cm above the block. 
    Xstart = Tsei 
    Xend = Tsci.dot(Tces)
    Tf = 5.0
    N = Tf*k/0.01
    traj = mr.ScrewTrajectory(Xstart,Xend,Tf,N,method=5)
    gripper_state = 0
    for i in traj:
        R = np.array(i[:3,:3])
        R.resize(1,9)
        p = np.array(i[:3,3])
        p.resize(1,3)
        config = np.append(np.append(R,p),gripper_state)
        L_Tse.append(config) 

    # Stage2: A trajectory to move the gripper down to the grasp position
    Xstart = Tsci.dot(Tces)
    Xend = Tsci.dot(Tceg)
    Tf = 1.0
    N = Tf*k/0.01
    traj = mr.ScrewTrajectory(Xstart,Xend,Tf,N,method=5)
    gripper_state = 0
    for i in traj:
        R = np.array(i[:3,:3])
        R.resize(1,9)
        p = np.array(i[:3,3])
        p.resize(1,3)
        config = np.append(np.append(R,p),gripper_state)
        L_Tse.append(config) 

    # Stage3: Closing of the gripper
    Tf = 1.0
    N = int(Tf*k/0.01)
    gripper_state = 1
    config = L_Tse[-1]
    config[-1] = gripper_state
    for i in range(N):
        L_Tse.append(config)

    # Stage4: A trajectory to move the gripper back up to the "standoff" configuration
    Xstart = Tsci.dot(Tceg)
    Xend = Tsci.dot(Tces)
    Tf = 1.0
    N = Tf*k/0.01
    traj = mr.ScrewTrajectory(Xstart,Xend,Tf,N,method=5)
    gripper_state = 1
    for i in traj:
        R = np.array(i[:3,:3])
        R.resize(1,9)
        p = np.array(i[:3,3])
        p.resize(1,3)
        config = np.append(np.append(R,p),gripper_state)
        L_Tse.append(config)   
    
    # Stage5: A trajectory to move the gripper to a "standoff" configuration above the final configuration
    Xstart = Tsci.dot(Tces)
    Xend = Tscf.dot(Tces)
    Tf = 5.0
    N = Tf*k/0.01
    traj = mr.ScrewTrajectory(Xstart,Xend,Tf,N,method=5)
    gripper_state = 1
    for i in traj:
        R = np.array(i[:3,:3])
        R.resize(1,9)
        p = np.array(i[:3,3])
        p.resize(1,3)
        config = np.append(np.append(R,p),gripper_state)
        L_Tse.append(config)  

    # Stage6: A trajectory to move the gripper to the final configuration of the object
    Xstart = Tscf.dot(Tces)
    Xend = Tscf.dot(Tceg)
    Tf = 1.0
    N = Tf*k/0.01
    traj = mr.ScrewTrajectory(Xstart,Xend,Tf,N,method=5)
    gripper_state = 1
    for i in traj:
        R = np.array(i[:3,:3])
        R.resize(1,9)
        p = np.array(i[:3,3])
        p.resize(1,3)
        config = np.append(np.append(R,p),gripper_state)
        L_Tse.append(config) 

    # Stage7: Opening of the gripper
    Tf = 1.0
    N = int(Tf*k/0.01)
    gripper_state = 0
    config = L_Tse[-1]
    config[-1] = gripper_state
    for i in range(N):
        L_Tse.append(config)

    # Stage8: A trajectory to move the gripper back to the "standoff" configuration
    Xstart = Tscf.dot(Tceg)
    Xend = Tscf.dot(Tces)
    Tf = 1.0
    N = Tf*k/0.01
    traj = mr.ScrewTrajectory(Xstart,Xend,Tf,N,method=5)
    gripper_state = 0
    for i in traj:
        R = np.array(i[:3,:3])
        R.resize(1,9)
        p = np.array(i[:3,3])
        p.resize(1,3)
        config = np.append(np.append(R,p),gripper_state)
        L_Tse.append(config) 

    return L_Tse

def FeedbackControl(X,Xd,Xd_next,Kp,Ki,dt,C_current,Xerr_i):
    """
    The function simulate the motion of the robot and generate a reference trajectory for the end-effector
    Inputs:
        X: The current actual end-effector configuration X (also written Tse).
        Xd: The current end-effector reference configuration Xd (i.e., Tse,d).
        Xd_next: The end-effector reference configuration at the next timestep in the reference trajectory, Xd,next (i.e., Tse,d,next), at a time Δt later.
        Kp: The Kp gain matrix.
        Ki: The Ki gain matrix.
        dt: The timestep Δt between reference trajectory configurations. 
        Xerr_i: The intergration of errors 
    Output:
        V: The commanded end-effector twist V expressed in the end-effector frame {e}.

    """
    
    # The fixed offset from the chassis frame {b} to the base frame of the arm {0}
    Tb0 = np.array([[1,0,0,0.1662],
                    [0,1,0,0],
                    [0,0,1,0.0026],
                    [0,0,0,1]])

    # The end-effector frame {e} relative to the arm base frame {0}
    M0e = np.array([[1,0,0,0.033],
                    [0,1,0,0],
                    [0,0,1,0.6546],
                    [0,0,0,1]])

    # The screw axes B for the five joints are expressed in the end-effector frame {e}
    Blist = np.array([[0,0,1,0,0.033,0],
                      [0,-1,0,-0.5076,0,0],
                      [0,-1,0,-0.3526,0,0],
                      [0,-1,0,-0.2176,0,0],
                      [0,0,1,0,0,0]]).T
    arm_joints = C_current[3:8]
    r = 0.0475
    w = 0.3/2
    l = 0.47/2
    # Define the 6*m F matrix
    F = r/4 * np.array([[0,0,0,0],
                        [0,0,0,0],
                        [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                        [1,1,1,1],
                        [-1,1,-1,1],
                        [0,0,0,0]])
    
    Vd = mr.se3ToVec(mr.MatrixLog6(np.linalg.inv(Xd)@Xd_next)/dt)
    print("Vd:",Vd)
    Ad = mr.Adjoint(np.linalg.inv(X)@Xd)
    AdVd = Ad@Vd
    print("AdVd:",AdVd)
    
    Xerr = mr.se3ToVec(mr.MatrixLog6((np.linalg.inv(X))@Xd))
    print("Xerr:",Xerr)
    Xerr_i += Xerr*dt


    V = AdVd + Kp@Xerr + Ki@Xerr_i
    print("V:",V)
    # The transformation matrix of ee wrt frame 0 
    T0e = mr.FKinBody(M0e, Blist, arm_joints)
    # The transformation matrix of chassis body frame wrt to ee
    Te0 = np.linalg.inv(T0e)
    T0b = np.linalg.inv(Tb0)
    Teb = Te0.dot(T0b)
    # The arm, body and end-effector Jacobian matrices:
    Ja = mr.JacobianBody(Blist, arm_joints)
    Jb = (mr.Adjoint(Teb)).dot(F)
    Je = np.concatenate((Jb, Ja), axis=1)
    print("Je =")
    print(Je)

    # The wheel and arm joint speeds:
    Je_inv = np.linalg.pinv(Je)
    Vs = Je_inv@V
    print("Vs =")
    print(Vs)

    return V,Vs,Xerr,Xerr_i

def main_TG():
    """
    The main function for TrajectoryGenerator
    """
    # Initialize the arguments
    Tsei = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.6546],[0,0,0,1]])
    Tsci = np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0.025],[0,0,0,1]])
    Tscf = np.array([[0,1,0,0],[-1,0,0,-1],[0,0,1,0.025],[0,0,0,1]])
    theta = pi/2 + pi/4
    Tces = np.array([[cos(theta),0,sin(theta),0],[0,1,0,0],[-sin(theta),0,cos(theta),0.05],[0,0,0,1]])
    Tceg = np.array([[cos(theta),0,sin(theta),0.01],[0,1,0,0],[-sin(theta),0,cos(theta),-0.01],[0,0,0,1]])
    k = 10
    traj = TrajectoryGenerator(Tsei,Tsci,Tscf,Tceg,Tces,k)
    np.savetxt("traj.csv", traj , delimiter=",")
def main_NS():
    """
    The main function for NextState
    """
    # Test with different velocities
    C_current = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    V = np.array([0,0,0,0,0,-10,10,10,-10])
    dt = 0.01
    t = 1
    v_max = 10
    n = int(t/dt)
    C_list = []
    C_list.append(C_current)
    for i in range(n):
        C_next = NextState(C_current,V,dt,v_max)
        C_current = C_next
        C_list.append(C_next)
    np.savetxt("c3.csv",C_list,delimiter=",")

def main_FC():
    """
    The main function for FeedbackControl
    """
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # Test FeedbackControl Function
    C_current = np.array([0,0,0,0,0,0.2,-1.6,0])
    X = np.array([[0.170,0,0.985,0.387],[0,1,0,0],[-0.985,0,0.170,0.570],[0,0,0,1]])
    Xd = np.array([[0,0,1,0.5],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])
    Xd_next = np.array([[0,0,1,0.6],[0,1,0,0],[-1,0,0,0.3],[0,0,0,1]])
    Kp = np.eye(6)*1
    Ki = np.eye(6)*0
    dt = 0.01
    Xerr_i = np.zeros(6)
    V, Vs,Xerr,Xerr_i = FeedbackControl(X,Xd,Xd_next,Kp,Ki,dt,C_current,Xerr_i)

def main():
    """
    The main function
    """
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # The initial configuration of the end-effector reference trajectory
    Tsei = np.array([[0,0,1,0],
                     [0,1,0,0],
                     [-1,0,0,0.5],
                     [0,0,0,1]])
    # The initial configuration of the cube 
    # Tsci = np.array([[1,0,0,1],
    #                  [0,1,0,0],
    #                  [0,0,1,0.025],
    #                  [0,0,0,1]])
    Tsci = np.array([[1,0,0,0.5],
                    [0,1,0,0.5],
                    [0,0,1,0.025],
                    [0,0,0,1]])
    # The goal configuration of the cube
    # Tscf = np.array([[0,1,0,0],
    #                  [-1,0,0,-1],
    #                  [0,0,1,0.025],
    #                  [0,0,0,1]])

    Tscf = np.array([[0,1,0,0],
                     [-1,0,0,-0.8],
                     [0,0,1,0.025],
                     [0,0,0,1]])
    # The angle between the finger and the cube
    theta = pi/2 + pi/4
    # The transformation of end-effector wrt cube at the standoff position
    Tces = np.array([[cos(theta),0,sin(theta),0],
                     [0,1,0,0],
                     [-sin(theta),0,cos(theta),0.1],
                     [0,0,0,1]])
    # The transformation of end-effector wrt cube at the grasp position
    Tceg = np.array([[cos(theta),0,sin(theta),0.01],
                     [0,1,0,0],
                     [-sin(theta),0,cos(theta),-0.01],
                     [0,0,0,1]])
    # The fixed offset from the chassis frame {b} to the base frame of the arm {0}
    Tb0 = np.array([[1,0,0,0.1662],
                    [0,1,0,0],
                    [0,0,1,0.0026],
                    [0,0,0,1]])
    # The end-effector frame {e} relative to the arm base frame {0}
    M0e = np.array([[1,0,0,0.033],
                    [0,1,0,0],
                    [0,0,1,0.6546],
                    [0,0,0,1]])
    # The screw axes B for the five joints are expressed in the end-effector frame {e}
    Blist = np.array([[0,0,1,0,0.033,0],
                      [0,-1,0,-0.5076,0,0],
                      [0,-1,0,-0.3526,0,0],
                      [0,-1,0,-0.2176,0,0],
                      [0,0,1,0,0,0]]).T

    # Ki and Kp
    Kp_gain = 15
    Ki_gain = 3
    Kp = np.eye(6)*Kp_gain
    Ki = np.eye(6)*Ki_gain
    # Initializations
    C_list = []
    Xerr_i = np.zeros(6)
    k = 1
    dt = 0.01
    t = 16
    N = int(t/dt)
    Xerr_list = np.zeros((N-1, 6))
    # Initial configurations with some errors
    C_initial = np.array([0.6, -0.2, -0.2, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0])
    C_list.append(C_initial)

    logging.info('Generating reference trajectory')
    # Generates configurations
    traj = TrajectoryGenerator(Tsei = Tsei,
                               Tsci = Tsci,
                               Tscf = Tscf,
                               Tceg = Tceg,
                               Tces = Tces,
                               k    = k )
    logging.info('Generating configurations and Xerrors')
    for i in range(N-1):
        C_current = C_list[-1]
        phi = C_current[0]
        x = C_current[1]
        y = C_current[2]
        Tsb = np.array([[cos(phi), -sin(phi),0, x],
                        [sin(phi),cos(phi),0,y],
                        [0,0,1,0.0963],
                        [0,0,0,1]])
        theta_list = C_current[3:8]
        T0e = mr.FKinBody(M0e,Blist,theta_list)
        Tbe = Tb0@T0e
        X = Tsb@Tbe

        Xd = np.array([[traj[i][0],traj[i][1],traj[i][2],traj[i][9]],
                       [traj[i][3],traj[i][4],traj[i][5],traj[i][10]],
                       [traj[i][6],traj[i][7],traj[i][8],traj[i][11]],
                       [0,0,0,1]])

        Xd_next = np.array([[traj[i+1][0],traj[i+1][1],traj[i+1][2],traj[i+1][9]],
                            [traj[i+1][3],traj[i+1][4],traj[i+1][5],traj[i+1][10]],
                            [traj[i+1][6],traj[i+1][7],traj[i+1][8],traj[i+1][11]],
                            [0,0,0,1]])

        V,Vs, Xerr, Xerr_i = FeedbackControl(X,Xd,Xd_next,Kp,Ki,dt,C_current,Xerr_i)
        Vs = np.concatenate((Vs[4:], Vs[:4]), axis=None)

        C_next = NextState(C_current[:12],Vs,dt,v_max = 10)
        C_current = C_next
        C_next = np.concatenate((C_next,traj[i][-1]),axis=None)
        C_list.append(C_next)
        Xerr_list[i] = Xerr
    logging.info('Storing configurations as csv file')
    np.savetxt("C_list.csv",C_list,delimiter=",")
    logging.info('Storing Xerror as csv file')
    np.savetxt("Xerr_list.csv",Xerr_list,delimiter=",")

    logging.info("Plotting the Xeror")
    t_val = np.linspace(0,15.99,1599)
    plt.figure()
    plt.plot(t_val,Xerr_list[:,0], label='Xerr_0')
    plt.plot(t_val,Xerr_list[:,1], label='Xerr_1')
    plt.plot(t_val,Xerr_list[:,2], label='Xerr_2')
    plt.plot(t_val,Xerr_list[:,3], label='Xerr_3')
    plt.plot(t_val,Xerr_list[:,4], label='Xerr_4')
    plt.plot(t_val,Xerr_list[:,5], label='Xerr_5')
    plt.title(f'Xerr vs time with a Kp of {Kp_gain} and a Ki of {Ki_gain}')
    plt.xlabel('Time/s')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(f'Kp={Kp_gain}&Ki={Ki_gain}.png')
    logging.info("Done!")
    plt.show()


        


if __name__ == "__main__":
    main()