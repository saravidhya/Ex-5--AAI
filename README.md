<H1 ALIGN =CENTER> Implementation of Kalman Filter</H1>

<H3>NAME: Vidhiya Lakshmi S </H3>
<H3>REGISTER NO: 212223230238</H3>
<H3>EX. NO.5</H3>
<H3>DATE: 24/8/2026</H3>

<H3>Aim:</H3> To Construct a Python Code to implement the Kalman filter to predict the position and velocity of an object.

<H3>Algorithm:</H3>

Step 1: Define the state transition model F, the observation model H, the process noise covariance Q, the measurement noise covariance R, the initial state estimate x0, and the initial error covariance P0.<BR>
Step 2:  Create a KalmanFilter object with these parameters.<BR>
Step 3: Simulate the movement of the object for a number of time steps, generating true states and measurements. <BR>
Step 3: For each measurement, predict the next state using kf.predict().<BR>
Step 4: Update the state estimate based on the measurement using kf.update().<BR>
Step 5: Store the estimated state in a list.<BR>
Step 6: Plot the true and estimated positions.<BR>

<H3>Program:</H3>

```python
import numpy as np
import matplotlib.pyplot as plt
class KalmanFilter:
    def __init__(self,F,H,Q,R,p0,x0):
        self.F=F
        self.H=H
        self.Q=Q
        self.R=R
        self.P=p0
        self.X=x0
    def predict(self):
        self.X=self.F@self.X # x k+1=F xk
        self.P=self.F@self.P@self.F.T+self.Q #Pk+1=FPF.T+Q
    def update(self,z):
        y=z-self.H@self.X # y=z-HX
        s=self.H@self.P@self.H.T+self.R  #HPH.T+R
        K=self.P@self.H.T@np.linalg.inv(s)
        self.X=self.X+K@y
        self.P=np.dot(np.eye(self.F.shape[0])-np.dot(K,self.H),self.P)
        #P1z=(I-KH) Pz
        
        
        
dt=0.1
F=np.array([[1,dt],[0,1]])
H=np.array([[1,0]])
Q=np.diag([0.1,0.1])
R=np.array([[1]])
p0=np.diag([1,1])
x0=np.array([0,0])
truestates=[]
measurements=[]
kf=KalmanFilter(F,H,Q,R,p0,x0)
for i in range(100):
    truestates.append([i*dt,1])
    measurements.append(i*dt+np.random.normal(scale=1))
estimatedstates=[]
for z in measurements:
    kf.predict()
    kf.update(np.array([z]))
    estimatedstates.append(kf.X)
plt.plot([s[0] for s in truestates],label="True States")
plt.plot(measurements,label="Measurements")
plt.plot([s[0]for s in estimatedstates],label="Estimated States")
plt.legend()
plt.show()

```
<H3>Output:</H3>

![alt text](output/graph.png)

<H3>Results:</H3>
Thus, Kalman filter is implemented to predict the next position and   velocity in Python
