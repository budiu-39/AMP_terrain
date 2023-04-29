import pybullet
import pybullet_data
import time
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pd.getDataPath())

# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# offset = 0
# p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
# for scale in range (1,10,1):
#   ball = p.loadURDF("soccerball.urdf",[offset,0,1], globalScaling=scale*0.1)
#   p.changeDynamics(ball,-1,linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001)
#   p.changeVisualShape(ball,-1,rgbaColor=[0.8,0.8,0.8,1])
#   offset += 2*scale*0.1
# p.loadURDF("scene.urdf", (0, 0, 0.1), p.getQuaternionFromEuler([1.57, 0, 0]))
# p.setGravity(0,0,-10)
# p.setRealTimeSimulation(1)
# while p.isConnected():
#   #p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#   time.sleep(0.5)


if __name__ == '__main__':
    client = pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
    # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0)

    pybullet.setGravity(0, 0, -9.8)
    pybullet.setRealTimeSimulation(1)
     # 载入urdf格式是场景
    pybullet.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    # 载入urdf格式的机器人
    r_ind = pybullet.loadURDF('scene.urdf', (0, 0, 0.1), pybullet.getQuaternionFromEuler([1.57, 0, 0]))

    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    while True:
        time.sleep(1. / 240.)

# p.disconnect()
