import pybullet as p
import pybullet_data
import time
import math


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setTimeStep(1. / 120.)
# useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
p.loadURDF("plane100.urdf", useMaximalCoordinates=True)


shift = [0, 0, 0]
meshScale = [0.1, 0.1, 0.1]
# the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName="duck.obj",
                                    rgbaColor=[1, 1, 1, 1],
                                    specularColor=[0.4, .4, 0],
                                    visualFramePosition=shift,
                                    meshScale=meshScale)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName="duck_vhacd.obj",
                                          collisionFramePosition=shift,
                                          meshScale=meshScale)

rangex = 5
rangey = 5
for i in range(rangex):
  for j in range(rangey):
    p.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=[((-rangex / 2) + i) * meshScale[0] * 2,
                                    (-rangey / 2 + j) * meshScale[1] * 2, 1],
                      useMaximalCoordinates=True)

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)


while(1):
    sleeptime(1./240.)


