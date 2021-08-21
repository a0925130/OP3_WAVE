import time
import numpy as np
import pybullet_data
import pybullet as p
from niapy.benchmarks import Benchmark
from niapy.task import StoppingTask, OptimizationType
from niapy.algorithms.basic import ParticleSwarmOptimization
import matplotlib.pyplot as plt

l_sho_pitch = 12
l_sho_roll = 13
l_el = 14

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")
op3StartPos = [0, 0, 1]
op3StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF("robotis_op3.urdf", op3StartPos, op3StartOrientation)
circleStartPos = [0.14820810926653658, 0.11138307132884889, 0.32432147609577605]
circle = p.loadURDF("OP3_WAVE/ball.urdf", circleStartPos)

numJoints = p.getNumJoints(robot)
for joint in range(numJoints):
    print(p.getJointInfo(robot, joint))
    p.setJointMotorControl(robot, joint, p.POSITION_CONTROL, 0, 100)

motor = [-5, -10, -15, -20, -25, -30, 5, 10, 15, 20, 25, 30, -3, -6, -9, -12, -15, -18]
best_val = 1


def run(sol):
    for i in range(6):
        p.setJointMotorControl2(bodyUniqueId=robot, jointIndex=l_sho_pitch, controlMode=p.POSITION_CONTROL,
                                targetVelocity=sol[i], force=10)
        p.setJointMotorControl2(bodyUniqueId=robot, jointIndex=l_sho_roll, controlMode=p.POSITION_CONTROL,
                                targetVelocity=sol[i + 6], force=10)
        p.setJointMotorControl2(bodyUniqueId=robot, jointIndex=l_el, controlMode=p.POSITION_CONTROL,
                                targetVelocity=sol[i + 12], force=10)
        # time.sleep(0.05)


class MyBenchmark(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, -1, 1)
        self.fitness_list = []
        self.best_motor = []
        self.best_val = best_val

    def function(self):
        def evaluate(D, sol):
            run(np.array(motor) - np.array(sol))
            # time.sleep(0.1)
            pos = p.getLinkState(robot, l_el)
            val = np.linalg.norm((np.array(pos[0]) - np.array(circleStartPos)))
            if val < self.best_val:
                self.best_motor = sol
                self.best_val = val
            self.fitness_list.append(val)
            p.resetBasePositionAndOrientation(bodyUniqueId=robot, posObj=StartPos[0], ornObj=op3StartOrientation)
            return val

        return evaluate


p.setRealTimeSimulation(1)
time.sleep(2)
StartPos = p.getBasePositionAndOrientation(bodyUniqueId=robot)

my_bench = MyBenchmark()
ngen = 20
np1 = 10

task = StoppingTask(dimension=18, max_iters=ngen, optimization_type=OptimizationType.MINIMIZATION, benchmark=my_bench, enable_logging=True)
algo = ParticleSwarmOptimization(NP=np1, vMin=-1, vMax=1)
best = algo.run(task)

generator = []
fitness = []
sum_fitness = 0
for i in range(ngen):
    fitness.append(my_bench.fitness_list[i * np1])
    generator.append(i)
    for j in range(np1):
        sum_fitness += my_bench.fitness_list[(i * np1) + j]
    fitness[i] = sum_fitness/np1
    sum_fitness = 0

print(fitness)
print(generator)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(generator, fitness, color='yellow', linewidth=3, marker='o')
plt.xlim(-1, ngen)
plt.savefig('homework.png')
plt.show()
time.sleep(1)

print(my_bench.best_motor)
run(np.array(motor) - np.array(my_bench.best_motor))

while 1:
    time.sleep(1. / 240.)
