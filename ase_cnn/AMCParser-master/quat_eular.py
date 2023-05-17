
from scipy.spatial.transform import Rotation as R
import numpy as np


a = R.from_quat([-0.01799768, -0.97712515,  0.21110202,  0.01839761]).as_matrix()
ainv = np.linalg.inv(a)
b = R.from_matrix(a).as_quat()
print(b)
print(a.dot(ainv))