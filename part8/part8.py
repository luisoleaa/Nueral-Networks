import numpy as np
# using categorical cross-entropy loss formula to find loss

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

y_clipped = np.clip(y_pred, le-7)

np_loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(np_loss)
print(average_loss)