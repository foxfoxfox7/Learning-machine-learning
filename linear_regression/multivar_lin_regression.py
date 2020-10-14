import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


instances = 50
features = 3

theta0 = np.random.randint(0, 9000, 1)
thetai = np.random.randint(1, 9, features)

df_inputs = pd.DataFrame()
df_targets = pd.DataFrame()
df_targets['target'] = [theta0[0]] * instances

for ii in range(features):
    df_inputs[str(ii)] = np.random.randint(1,9,instances)
    df_targets['target'] += (df_inputs[str(ii)] * thetai[ii])

# requires .float() to match the wight and bias tensors created below
targets = torch.tensor(df_targets.values).float()
inputs = torch.tensor(df_inputs.values).float()


def linear_func_mv(x, weights, biases):
    return x @ weights.t() + biases

def cost_func_mv(X, y, thi, th0):
    mm = len(X)
    return (1/2*mm) * torch.sum((linear_func_mv(X, thi, th0) - y)**2)

def mv_linear_regression_alg(inputs, targets, loops, alpha, w_in=None, b_in=None):

    # require_grad = True in order to backwards compute derivatives of
    # the weights and biases
    # number of weight coefficients equal to the number of features
    w = torch.randn(1, inputs.shape[1], requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    if w_in is not None:
        w == w_in
    if b_in is not None:
        b == b_in

    print('initial cost - ', cost_func_mv(inputs, targets, w, b))
    print('Improving parameters...')

    t1 = time.time()
    for i in range(loops):
        cost = cost_func_mv(inputs, targets, w, b)
        cost.backward()
        with torch.no_grad():
            w -= w.grad * alpha
            b -= b.grad * alpha
            w.grad.zero_()
            b.grad.zero_()
    t2 = time.time()

    print(f'...time taken is {t2 - t1} s to complete {loops} loops')
    print('final cost - ', cost_func_mv(inputs, targets, w, b))

    return w, b

w_in = torch.randn(1, features, requires_grad=True)
b_in = torch.randn(1, requires_grad=True)

alpha = 1e-5
loops = 30000
w, b =  mv_linear_regression_alg(inputs, targets, loops, alpha, w_in, b_in)

print('\nRESULTS\n')
calc_targets = linear_func_mv(inputs, w, b).t()
df_targets['results'] = calc_targets[0].detach().numpy()
print('Pearson R - ', df_targets['target'].corr(df_targets['results']))
print('weights - ', thetai)
print('calculated weights - ', w)
print('bias - ', theta0)
print('calculated bias - ', b)

plt.scatter(df_targets['target'].values, df_targets['results'].values)
plt.title('Correlation between calculated and target values')
plt.xlabel('Target values')
plt.ylabel('Calculated values')
plt.show()


