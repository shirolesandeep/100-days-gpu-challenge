import torch
import torch.cuda.amp
import uuid

class LBFGSOptimizer(torch.optim.Optimizer):
    def __init__(self, params, m=5, lr=1.0, c1=1e-4, c2=0.9, max_iter=100):
        defaults = dict(lr=lr, m=m, c1=c1, c2=c2, max_iter=max_iter)
        super(LBFGSOptimizer, self).__init__(params, defaults)
        self.m = m
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.state = dict()
        for param_group in self.param_groups:
            for p in param_group['params']:
                self.state[p] = {'s': [], 'y': [], 'rho': [], 'curr_m': 0}

    def two_loop_recursion(self, grad, s, y, rho, curr_m):
        q = grad.clone()
        alpha = torch.zeros(curr_m, device=grad.device, dtype=grad.dtype)

        # First loop
        for i in range(curr_m - 1, -1, -1):
            alpha[i] = rho[i] * torch.dot(s[i], q)
            q -= alpha[i] * y[i]

        # Apply initial Hessian approximation (identity scaling)
        gamma = 1.0  # Simplified: assume initial Hessian is identity
        p = -gamma * q

        # Second loop
        for i in range(curr_m):
            beta = rho[i] * torch.dot(y[i], p)
            p += s[i] * (alpha[i] - beta)

        return p

    def line_search(self, closure, x, p, grad, loss, max_ls_iter=10):
        alpha = 1.0
        grad_p = torch.dot(grad, p)

        for _ in range(max_ls_iter):
            with torch.no_grad():
                x_new = x + alpha * p
                self.zero_grad()
                with torch.cuda.amp.autocast():
                    new_loss = closure(x_new)
                if new_loss <= loss + self.c1 * alpha * grad_p:
                    with torch.no_grad():
                        new_grad = torch.autograd.grad(new_loss, x_new)[0]
                    new_grad_p = torch.dot(new_grad, p)
                    if abs(new_grad_p) <= self.c2 * abs(grad_p):
                        return alpha, new_loss, new_grad
            alpha *= 0.5

        return 0.0, loss, grad  # Line search failed

    def step(self, closure):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                s = state['s']
                y = state['y']
                rho = state['rho']
                curr_m = state['curr_m']

                # Evaluate closure to get loss and gradient
                with torch.cuda.amp.autocast():
                    loss = closure(p)
                    grad = p.grad

                # Compute search direction
                if curr_m > 0:
                    p_dir = self.two_loop_recursion(grad, s, y, rho, curr_m)
                else:
                    p_dir = -grad

                # Line search
                alpha, new_loss, new_grad = self.line_search(closure, p, p_dir, grad, loss)

                if alpha > 0:
                    # Update parameters
                    with torch.no_grad():
                        s_new = alpha * p_dir
                        y_new = new_grad - grad
                        y_s = torch.dot(y_new, s_new)
                        rho_new = 1.0 / y_s if y_s > 1e-10 else 0.0

                        # Update history
                        if curr_m < self.m:
                            s.append(s_new.clone())
                            y.append(y_new.clone())
                            rho.append(rho_new)
                            state['curr_m'] += 1
                        else:
                            s.pop(0)
                            y.pop(0)
                            rho.pop(0)
                            s.append(s_new.clone())
                            y.append(y_new.clone())
                            rho.append(rho_new)

                        p.add_(s_new)

                # Check convergence
                grad_norm = torch.norm(grad)
                if grad_norm < 1e-5:
                    return loss

        return loss

# Example usage: Quadratic loss function
def quadratic_loss(x, A, b):
    return 0.5 * torch.dot(x, A @ x) - torch.dot(b, x)

def main():
    # Problem setup
    N = 256  # Problem dimension
    M = 5    # L-BFGS history size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize data
    x = torch.zeros(N, device=device, dtype=torch.float32, requires_grad=True)
    A = torch.eye(N, device=device) * 2.0  # Positive definite matrix
    b = torch.ones(N, device=device)

    # Closure for loss computation
    def closure(x_param=x):
        if x_param.grad is not None:
            x_param.grad.zero_()
        with torch.cuda.amp.autocast():
            loss = quadratic_loss(x_param, A, b)
        loss.backward()
        return loss

    # Initialize optimizer
    optimizer = LBFGSOptimizer([x], m=M)

    # Optimization loop
    for i in range(100):
        optimizer.zero_grad()
        loss = optimizer.step(closure)
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

        # Check convergence
        if torch.norm(x.grad) < 1e-5:
            print(f"Converged after {i} iterations")
            break

    # Print result
    print("Optimized x (first 5 elements):", x[:5].detach().cpu().numpy())

if __name__ == "__main__":
    main()