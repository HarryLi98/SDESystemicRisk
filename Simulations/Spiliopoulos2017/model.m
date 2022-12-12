classdef model
    % Object-oriented approach to model Spiliopoulos 
    % with options for common-noise and network structures

    properties
        dt      % Time increment
        t       % Vector of times from 0 to T by dt
        N       % Number of Banks
        x0      % Initial condition
        x       % Dynamics of N banks
        xbar    % Mean of N banks
        alpha   % Drift coefficient - scalar or Nx1 vector
        sig     % Diffusion coefficient - scalar or Nx1 vector
        f       % Drift term - function handle
        g       % Diffusion term - function handle
        opts    % sde_euler options
        eta     % Default level
        rho     % Common-noise parameter
        D       % Common-noise matrix
        A       % NxN adjacency matrix (fixed for all t) 
    end

    methods
        function obj = model(N, alpha, sig)
            % Initialise variables
            obj.dt = 1e-3;
            obj.t = 0:obj.dt:1;
            obj.N = N;
            obj.x0 = zeros(obj.N, 1); 
            obj.x = zeros(length(obj.t), obj.N);
            obj.xbar = zeros(length(obj.t), 1);
            obj.alpha = alpha;
            obj.sig = sig;
            obj.f = @(t, x) alpha.*(mean(x) - x);
            obj.g = @(t, x) sig;
            obj.opts = sdeset('RandSeed', 10);
            obj.eta = -0.7;
        end

        function obj = common_noise(obj, rho)
            % Include common noise
            obj.opts = sdeset('RandSeed', 1, 'Diagonal', 'no');
            obj.rho = rho;
            obj.D = horzcat(obj.rho*ones(obj.N, 1), sqrt(1-obj.rho^2)*eye(obj.N));
            obj.g = @(t, x) obj.sig .* obj.D;
        end

        function obj = erdosrenyi(obj, p)
            % p is probability of aij=aji=1
            obj.A = rand(obj.N) < p;
            % Make A symmetric
            obj.A = triu(obj.A, 1);
            obj.A = eye(obj.N) + obj.A + obj.A';
        end

        function obj = star(obj, n1, n2)
            % The nodes n1:n2 are fully-connected
            obj.A = eye(obj.N);
            obj.A(n1:n2, :) = 1;
            obj.A(:, n1:n2) = 1;
        end

        function obj = random_star(obj, p)
            % Generate star-network, with random nodes being fully-connected
            idx = find(rand(obj.N, 1) < p);
            obj.A = eye(obj.N);
            obj.A(idx, :) = 1;
            obj.A(:, idx) = 1;
        end

        function obj = homophily(obj, threshold)
            function values = f(~, x)
                A = pdist2(x, x);
                A(A <= threshold) = 1;
                A(A > threshold) = 0;
                % Initialise values
                values = NaN .* zeros(obj.N, 1);
                % Compute for j = 1,...,N
                for j = 1:obj.N
                    values(j) = 1/obj.N .* A(j, :) * (x - x(j));
                end
            end
            obj.f = @(t, x) obj.alpha/obj.N .* f(t, x);
        end

        function obj = heterophily(obj, threshold)
            function values = f(~, x)
                A = pdist2(x, x);
                A(A <= threshold) = 0;
                A(A > threshold) = 1;
                A = A + eye(obj.N);
                % Initialise values
                values = NaN .* zeros(obj.N, 1);
                % Compute for j = 1,...,N
                for j = 1:obj.N
                    values(j) = 1/obj.N .* A(j, :) * (x - x(j));
                end
            end
            obj.f = @(t, x) obj.alpha/obj.N .* f(t, x);
        end

        function obj = network(obj)
            % Drift function for when obj.A is fixed
            obj.f = @(t, x) obj.alpha/obj.N .* (obj.A*x - x.*sum(obj.A)');
        end

        function obj = sde_euler(obj)
            % Approximate SDE using Euler-Maruyama
            obj.x = sde_euler(obj.f, obj.g, obj.t, obj.x0, obj.opts);
        end

        function plot_trajectory(obj, opt)
            % Plot the trajectories of X and X_bar
            arguments
                obj                
                opt {mustBeNonempty} = true % Flag to plot individual trajectories
            end

            obj.xbar = mean(obj.x, 2);
            figure;

            if opt
                plot(obj.t, obj.x, 'k');
                hold on
            end
            % Average Line
            plot(obj.t, obj.xbar, 'green', 'LineWidth', 2)
            hold on
            % Default Line
            plot(obj.t, obj.eta .* ones(length(obj.t), 1), 'r', 'LineWidth', 2)
            hold off
            xlabel('t'); ylabel('X_{t}');
        end

        function plot_loss(obj, reps)
            % Plot the loss distribution under repeated simulation
            arguments
                obj                
                reps {mustBeNonempty} = 500 % 
            end

            repeats = reps;
            n_defaults = zeros(repeats, 1);

            if isempty(obj.D)
                opt = sdeset();
            else
                opt = sdeset('Diagonal', 'no');
            end

            % Loop
            for rep = 1:repeats
               y = sde_euler(obj.f, obj.g, obj.t, obj.x0, opt);
               n_defaults(rep) = sum(min(y) <= -0.7);
            end

            % Plot number of defaults
            tbl = tabulate(n_defaults);
            figure;
            plot(tbl(:, 1), tbl(:, 3)/100);
            title("Loss distribution")
            xlabel('Number of defaults'); ylabel('Probability of number of defaults');
        end

    end
end