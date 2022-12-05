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
        A       % NxN adjancency matrix 
    end

    methods
        function obj = model(alpha, sig)
            % Initialise variables
            obj.dt = 1e-3;
            obj.t = 0:obj.dt:1;
            obj.N = 10;
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
            obj.A = rand(obj.N, obj.N) < p;
            % Make A symmetric
            obj.A = triu(obj.A, 1);
            obj.A = obj.A + obj.A';
        end

        function obj = network(obj)
            obj.f = @(t, x) obj.alpha/obj.N .* (obj.A*x - x.*sum(obj.A)');
        end

        function obj = sde_euler(obj)
            % Approximate SDE using Euler-Maruyama
            obj.x = sde_euler(obj.f, obj.g, obj.t, obj.x0, obj.opts);
        end

        function plot_trajectory(obj)
            % Plot the trajectory of X and X_bar
            obj.xbar = mean(obj.x, 2);
            figure; 
            plot(obj.t, obj.x, 'k');
            hold on
            % Average Line
            plot(obj.t, obj.xbar, 'green', 'LineWidth', 2)
            hold on
            % Default Line
            plot(obj.t, obj.eta .* ones(length(obj.t), 1), 'r', 'LineWidth', 2)
            hold off
            xlabel('t'); ylabel('X_{t}');
        end

        function plot_loss(obj)
            % Plot the loss distribution under repeated simulation
            repeats = 500;
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