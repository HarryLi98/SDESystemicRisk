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
        values  % For computing sde_euler
        sims    % Array of sample paths - (N_T, N, reps) 
        probfail% Estimated probability of financial system failing
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

%         function obj = multiplicative_common_noise(obj, rho, gamma1, gamma2)
%             % Include multiplicative common noise, as in Giesecke
%             obj.opts = sdeset('RandSeed', 1, 'Diagonal', 'no');
%             obj.rho = rho;
%             % The power of the multiplicative noise makes a big difference.
%             % Here we use gamma = 0.5, but if we take gamma around 0.79 for
%             % common noise, it blows up.
%             obj.D = @(x) horzcat(obj.rho.*ones(obj.N, 1).*(exp(x).^gamma1), sqrt(1-obj.rho^2).*(exp(x).^gamma2).*eye(obj.N));
%             obj.g = @(t, x) obj.sig .* obj.D(x);
%         end

        function obj = multiplicative_common_noise(obj, rho, gamma1, gamma2)
            % Include multiplicative common noise, as in Giesecke
            obj.opts = sdeset('RandSeed', 1, 'Diagonal', 'no');
            obj.rho = rho;
            obj.D = @(x) horzcat(obj.rho.*ones(obj.N, 1).*(x.^gamma1), sqrt(1-obj.rho^2).*(x.^gamma2).*eye(obj.N));
            obj.g = @(t, x) obj.sig .* obj.D(x);
        end

        function obj = exp_noise(obj, rho)
            % Experiment with inverse multiplicative common noise
            % The lower the reserves, the more volative
            obj.opts = sdeset('RandSeed', 1, 'Diagonal', 'no');
            obj.rho = rho;
            obj.D = @(x) horzcat(obj.rho.*ones(obj.N, 1).*sqrt(exp(-x)), sqrt(1-obj.rho^2).*sqrt(exp(-x)).*eye(obj.N));
            obj.g = @(t, x) obj.sig .* obj.D(x);
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
            function values = v(~, x)
                obj.A = pdist2(x, x);
                obj.A(obj.A <= threshold) = 1;
                obj.A(obj.A > threshold) = 0;
                obj.values = (obj.alpha/obj.N) .* (obj.A*x - x.*sum(obj.A)');
                values = obj.values;
            end
            obj.f = @(t, x) v(t, x);
        end

        function obj = heterophily(obj, threshold)
            function values = v(~, x)
                obj.A = pdist2(x, x);
                obj.A(obj.A <= threshold) = 0;
                obj.A(obj.A > threshold) = 1;
                obj.values = (obj.alpha/obj.N) .* (obj.A*x - x.*sum(obj.A)');
                values = obj.values;
            end
            obj.f = @(t, x) v(t, x);
        end

        function obj = network(obj)
            % Drift function for when obj.A is fixed
            obj.f = @(t, x) obj.alpha/obj.N .* (obj.A*x - x.*sum(obj.A)');
        end

        % Function that removes bank from network if it defaults
        function values = update_network(obj, x, threshold)
            idx = x <= threshold;
            obj.A(idx, :) = 0;
            obj.A(:, idx) = 0;
            obj.values = (obj.alpha/obj.N) .* (obj.A*x - x.*sum(obj.A)');
            values = obj.values;
        end

        function obj = adaptive_network(obj, threshold)
            obj.f = @(t, x) update_network(obj, x, threshold);
        end

        % Function to remove systemically important banks from the network
        function values = remove_systemic_banks(obj, x, threshold)
            n_systemic = sum(obj.alpha == obj.alpha(1));
            idx = x <= threshold;
            idx = idx(1:n_systemic);
            obj.A(idx, :) = 0;
            obj.A(:, idx) = 0;
            obj.values = (obj.alpha/obj.N) .* (obj.A*x - x.*sum(obj.A)');
            values = obj.values;
        end

        function obj = adaptive_network_2(obj, threshold)
            obj.f = @(t, x) remove_systemic_banks(obj, x, threshold);
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

        function obj = monte_carlo(obj, reps)
            % Simulate the sample paths repeatedly
            arguments
                obj
                reps {mustBeNonempty} = 500
            end
            
            simulations = zeros(length(obj.t), obj.N, reps);

            if isempty(obj.D)
                opt = sdeset();
            else
                opt = sdeset('Diagonal', 'no');
            end

            parfor rep = 1:reps
                simulations(:, :, rep) = sde_euler(obj.f, obj.g, obj.t, obj.x0, opt);
            end

            obj.sims = simulations;
        
            % Estimate the probability that the banking system fails
            % which occurs when obj.xbar <= obj.eta
            xb = squeeze(min(mean(obj.sims, 2), [], 1));
            obj.probfail = 1/reps * sum(xb <= obj.eta);
        end

        function plot_loss(obj)
            % Plot the loss distribution under repeated simulation
            arguments
                obj                
            end

            n_defaults = squeeze(sum(min(obj.sims, [], 1) <= obj.eta, 2));
            tbl = tabulate(n_defaults);
            % Pad the table to include zero if no banks default
            if tbl(1, 1) > 0
                tbl = [zeros(1, 3); tbl];
            end

            % Plot number of defaults
            
            figure;
            n_banks = 0:obj.N;
            default_rate = tbl(:, 3)/100;
            if numel(default_rate) < numel(n_banks)
                default_rate(numel(n_banks)) = 0;
            end
            plot(n_banks, default_rate);
            title("Loss distribution")
            xlabel('Number of defaults'); ylabel('Probability');
        end

    end
end