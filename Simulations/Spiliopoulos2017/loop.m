%% Parameters
dt = 1e-2;     % Step size 0.01
t = 0:dt:1;    % 1x101 time vector from 0 to 1
N = 10;        % Size of Y
y0 = zeros(N, 1);         % Preallocate array
alpha = [repelem(1, 2, 1); repelem(10, 5, 1); repelem(100, 3, 1)];
sig = [repelem(2, 2, 1); repelem(1, 5, 1); repelem(0.5, 3, 1)];
eta = -0.7 .* ones(length(t), 1);   % Default level

f = @(t, y) alpha.*(mean(y) - y); 
g = @(t, y) sig;

repeats = 500;
n_defaults = zeros(repeats, 1);
%% Loop

for rep = 1:repeats
   % Specify Ito, set random seed
   opts = sdeset('RandSeed', rep);  
    
   y = sde_euler(f, g, t, y0, opts);
   n_defaults(rep) = sum(min(y) <= -0.7);
end

%% Plot
tbl = tabulate(n_defaults);
plot(tbl(:, 1), tbl(:, 3)/100)