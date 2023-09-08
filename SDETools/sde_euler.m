function [Y,W,TE,YE,WE,IE] = sde_euler(f,g,tspan,y0,options)

solver = 'SDE_EULER';

% Check inputs and outputs
if nargin < 5
    if nargin < 4
        error('SDETools:sde_euler:NotEnoughInputs',...
              'Not enough input arguments.  See %s.',solver);
    end
    if isa(y0,'struct')
        error('SDETools:sde_euler:NotEnoughInputsOptions',...
             ['An SDE options structure was provided as the last argument, '...
              'but one of the first four input arguments is missing.'...
              '  See %s.'],solver);
    end
    options = [];
elseif nargin == 5
    if isempty(options) && (~sde_ismatrix(options) ...
            || any(size(options) ~= 0) || ~(isstruct(options) ...
            || iscell(options) || isnumeric(options))) ...
            || ~isempty(options) && ~isstruct(options)
        error('SDETools:sde_euler:InvalidSDESETStruct',...
              'Invalid SDE options structure.  See SDESET.');
    end
else
    error('SDETools:sde_euler:TooManyInputs',...
          'Too many input arguments.  See %s.',solver);
end

% Handle solver arguments (NOTE: ResetStream is called by onCleanup())
[N,D,tspan,tdir,lt,y0,fout,gout,dgout,dg,h,ConstStep,dataType,NonNegative,...
    idxNonNegative,DiagonalNoise,ScalarNoise,OneDNoise,ConstFFUN,ConstGFUN,...
    ConstDGFUN,Stratonovich,RandFUN,ResetStream,EventsFUN,EventsValue,...
    OutputFUN,WSelect] = sdearguments(solver,f,g,tspan,y0,options);	%#ok<ASGLU>

% Initialize outputs for zero-crossing events
isEvents = ~isempty(EventsFUN);
if isEvents
    if nargout > 6
        error('SDETools:sde_euler:EventsTooManyOutputs',...
              'Too many output arguments.  See %s.',solver);
    else
        if nargout >= 3
            TE = [];
            if nargout >= 4
                YE = [];
                if nargout >= 5
                    WE = [];
                    if nargout == 6
                        IE = [];
                    end
                end
            end
        end
    end
else
    if nargout > 2
        if nargout <= 6
            error('SDETools:sde_euler:NoEventsTooManyOutputs',...
                 ['Too many output arguments. An events function has not '...
                  'been specified.  See %s.'],solver);
        else
            error('SDETools:sde_euler:TooManyOutputs',...
                  'Too many output arguments.  See %s.',solver);
        end
    end
end

% Initialize output function
isOutput = ~isempty(OutputFUN);

% If drift or diffusion functions, FFUN and GFUN, exist
isDrift = ~(ConstFFUN && isscalar(fout) && fout == 0);
isDiffusion = ~(ConstGFUN && isscalar(gout) && gout == 0);

isW = isDiffusion && (isEvents || WSelect);
if isW
    Wi = 0;
else
    Wi = [];
end

% Is Y allocated and output
isYOutput = (nargout > 0);

% Check if alternative RandFUN function or W matrix is present
if isempty(RandFUN) && isfield(options,'RandFUN')
    CustomRandFUN = isa(options.RandFUN,'function_handle');
    CustomWMatrix = ~CustomRandFUN;
else
    CustomRandFUN = false;
    CustomWMatrix = false;
end

% Location of stored Wiener increments, if they are needed and pre-calculated
dWinY = (D <= N && isYOutput && ~CustomWMatrix || ~isDiffusion);  	% Store in Y
dWinW = (isDiffusion && D > N && nargout >= 2 || CustomWMatrix);	% Store in W

% Allocate state array, Y, if needed (may be allocated in place below)
if isYOutput && (~(CustomRandFUN && dWinY) ...
        || ~(ConstFFUN && ConstGFUN && dWinW && ~NonNegative))
    Y(lt,N) = cast(0,dataType);
end
    
% Calculate Wiener increments from normal variates, store in Y if possible, or W
sh = tdir*sqrt(h);
h = tdir*h;
if isDiffusion || isW
    if CustomRandFUN                            % Check alternative RandFUN
        if CustomWMatrix
            W = sdeget(options,'RandFUN',[],'flag');
            if ~isfloat(W) || ~sde_ismatrix(W) || any(size(W) ~= [lt D])
                error('SDETools:sde_euler:RandFUNInvalidW',...
                     ['RandFUN must be a function handle or a '...
                      'LENGTH(TSPAN)-by-D (%d by %d) floating-point matrix '...
                      'of integrated Wiener increments.  See %s.'],lt,D,solver);
            end
            error('SDETools:sde_euler:RandFUNWMatrixNotSupportedYet',...
                 ['Custom W matrices specified via RandFUN not supported '...
                  'yet in SDE_EULER and SDE_MILSTEIN.']);
        else
            % User-specified function handle
            RandFUN = sdeget(options,'RandFUN',[],'flag');
            
            try
                if dWinY                        % Store Wiener increments in Y
                    Y = feval(RandFUN,lt-1,D);
                    if ~sde_ismatrix(Y) || isempty(Y) || ~isfloat(Y)
                        error('SDETools:sde_euler:RandFUNNot2DArray3',...
                             ['RandFUN must return a non-empty matrix of '...
                              'floating-point values.  See %s.'],solver);
                    end
                    [m,n] = size(Y);
                    if m ~= lt-1 || n ~= D
                        error('SDETools:sde_euler:RandFUNDimensionMismatch3',...
                             ['The specified alternative RandFUN did not '...
                              'output a %d by %d matrix as requested.  '...
                              'See %s.'],lt-1,D,solver);
                    end

                    if ScalarNoise || ConstStep
                        Y = [zeros(1,D,dataType);
                             sh.*Y zeros(lt-1,N-D,dataType)];
                    else
                        Y = [zeros(1,D,dataType);
                             bsxfun(@times,sh,Y) zeros(lt-1,N-D,dataType)];
                    end
                    if nargout >= 2
                        W = cumsum(Y(:,1:D),1); % Integrated Wiener increments
                    end
                elseif dWinW                    % Store Wiener increments in W
                    W = feval(RandFUN,lt-1,D);
                    if ~sde_ismatrix(W) || isempty(W) || ~isfloat(W)
                        error('SDETools:sde_euler:RandFUNNot2DArray1',...
                             ['RandFUN must return a non-empty matrix of '...
                              'floating-point values.  See %s.'],solver);
                    end
                    [m,n] = size(W);
                    if m ~= lt-1 || n ~= D
                        error('SDETools:sde_euler:RandFUNDimensionMismatch1',...
                             ['The specified alternative RandFUN did not '...
                              'output a %d by %d matrix as requested.  '...
                              'See %s.'],lt-1,D,solver);
                    end

                    if ConstStep
                        W = [zeros(1,D,dataType);sh.*W];
                    else
                        W = [zeros(1,D,dataType);bsxfun(@times,sh,W)];
                    end
                else                            % Cannot store Wiener increments
                    dW = feval(RandFUN,1,D);
                    if ~isvector(dW) || isempty(dW) || ~isfloat(dW)
                        error('SDETools:sde_euler:RandFUNNot2DArray2',...
                             ['RandFUN must return a non-empty matrix of '...
                              'floating-point values.  See %s.'],solver);
                    end
                    [m,n] = size(dW);
                    if m ~= 1 || n ~= D
                        error('SDETools:sde_euler:RandFUNDimensionMismatch2',...
                             ['The specified alternative RandFUN did not '...
                              'output a 1 by %d column vector as '...
                              'requested.  See %s.'],D,solver);
                    end

                    dW = sh(1)*dW;
                    if ConstStep
                        RandFUN = @(i)sh*feval(RandFUN,1,D);
                    else
                        RandFUN = @(i)sh(i)*feval(RandFUN,1,D);
                    end
                end
            catch err
                switch err.identifier
                    case 'MATLAB:TooManyInputs'
                        error('SDETools:sde_euler:RandFUNTooFewInputs',...
                             ['RandFUN must have at least two inputs.  '...
                              'See %s.'],solver);
                    case 'MATLAB:TooManyOutputs'
                        error('SDETools:sde_euler:RandFUNNoOutput',...
                             ['The output of RandFUN was not specified. '...
                              'RandFUN must return a non-empty matrix.  '...
                              'See %s.'],solver);
                    case 'MATLAB:unassignedOutputs'
                        error('SDETools:sde_euler:RandFUNUnassignedOutput',...
                             ['The first output of RandFUN was not '...
                              'assigned.  See %s.'],solver);
                    case 'MATLAB:minrhs'
                        error('SDETools:sde_euler:RandFUNTooManyInputs',...
                             ['RandFUN must not require more than two '...
                              'inputs.  See %s.'],solver);
                    otherwise
                        rethrow(err);
                end
            end
        end
    else
        % No error checking needed if default RANDN used
        if dWinY                            % Store Wiener increments in Y
            if ScalarNoise || ConstStep
                Y(2:end,1:D) = sh.*feval(RandFUN,lt-1,D);
            else
                Y(2:end,1:D) = bsxfun(@times,sh,feval(RandFUN,lt-1,D));
            end
            if nargout >= 2
                W = cumsum(Y(:,1:D),1);     % Integrated Wiener increments
            end
        elseif dWinW                        % Store Wiener increments in W
            if ConstStep
                W = [zeros(1,D,dataType);sh.*feval(RandFUN,lt-1,D)];
            else
                W = [zeros(1,D,dataType);...
                     bsxfun(@times,sh,feval(RandFUN,lt-1,D))];
            end
        else                                % Unable to store Wiener increments
            if ConstStep
                RandFUN = @(i)sh*feval(RandFUN,1,D);
            else
                RandFUN = @(i)sh(i)*feval(RandFUN,1,D);
            end
            dW = RandFUN(1);
        end
    end
elseif ~isDiffusion && nargout >= 2
    W = zeros(lt,0,dataType);
end

% Integrate
if ConstFFUN && ConstGFUN && ((~isDiffusion && isYOutput) || dWinY || dWinW) ...
        && ~NonNegative
    % No FOR loop needed
    if dWinY
     	Y = cumsum(Y,1);                    % Integrate Wiener increments in Y
        if isW
            W = Y(:,1:D);                   % If needed for events or output
        end
        if isDrift
            if OneDNoise                    % 1-D scalar (and diagonal) noise
                Y = y0+tspan*fout+Y*gout;
            elseif ScalarNoise
                Y = bsxfun(@plus,y0.',bsxfun(@plus,tspan*fout.',Y(:,1)*gout.'));
            elseif DiagonalNoise
                Y = bsxfun(@plus,y0.',tspan*fout.'+bsxfun(@times,Y,gout.'));
            else
                Y = bsxfun(@plus,y0.',tspan*fout.'+Y(:,1:D)*gout.');
            end
        else
            if OneDNoise                    % 1-D scalar (and diagonal) noise
                Y = y0+Y*gout;
            elseif DiagonalNoise
                Y = bsxfun(@plus,y0.',bsxfun(@times,Y,gout.'));
            else
                Y = bsxfun(@plus,y0.',Y(:,1:D)*gout.');
            end
        end
    elseif dWinW
        W = cumsum(W,1);                    % Integrate Wiener increments in W
        if isDrift
            Y = bsxfun(@plus,y0.',tspan*fout.'+W*gout.');
        else
            Y = bsxfun(@plus,y0.',W*gout.');
        end
    else
        if isW                              % If needed for events or output
            if strcmp(dataType,'double')
                W(lt,D) = 0;
            else
                W(lt,D) = single(0);
            end
        end
        if isDrift
            if N == 1
                Y = y0+tspan*fout;
            else
                Y = bsxfun(@plus,y0.',tspan*fout.');
            end
        else
            Y = ones(lt,1,dataType)*y0.';
        end
    end
    
    % Check for and handle zero-crossing events, and output function
    if isEvents
        for i = 2:lt
            [te,ye,we,ie,EventsValue,IsTerminal] ...
                = sdezero(EventsFUN,tspan(i),Y(i,:).',W(i,:).',EventsValue);
            if ~isempty(te)
                if nargout >= 3
                    TE = [TE;te];               %#ok<AGROW>
                    if nargout >= 4
                        YE = [YE;ye];           %#ok<AGROW>
                        if nargout >= 5
                            WE = [WE;we];       %#ok<AGROW>
                            if nargout == 6
                                IE = [IE;ie];	%#ok<AGROW>
                            end
                        end
                    end
                end
                if IsTerminal
                    Y = Y(1:i,:);
                    if nargout >= 2
                        W = W(1:i,:);
                    end
                    break;
                end
            end
            
            if isOutput
                OutputFUN(tspan(i),Y(i,:).','',W(i,:).');
            end
        end
    elseif isOutput
        if isW
            for i = 2:lt
                OutputFUN(tspan(i),Y(i,:).','',W(i,:).');
            end
        else
            for i = 2:lt
                OutputFUN(tspan(i),Y(i,:).','',[]);
            end
        end
    end
else
    dt = h(1);                              % Fixed step size
    Ti = tspan(1);                          % Current time
  	Yi = y0;                                % Set initial conditions
    
    if OneDNoise
        % Optimized 1-D scalar (and diagonal) noise case
        if isYOutput
            Y(1) = Yi;                      % Store initial conditions
        end
        
        % Integration loop using Wiener increments stored in Y(i+1)
        for i = 1:lt-1
            if ~ConstStep
                dt = h(i);                  % Step size
            end
            if dWinY
                dW = Y(i+1);                % Wiener increment
            elseif i > 1
                dW = RandFUN(i);            % Generate Wiener increment
            end
            
            % Calculate next time step
            if ConstGFUN
              	Yi = Yi+f(Ti,Yi)*dt+gout*dW;
            else
                if ~ConstFFUN
                    fout = f(Ti,Yi)*dt;
                end
                gout = g(Ti,Yi);
                
                if Stratonovich             % Use Euler-Heun step
                    Yi = Yi+fout+0.5*(gout+g(Ti,Yi+gout*dW))*dW;
                else
                    Yi = Yi+fout+gout*dW;
                end
            end
            
            % Force solution to be >= 0
            if NonNegative
                Yi = abs(Yi);
            end
            
            Ti = tspan(i+1);                % Increment current time
            if isYOutput
                Y(i+1) = Yi;              	% Store solution
            end
            
            % Integrated Wiener increments for events and output functions
            if isW
                if nargout >= 2
                    Wi = W(i+1);            % Use stored W
                else
                    Wi = Wi+dW;             % Integrate Wiener increment
                end
            end
            
            % Check for and handle zero-crossing events
            if isEvents
                [te,ye,we,ie,EventsValue,IsTerminal] ...
                    = sdezero(EventsFUN,Ti,Yi,Wi,EventsValue);
                if ~isempty(te)
                    if nargout >= 3
                        TE = [TE;te];               %#ok<AGROW>
                        if nargout >= 4
                            YE = [YE;ye];           %#ok<AGROW>
                            if nargout >= 5
                                WE = [WE;we];       %#ok<AGROW>
                                if nargout == 6
                                    IE = [IE;ie];	%#ok<AGROW>
                                end
                            end
                        end
                    end
                    if IsTerminal
                        Y = Y(1:i+1);
                        if nargout >= 2
                            W = W(1:i+1);
                        end
                        break;
                    end
                end
            end
            
            % Check for and handle output function
            if isOutput
                OutputFUN(Ti,Yi,'',Wi);
            end
        end
    else
        % General case
        if isYOutput
            Y(1,:) = Yi;                	% Store initial conditions
        end
        
        % Integration loop using cached state, Yi, and Wiener increments, dW
        for i = 1:lt-1
            if ~ConstStep
                dt = h(i);                  % Variable step size
            end
            if dWinY
                dW = Y(i+1,1:D);            % Wiener increments stored in Y
            elseif dWinW
                dW = W(i+1,:);              % Wiener increments stored in W
                W(i+1,:) = W(i,:)+dW;       % Integrate Wiener increments
            elseif i > 1
                dW = RandFUN(i);            % Generate Wiener increments
            end
            dW = dW(:);                     % Wiener increments
            
            % Calculate next time step
            if ConstGFUN
                if DiagonalNoise
                    Yi = Yi+f(Ti,Yi)*dt+gout.*dW;
                else
                    Yi = Yi+f(Ti,Yi)*dt+gout*dW;
                end
            else
                if ~ConstFFUN
                    fout = f(Ti,Yi)*dt;
                end
                gout = g(Ti,Yi);

                if Stratonovich	% Use Euler-Heun step
                    if DiagonalNoise
                        Yi = Yi+fout+0.5*(gout+g(Ti,Yi+gout.*dW)).*dW;
                    else
                        Yi = Yi+fout+0.5*(gout+g(Ti,Yi+gout*dW))*dW;
                    end
                else
                    if DiagonalNoise
                        Yi = Yi+fout+gout.*dW;
                    else
                        Yi = Yi+fout+gout*dW;
                    end
                end
            end
            
            % Force specified solution indices to be >= 0
            if NonNegative
                Yi(idxNonNegative) = abs(Yi(idxNonNegative));
            end
            
            Ti = tspan(i+1);                % Increment current time
            if isYOutput
                Y(i+1,:) = Yi;           	% Store solution
            end
            
            % Integrated Wiener increments for events and output functions
            if isW
                if nargout >= 2
                    Wi = W(i+1,:);          % Use stored W
                    Wi = Wi(:);
                else
                    Wi = Wi+dW;             % Integrate Wiener increments
                end
            end
            
            % Check for and handle zero-crossing events
            if isEvents
                [te,ye,we,ie,EventsValue,IsTerminal] ...
                    = sdezero(EventsFUN,Ti,Yi(:),Wi,EventsValue);
                if ~isempty(te)
                    if nargout >= 3
                        TE = [TE;te];               %#ok<AGROW>
                        if nargout >= 4
                            YE = [YE;ye];           %#ok<AGROW>
                            if nargout >= 5
                                WE = [WE;we];       %#ok<AGROW>
                                if nargout == 6
                                    IE = [IE;ie];	%#ok<AGROW>
                                end
                            end
                        end
                    end
                    if IsTerminal
                        Y = Y(1:i+1,:);
                        if nargout >= 2
                            W = W(1:i+1,:);
                        end
                        break;
                    end
                end
            end
            
            % Check for and handle output function
            if isOutput
                OutputFUN(Ti,Yi(:),'',Wi);
            end
        end
    end
end

% Finalize output
if isOutput
    OutputFUN([],[],'done',[]);
end