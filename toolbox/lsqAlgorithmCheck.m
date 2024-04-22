function [boundsOnlySolvers,optimgetFlag,LB,UB,A,B,Aeq,Beq,NONLCON,options,varArg] = ...
    lsqAlgorithmCheck(solver,LB,UB,A,B,Aeq,Beq,NONLCON,options,defaultopt,numInputs,varargin)

% Bound-only API (V1) and general constrained API (V2) are:
% lsqnonlin(FUN,xCurrent,LB,UB,options,varargin) - V1
% lsqnonlin(FUN,xCurrent,LB,UB,A      ,B,Aeq,Beq,NONLCON,options) - V2
%
% This utility checks the arguments of lsq* functions from V1 API and
% ensures the arguments are upgraded to V2.
%
% numInputs counts only FUN,X0,LB,UB,A,B,Aeq,Beq,NONLCON,options, and varargin (V1) and
% NOT XDATA, YDATA (common utility to lsq*).

%   Copyright 2022 The MathWorks, Inc.

% Two legacy solvers
boundsOnlyAlgos = {'trust-region-reflective', 'levenberg-marquardt'};
% One new solver (if we decide to support SQP, this is the only place we need to
% update other than optimoptions)
constrainedAlgos = {'interior-point'};

varArg = {};
if numInputs == 1 % Problem struct
    % For a problem struct, we can't count inputs since all fields are
    % always present. Also, there is no varargin case to muddy the picture
    % Check:
    % 1) What algorithm was chosen, if any?
    if ~isempty(options) && (isstruct(options) || ...
            isa(options,'optim.options.SolverOptions'))
        % Use bounds-only solvers if
        boundsOnlySolvers = ~isfield(options, 'Algorithm') || ... % Algorithm not there OR
                        isfield(options, 'Algorithm') && ...      % Algorithm is there AND
                        (isempty(options.Algorithm) || ...        % it's empty OR
               any(strcmpi(options.Algorithm, boundsOnlyAlgos))); % it's set to a bounds only alg
    else
       % 2) If no options, check if linear/nonlinear constraints are given
        boundsOnlySolvers = isempty(A) && isempty(B) && ...
            isempty(Aeq) && isempty(Beq) && isempty(NONLCON);
    end
elseif numInputs <= 5 % && numInputs >= 2
    % Fifth argument (A) is options (empty is okay)
    boundsOnlySolvers = true;
    % Upgrade to V2; other args are already []
    options = A;
    A = [];
    % A wrong call for new API will have A (but no B) which is not possible to know so we
    % assume it is bound-only API (more likely)
elseif numInputs > 10
    boundsOnlySolvers = true;
    % Collect all the varargins (after 5th arg)
    varArg = [{B,Aeq,Beq,NONLCON,options}, varargin];
    varArg = varArg(1:numInputs-5);

    % Must upgrade to V2 API
    options = A;
    A = []; B = []; Aeq = []; Beq = []; NONLCON = [];

elseif numInputs == 10 && ...
        (isstruct(options) || isa(options,'optim.options.SolverOptions'))
    %10 inputs; tenth argument is struct/options (most common use case for the new API)
    % This also the case for PROBLEM struct input when we have exactly 10 inputs.

    if isstruct(A) || isa(A,'optim.options.SolverOptions')
        % 5th is also a struct. Must be the old API
        boundsOnlySolvers = true;
        varArg = {B,Aeq,Beq,NONLCON,options};
        varArg = varArg(1:numInputs-5);
        % Must upgrade to V2 API
        options = A;
        A = []; B = []; Aeq = []; Beq = []; NONLCON = [];
    elseif isfield(options, 'Algorithm') && ...
            (isempty(options.Algorithm) || ...
            any(strcmpi(options.Algorithm, boundsOnlyAlgos)))
        % Have algo set to TRR/LM
        boundsOnlySolvers = true;
        % No need to update arg list for this case
    elseif ~isfield(options, 'Algorithm')
        % Old struct-based syntax may not have Algorithm
        boundsOnlySolvers = true;
    else
        % Assume new API
        boundsOnlySolvers = false;
    end
elseif numInputs > 5 && numInputs <= 10
    % Either this is still an old API with varargin or constraints are given but no options.

    if isstruct(A) || isa(A,'optim.options.SolverOptions')
        % Fifth arg is an option
        boundsOnlySolvers = true;
        varArg = [{B,Aeq,Beq,NONLCON,options} ,varargin];
        varArg = varArg(1:numInputs-5);
        % Must upgrade to V2 API
        options = A;
        A = []; B = []; Aeq = []; Beq = []; NONLCON = [];
    else
        % Default to the new API and have user upgrade the call to V2
        boundsOnlySolvers = false;
    end
end

throwIncompatibleAlgorithmWarning = true;
% Set Algorithm if not specified and optimgetFlag
if isempty(options)
    % Options not set. Just set the default algo (needed later)
    options.Algorithm = defaultopt.Algorithm;
    optimgetFlag = 'fast';
    if ~boundsOnlySolvers
        % Change the default choice to the new algorithm
        options.Algorithm = constrainedAlgos{1}; % First choice is the default.
    end
    throwIncompatibleAlgorithmWarning = false;
elseif isa(options, 'struct') 
    optimgetFlag = 'fast';
    if ~isfield(options, 'Algorithm')
        % Algorithm field is not present so use the default
        if ~boundsOnlySolvers
            % Change the default choice to the new algorithm
            options.Algorithm = constrainedAlgos{1}; % First choice is the default.
        else
            options.Algorithm = boundsOnlyAlgos{1}; % First choice is the default.
        end
        throwIncompatibleAlgorithmWarning = false;
    end
elseif isa(options, 'optim.options.SolverOptions') 
    optimgetFlag = 'optimoptions';
    if ~isprop(options, 'Algorithm')
        % options is from a different solver, convert to lsq*
        options = optimoptions(solver,options);
    end
else
    errid = sprintf('optimlib:%s:InvalidOptions',lower(solver));
    error(errid, getString(message('optimlib:commonMsgs:InvalidOptions')));
end

% Check algorithm choice alone and ALLOW new API for just bounded problems too.
if isa(options, 'optim.options.SolverOptions') && ...
        isprop(options, 'Algorithm') && any(strcmpi(options.Algorithm, constrainedAlgos))
    boundsOnlySolvers = false;
elseif isa(options, 'struct') && ...
        isfield(options, 'Algorithm') && any(strcmpi(options.Algorithm, constrainedAlgos))
    boundsOnlySolvers = false;
end

if isa(options, 'optim.options.SolverOptions') && ~isSetByUser(options, 'Algorithm')
    throwIncompatibleAlgorithmWarning = false;
end

% If the argument list detects new API (boundsOnlySolvers = false) but the algorithm selected is
% legacy, it is fine. ALLOW legacy solvers for bounds only but new API.
if ~boundsOnlySolvers && any(strcmpi(options.Algorithm, boundsOnlyAlgos)) && ...
        (isempty(A) && isempty(B) && isempty(Aeq) && isempty(Beq) && isempty(NONLCON))
    boundsOnlySolvers = true;
end

% If using the legacy solvers, no constraints are allowed so we warn and switch.
if (isstruct(options) || isa(options,'optim.options.SolverOptions')) && ... 
    any(strcmpi(options.Algorithm, boundsOnlyAlgos)) && ...
    (~isempty(A) || ~isempty(B) || ~isempty(Aeq) || ~isempty(Beq) || ~isempty(NONLCON))
    % Override algorithm. 
    options.Algorithm = constrainedAlgos{1}; % First choice is the default.
    boundsOnlySolvers = false;
    
    if throwIncompatibleAlgorithmWarning
        % and warn for the wrong algorithm choice if set by user.
        warning(['optimlib:', solver, ':IncompatibleAlgorithm'],...
            getString(message('optimlib:lsqnonlin:IncompatibleAlgorithm','interior-point')));
    end
end
